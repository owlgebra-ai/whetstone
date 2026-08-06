"""Stage-C segment-routed DAPO trainer — runs on **spark** (packet P7 §6).

Topology (P7 §4, validated by activity 010 finding 7): the trainer lives on
spark because it is the only box where the packet's numerics fit — turing OOMs
on fp32 AdamW for this model, spark runs it at 34.7 GB of 128. Rollouts come
from turing over the `/data` bus (:mod:`whetstone.rollout_bus`).

**Deviation from P7 §4, attested:** the design's third server — a frozen π_0
anchor on ``turing:8002`` serving ``prompt_logprobs`` — is replaced by a
**resident frozen π_0 on the trainer**. Activity 007 needed the separate server
because turing's 32 GB could not hold a second copy beside the optimizer state
(that is why ``precompute_pi0_cache.py`` exists); spark's 128 GB can, at 3.4 GB
in bf16. This removes a network round-trip per step and makes the packet's own
gotcha — "reloading :8002 with student weights corrupts the KL anchor silently"
— structurally impossible, since the anchor is loaded once from the original
checkpoint and never written to.

Per step:
  1. sample a batch from the saturation-paced curriculum
  2. request K rollouts per problem from turing; wait on the bus
  3. score strictly, build the Stage-C scalar reward per group
  4. dynamic sampling: drop all-correct and all-wrong groups (log the rate —
     it is the phase-exhaustion signal)
  5. group advantages → per-token routing with difficulty amplification
  6. forward policy + π_0, assemble ``policy − λ_TEA·L_TEA + λ_align·KL``
  7. optimizer step; publish weights every ``--sync_every`` steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.curriculum import Curriculum
from whetstone.dapo import (
    GroupStats,
    assert_masks_partition,
    compute_tea_weights,
    dynamic_sampling_keep,
    group_advantages,
    route_advantages,
    stagec_loss,
)
from whetstone.poolutil import read_jsonl
from whetstone.reward.stagec import (
    RolloutView,
    ThinkBudget,
    assert_invariants,
    compute_stagec_reward,
)
from whetstone.rollout_bus import RolloutBus, RolloutRequest

TOPK_ENTROPY = 512      # CurioSFT / entropy_audit convention — comparable numbers


# --- scoring a sequence under a model ---------------------------------------

def score_sequence(model, ids, prompt_len: int, chunk: int, want_entropy: bool):
    """Per-token ``logp`` of the realized tokens and top-512 entropy.

    Both are differentiable — TEA maximizes entropy through this graph, so a
    detached entropy would make the regularizer inert while its curve still
    moved.

    Memory (activity 009's Stage-B note): ``F.cross_entropy`` instead of
    ``log_softmax`` + ``gather``. Identical arithmetic, but the fused kernel
    recomputes in backward rather than saving an ``(N, 151936)`` fp32
    log-softmax — 1.7 GB of activation for a single long sequence. ``lm_head``
    is applied per position-chunk so full-vocab logits never exist for the whole
    sequence at once.

    Entropy at position ``t`` comes from the logits at ``t−1``; an off-by-one
    here mis-attributes tokens across the ``</think>`` boundary.
    """
    import torch
    import torch.nn.functional as F

    hidden = model.model(input_ids=ids).last_hidden_state       # [1, L, H]
    L = ids.shape[1]
    # Positions prompt_len-1 .. L-2 predict tokens prompt_len .. L-1.
    lo, hi = prompt_len - 1, L - 1
    logps, ents = [], []
    for s in range(lo, hi, chunk):
        e = min(s + chunk, hi)
        logits = model.lm_head(hidden[:, s:e, :]).float()        # [1, c, V]
        tgt = ids[:, s + 1:e + 1]                                # [1, c]
        lp = -F.cross_entropy(logits[0], tgt[0], reduction="none")
        logps.append(lp)
        if want_entropy:
            top = torch.topk(logits, TOPK_ENTROPY, dim=-1).values
            p = torch.softmax(top, dim=-1)
            ents.append(-(p * torch.log(p.clamp_min(1e-12))).sum(-1)[0])
            del top, p
        del logits
    logp = torch.cat(logps)
    ent = torch.cat(ents) if want_entropy else torch.zeros_like(logp)
    return logp, ent


# --- reward scoring for one group -------------------------------------------

def score_group(row: dict, budget: ThinkBudget) -> dict:
    """Rewards + diagnostics for one problem's K rollouts."""
    cands = row["candidates"]
    think_lens = [c["think_len"] for c in cands if c["g"] == 1]
    B = budget.effective_B(think_lens)

    breakdowns = []
    for c in cands:
        view = RolloutView(
            completion_text=c["text"], think_len=c["think_len"],
            answer_len=c["answer_len"], g=c["g"], gate_reason=c["gate_reason"],
        )
        breakdowns.append(compute_stagec_reward(view, row["ground_truth"], budget_B=B))
    budget.update(think_lens)
    return {"breakdowns": breakdowns, "B": B,
            "correct": [b.r_acc > 0 for b in breakdowns],
            "rewards": [b.total for b in breakdowns]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--init_model", required=True)
    ap.add_argument("--anchor_model", default="Qwen/Qwen3-1.7B",
                    help="frozen π_0 for the answer-segment KL — the ORIGINAL "
                         "checkpoint, never the student")
    ap.add_argument("--buckets", required=True, help="stagec_bucket.py buckets.jsonl")
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--problems_per_step", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=12288)
    ap.add_argument("--lambda_tea", type=float, default=0.05)
    ap.add_argument("--lambda_align", type=float, default=0.1)
    ap.add_argument("--tau_c", type=float, default=1.0)
    ap.add_argument("--tea_cap_c", type=float, default=100.0)
    ap.add_argument("--tea_scale", default="weighted_mean",
                    choices=["weighted_mean", "design_literal"])
    ap.add_argument("--b_init", type=float, default=0.0,
                    help="think budget B_0; 0 = median think length of the "
                         "first batch (measured, not guessed)")
    ap.add_argument("--b_floor", type=float, default=120.0)
    ap.add_argument("--b_anneal", type=float, default=0.995)
    ap.add_argument("--std_min", type=float, default=40.0)
    ap.add_argument("--sync_every", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--logit_chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout_timeout", type=float, default=3600.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--prefetch", action="store_true",
                    help="queue step N+1's rollouts before training on step N "
                         "so the two boxes overlap (010 finding 12)")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    os.makedirs(args.run_dir, exist_ok=True)
    inv = assert_invariants()       # fail fast on a reward mis-configuration
    print(f"[train] reward invariants OK: {inv}", flush=True)

    log_path = os.path.join(args.run_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    def log(rec: dict) -> None:
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    # --- curriculum --------------------------------------------------------
    pool = {r["_uid"]: r for r in read_jsonl(args.pool)}
    bucket_rows = read_jsonl(args.buckets)
    curric = Curriculum.from_bucket_rows(bucket_rows, pool)
    print(f"[train] curriculum: {curric.stats()}", flush=True)
    if not curric.problems:
        raise SystemExit("curriculum is empty — no mixed groups in the bucket file")

    # --- models ------------------------------------------------------------
    dev = "cuda"
    t0 = time.time()
    policy = AutoModelForCausalLM.from_pretrained(
        args.init_model, dtype=torch.float32, trust_remote_code=True).to(dev)
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy.train()
    anchor = AutoModelForCausalLM.from_pretrained(
        args.anchor_model, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    print(f"[train] policy(fp32)+anchor(bf16) loaded in {time.time()-t0:.1f}s", flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    theta0 = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
    theta0_norm = float(theta0.norm())

    bus = RolloutBus(args.run_dir)
    budget = ThinkBudget(args.b_init or 1e9, b_floor=args.b_floor,
                         anneal=args.b_anneal, std_min=args.std_min)
    rng = random.Random(args.seed)
    weights_version = 0
    posted: set = set()

    def _bf16_state_dict() -> dict:
        """A bf16 **copy** for serving/checkpointing. The live weights stay fp32.

        ``policy.to(torch.bfloat16)`` mutates parameters in place, and casting
        back to fp32 does **not** restore the discarded mantissa bits. At LR 1e-6
        an Adam step moves a weight by ~1e-6 while bf16's quantum near a typical
        weight magnitude is ~1e-4, so every export would silently erase all
        learning accumulated since the previous one — activity 007's documented
        "the update rounds to zero silently" failure, re-entering through the
        serialization path instead of the optimizer. Copy, never cast in place.
        """
        return {k: v.detach().to(torch.bfloat16)
                for k, v in policy.state_dict().items()}

    def _theta_norm() -> float:
        with torch.no_grad():
            return float(torch.cat([p.detach().flatten()
                                    for p in policy.parameters()]).norm())

    def export_weights(version: int) -> None:
        nonlocal weights_version
        stage = os.path.join(args.run_dir, f"_export_v{version:06d}")
        os.makedirs(stage, exist_ok=True)
        before = _theta_norm()
        policy.save_pretrained(stage, state_dict=_bf16_state_dict())
        after = _theta_norm()
        # Direct guard against the in-place-cast bug above: exporting must not
        # move the live weights at all.
        if before != after:
            raise AssertionError(
                f"export mutated the live policy: |θ| {before!r} -> {after!r}. "
                "The bf16 export must be a copy; casting in place destroys the "
                "fp32 master weights (activity 007)."
            )
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.init_model).save_pretrained(stage)
        bus.publish_weights(stage, version)
        bus.prune_weights(keep_versions=2)
        weights_version = version
        print(f"[train] published weights v{version} "
              f"(live |θ| unchanged at {after:.6f})", flush=True)

    # The worker must be generating from the checkpoint we are training. If it
    # already is (launched with the same --init_model), publishing a v1 that is
    # byte-for-byte the same policy just costs a ~55 s engine rebuild, so check
    # first. If it is serving something else, that is a silent-corruption bug —
    # the whole run would compute importance ratios against the wrong behaviour
    # policy — so publish and force the swap.
    ws = bus.worker_status()
    if ws.get("state") == "never started":
        print("[train] WARNING: no worker heartbeat yet; publishing v1 to be safe",
              flush=True)
        export_weights(1)
    elif os.path.realpath(ws.get("model", "")) != os.path.realpath(args.init_model):
        print(f"[train] worker is serving {ws.get('model')!r}, not {args.init_model!r}"
              " — publishing v1 to force a swap", flush=True)
        export_weights(1)
    else:
        print(f"[train] worker already on {args.init_model} (v{ws.get('version')}) "
              "— skipping the redundant initial publish", flush=True)

    # --- the loop ----------------------------------------------------------
    for step in range(1, args.steps + 1):
        step_t0 = time.time()

        def _post(n: int) -> bool:
            batch = curric.sample(args.problems_per_step, rng)
            if not batch:
                return False
            bus.post_request(RolloutRequest(
                step=n,
                items=[{"uid": p.uid, "prompt": p.prompt,
                        "ground_truth": p.ground_truth, "level": p.level,
                        "p_hat": p.effective_p_hat, "seen": p.seen}
                       for p in batch],
                params={"K": args.K, "temperature": args.temperature,
                        "top_p": args.top_p, "max_tokens": args.max_tokens,
                        "seed": args.seed},
                weights_version=weights_version))
            return True

        if step not in posted:
            if not _post(step):
                print("[train] curriculum exhausted — every problem retired",
                      flush=True)
                break
            posted.add(step)

        t_req = time.time()
        rows = bus.wait_response(step, timeout=args.rollout_timeout)
        t_rollout = time.time() - t_req

        # Prefetch: queue the NEXT batch before spending ~60 s on this one's
        # gradient, so turing generates while spark trains instead of the two
        # boxes taking turns. Activity 010 finding 12 measured the serialized
        # loop at gen 60 s / train 60 s — a perfectly balanced split, which is
        # exactly the case where prefetching nearly halves wall-clock. Costs a
        # one-step lag in the curriculum (step N+1's batch is drawn before
        # step N's `observe()` lands) and up to one extra sync of policy
        # staleness, both of which DAPO's clipping is built to tolerate.
        if args.prefetch and step + 1 <= args.steps and (step + 1) not in posted:
            if _post(step + 1):
                posted.add(step + 1)

        # --- reward + dynamic sampling -------------------------------------
        t_score0 = time.time()
        if args.b_init == 0 and step == 1:
            all_think = [c["think_len"] for r in rows for c in r["candidates"]
                         if c["g"] == 1]
            budget.B = float(statistics.median(all_think)) if all_think else 600.0
            print(f"[train] B_0 measured from batch 1: {budget.B:.0f} tokens", flush=True)

        kept, group_stats = [], []
        for row in rows:
            sc = score_group(row, budget)
            keep, reason = dynamic_sampling_keep(sc["correct"])
            n_ok = sum(sc["correct"])
            curric.observe(row["uid"], n_ok, len(sc["correct"]))
            rt = torch.tensor(sc["rewards"], dtype=torch.float32)
            adv = group_advantages(rt)
            group_stats.append(GroupStats(
                uid=row["uid"], n=len(sc["correct"]), n_correct=n_ok,
                reward_mean=float(rt.mean()), reward_std=float(rt.std(unbiased=False)),
                adv_abs_mean=float(adv.abs().mean()), kept=keep, drop_reason=reason,
                p_hat=n_ok / max(1, len(sc["correct"]))))
            if keep:
                kept.append((row, sc, adv))
        t_score = time.time() - t_score0

        n_drop = sum(1 for g in group_stats if not g.kept)
        if not kept:
            log({"step": step, "event": "all_groups_dropped",
                 "drop_rate": 1.0,
                 "reasons": [g.drop_reason for g in group_stats]})
            print(f"[train] step {step}: all {len(rows)} groups dropped "
                  f"({[g.drop_reason for g in group_stats]})", flush=True)
            continue

        # --- TEA weights: batch-scoped, computed BEFORE any forward ---------
        # Advantages are constant within a rollout, so centering them inside a
        # single-rollout micro-batch gives Cov ≡ 0 and a uniform softmax — TEA
        # degenerates to "add mean entropy" and selects nothing (010 finding 8).
        # Both inputs here are constants w.r.t. θ, so the weights can be built
        # once per step and handed to each micro-batch as a constant slice.
        micro: List[dict] = []
        for row, sc, adv in kept:
            for k, c in enumerate(row["candidates"]):
                comp = c["token_ids"]
                if not comp:
                    continue
                n_lp = min(len(comp), len(c["logp_old"]))
                tm = torch.zeros(1, n_lp)
                am = torch.zeros(1, n_lp)
                if c["think_end"] > c["think_start"]:
                    tm[0, c["think_start"]:min(c["think_end"], n_lp)] = 1.0
                if c["answer_end"] > c["answer_start"]:
                    am[0, c["answer_start"]:min(c["answer_end"], n_lp)] = 1.0
                if tm.sum() + am.sum() == 0:
                    continue
                assert_masks_partition(tm, am, [n_lp])
                lo = torch.tensor([c["logp_old"][:n_lp]], dtype=torch.float32)
                tok_adv = route_advantages(adv[k:k + 1], tm, row.get("p_hat", 0.5))
                micro.append({"row": row, "cand": c, "n": n_lp, "tmask": tm,
                              "amask": am, "logp_old": lo, "tok_adv": tok_adv})

        if not micro:
            log({"step": step, "event": "no_usable_rollouts"})
            continue

        tea_w, tea_stats = compute_tea_weights(
            torch.cat([m["logp_old"][m["tmask"] > 0] for m in micro]),
            torch.cat([m["tok_adv"][m["tmask"] > 0] for m in micro]),
            tau_c=args.tau_c, cap_c=args.tea_cap_c)
        policy_denom = float(sum(float((m["tmask"] + m["amask"]).sum()) for m in micro))
        kl_denom = float(sum(float(m["amask"].sum()) for m in micro)) or 1.0
        offs, off = [], 0
        for m in micro:
            n = int(m["tmask"].sum())
            offs.append((off, off + n))
            off += n

        # --- gradient ------------------------------------------------------
        t_fwd0 = time.time()
        opt.zero_grad(set_to_none=True)
        acc_stats: List[dict] = []
        logp_mismatch: List[float] = []

        for mi, m in enumerate(micro):
            row, c = m["row"], m["cand"]
            P = len(row["prompt_token_ids"])
            comp = c["token_ids"]
            ids = torch.tensor([row["prompt_token_ids"] + comp], device=dev)
            tmask = m["tmask"].to(dev)
            amask = m["amask"].to(dev)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp, ent = score_sequence(policy, ids, P, args.logit_chunk, True)
                with torch.no_grad():
                    logp_ref, _ = score_sequence(anchor, ids, P,
                                                 args.logit_chunk, False)
            n = m["n"]
            logp = logp[:n].unsqueeze(0)
            ent = ent[:n].unsqueeze(0)
            logp_ref = logp_ref[:n].unsqueeze(0).detach()
            lo = m["logp_old"].to(dev, dtype=logp.dtype)

            # Diagnostic: vLLM's logp_old vs the trainer's own forward. A large
            # gap means kernel divergence between the two engines and would
            # silently corrupt every importance ratio.
            with torch.no_grad():
                logp_mismatch.append(float((logp - lo).abs().mean()))

            s0, s1 = offs[mi]
            parts = stagec_loss(
                logp=logp, logp_old=lo, logp_ref=logp_ref, entropy=ent,
                token_advantages=m["tok_adv"].to(dev),
                think_mask=tmask, answer_mask=amask,
                lambda_tea=args.lambda_tea, lambda_align=args.lambda_align,
                tau_c=args.tau_c, cap_c=args.tea_cap_c,
                tea_scale=args.tea_scale,
                tea_weights=tea_w[s0:s1].to(dev),
                tea_weight_sum=tea_stats["weight_sum"],
                n_think_total=tea_stats["n_think"],
                policy_denom=policy_denom, kl_denom=kl_denom)
            # Already normalized by batch-level denominators, so each
            # micro-batch contributes its exact share — no /n_micro.
            parts.total.backward()
            acc_stats.append(parts.stats)
            del logp, ent, logp_ref, parts

        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        opt.step()
        torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd0

        # --- sync + logging -------------------------------------------------
        t_sync0 = time.time()
        if step % args.sync_every == 0:
            export_weights(weights_version + 1)
        t_sync = time.time() - t_sync0

        with torch.no_grad():
            theta = torch.cat([p.detach().flatten() for p in policy.parameters()])
            drift = float((theta - theta0).norm() / (theta0_norm + 1e-12))
            del theta

        def _m(key: str) -> float:
            vals = [s[key] for s in acc_stats if key in s]
            return float(statistics.mean(vals)) if vals else 0.0

        def _sum(key: str) -> float:
            """Loss terms are batch-normalized *contributions* — sum, never mean."""
            return float(sum(s[key] for s in acc_stats if key in s))

        all_bd = [b for _, sc, _ in kept for b in sc["breakdowns"]]
        rec = {
            "step": step, "t": time.time(),
            "wall": {"total": time.time() - step_t0, "rollout": t_rollout,
                     "scoring": t_score, "trainer": t_fwd, "sync": t_sync},
            "worker": bus.worker_status(),
            "n_groups": len(rows), "n_kept": len(kept),
            "drop_rate": n_drop / max(1, len(rows)),
            "drop_reasons": {r: sum(1 for g in group_stats if g.drop_reason == r)
                             for r in ("all_correct", "all_wrong")},
            "reward": {
                "mean": float(statistics.mean([b.total for b in all_bd])),
                "acc_rate": float(statistics.mean([b.r_acc > 0 for b in all_bd])),
                "lenient_only_rate": float(statistics.mean(
                    [b.lenient_only for b in all_bd])),
                "empty_think_rate": float(statistics.mean(
                    [b.empty_think for b in all_bd])),
                "g_rate": float(statistics.mean([b.g for b in all_bd])),
                "think_median": float(statistics.median(
                    [b.think_len for b in all_bd])),
                "answer_median": float(statistics.median(
                    [b.answer_len for b in all_bd])),
                "budget_B": budget.B, "budget_frozen_steps": budget.frozen_steps,
                "pen_contradiction": float(statistics.mean(
                    [b.penalties["contradiction"] for b in all_bd])),
                "pen_register_leak": float(statistics.mean(
                    [b.penalties["register_leak"] for b in all_bd])),
                "pen_ngram_loop": float(statistics.mean(
                    [b.penalties["ngram_loop"] for b in all_bd])),
                "word_stutter_rate": float(statistics.mean(
                    [b.flags["word_stutter"]["rate"] for b in all_bd])),
            },
            "loss": {"total": _sum("loss/total"), "policy": _sum("loss/policy"),
                     "tea_term": _sum("loss/tea_term"),
                     "kl_term": _sum("loss/kl_term")},
            "tea": {
                # Batch-level selection stats (computed before the forward pass).
                "l_tea": _sum("tea/l_tea_contrib"),
                "think_entropy_mean": _m("tea/think_entropy_mean"),
                "n_think": tea_stats["n_think"],
                "cap_hit_frac": tea_stats["cap_hit_frac"],
                "weight_sum": tea_stats["weight_sum"],
                "weight_max": tea_stats["weight_max"],
                # 1.0 = perfectly uniform = TEA selecting nothing (010 finding 8).
                "uniformity": tea_stats["uniformity"],
                "cov_max": tea_stats["cov_max"], "cov_min": tea_stats["cov_min"],
            },
            "kl": {"mean": _sum("kl/kl_mean"), "n_answer": _m("kl/n_answer")},
            "clip": {"low": _m("policy/clip_frac_low"),
                     "high": _m("policy/clip_frac_high"),
                     "ratio_mean": _m("policy/ratio_mean")},
            "opt": {"grad_norm": float(gnorm), "theta_drift_rel": drift,
                    "lr": args.lr, "weights_version": weights_version},
            "logp_old_mismatch": float(statistics.mean(logp_mismatch))
            if logp_mismatch else 0.0,
            "curriculum": curric.stats(),
            "groups": [g.__dict__ for g in group_stats],
        }
        log(rec)
        r = rec["reward"]
        print(f"[train] step {step:4d} | keep {len(kept)}/{len(rows)} | "
              f"acc {r['acc_rate']:.2f} | think {r['think_median']:.0f} | "
              f"ans {r['answer_median']:.0f} | H {rec['tea']['think_entropy_mean']:.3f} | "
              f"teaU {rec['tea']['uniformity']:.3f} | "
              f"drift {drift:.2e} | "
              f"wall {rec['wall']['total']:.0f}s "
              f"(gen {t_rollout:.0f} / train {t_fwd:.0f})", flush=True)

        if step % 25 == 0:
            curric.retilt()
            print(f"[train] retilt -> {curric.tilt}", flush=True)
        if step % args.ckpt_every == 0:
            d = os.path.join(args.run_dir, f"ckpt/step{step:04d}")
            os.makedirs(d, exist_ok=True)
            before = _theta_norm()
            policy.save_pretrained(d, state_dict=_bf16_state_dict())
            if _theta_norm() != before:
                raise AssertionError("checkpointing mutated the live policy")
            from transformers import AutoTokenizer as _AT
            _AT.from_pretrained(args.init_model).save_pretrained(d)
            print(f"[train] checkpoint -> {d}", flush=True)

    logf.close()
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
