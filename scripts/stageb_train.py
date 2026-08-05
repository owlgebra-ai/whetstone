"""Stage B — assimilation SFT: ZPD band-pass CE + SED (design §4, packet P6 Part 4).

The student is a fresh copy of the **original** checkpoint (never `scorer_v1`,
never a Round-0 artifact) trained on the certified teacher corpus with a plain
prompt. The register enters the weights here and only here.

Loss per sequence:

    L = (1 / Z) * sum_{t in completion} w_t * CE_t   +   alpha_sed * K2_think

* ``w_t`` is the ZPD band-pass of :mod:`whetstone.zpd`, precomputed offline per
  round under pi-S-at-round-start and read from the gate npz. Applied over
  **all** completion tokens: the student has to learn to write the answer
  segment too, not only the scratchpad.
* ``K2`` is the SED term (:mod:`whetstone.sed`), on **think tokens only** — its
  job is reasoning-entropy restoration, and pushing entropy into the answer
  channel is not what design §4.2 asks for. Recorded as a pinned decision.
* ``Z = max(sum w, 0.25 * n_completion)`` (design §4.1 + the degenerate-case
  floor). Normalizing by ``sum(w)`` rather than token count keeps a
  heavily-gated sequence from being quietly down-weighted; the floor keeps an
  almost-entirely-gated one from shouting over the batch.

Numerics are activity 007's, and they are not optional: **fp32 master weights**
with a bf16 autocast forward. At LR 2e-5 a bf16 weight update is an order of
magnitude below the format's quantum and rounds to zero *silently* — the run
completes, the loss curve looks plausible, and nothing has trained. Full fp32
AdamW then OOMs, so moments are 8-bit. ``theta_drift_rel`` is logged at every
eval and a zero after the first steps means stop.

Rounds. Round 1 trains 2 epochs from the original checkpoint. The gates are then
recomputed under the round-1 student (``stageb_zpd_gates.py``) and round 2 runs
1 further epoch from round 1's weights with a **fresh EMA copy** — phi never
carries over (design §12.4 rule 4). Training round 2 on round-1 gates is a named
drift failure, so this script refuses to start when the gate sidecar's pi_S
fingerprint disagrees with ``--pi-s-expected``. The check lives here rather than
in a checklist because checklists are how it happened before.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import marker_class_ids
from whetstone.sed import SEDRegularizer, row_logits, topk_entropy
from whetstone.zpd import (
    ALPHA_NOV_DEFAULT, GAMMA_INIT, KAPPA_DEFAULT, S_CAP_DEFAULT,
    band_pass, sequence_normalizer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stageb_zpd_gates import pi_s_fingerprint


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Record:
    """One training sequence with its precomputed band-pass weights."""

    __slots__ = ("uid", "level", "weight", "ids", "prompt_len", "think_start",
                 "think_end", "answer_start", "answer_end", "w", "n_completion",
                 "z", "z_floored", "masked_frac")

    def __init__(self, r: dict, s: np.ndarray, gamma: float, kappa: float,
                 alpha_nov: float, s_cap: float, floor_frac: float):
        self.uid = r["_uid"]
        self.level = r["level"]
        self.weight = r["weight"]
        self.ids = r["ids"]
        self.prompt_len = r["prompt_len"]
        self.think_start = r["think_start"]
        self.think_end = r["think_end"]
        self.answer_start = r["answer_start"]
        self.answer_end = r["answer_end"]
        gate, _, w = band_pass(s, gamma, kappa, alpha_nov, s_cap)
        self.w = w.astype(np.float32)           # indexed by completion offset
        self.n_completion = len(w)
        z_raw = float(w.sum())
        self.z = sequence_normalizer(w, self.n_completion, floor_frac)
        self.z_floored = self.z > z_raw + 1e-9
        self.masked_frac = float((gate < 0.1).mean())


def load_records(train_path: str, gates_path: str, args) -> list:
    rows = [json.loads(l) for l in open(train_path)]
    npz = np.load(gates_path)
    out, missing = [], 0
    for r in rows:
        key = f"{r['_uid']}#{r['trace_idx']}"
        if key not in npz:
            missing += 1
            continue
        s = npz[key]
        n_comp = len(r["ids"]) - r["prompt_len"]
        if len(s) != n_comp:
            raise SystemExit(
                f"{key}: gate array has {len(s)} entries for {n_comp} completion "
                "tokens — the gate file was built from a different train.jsonl")
        out.append(Record(r, s, args.gamma, args.kappa, args.alpha_nov,
                          args.s_cap, args.z_floor_frac))
    if missing:
        print(f"[data] WARNING {missing} records had no gate array and were dropped",
              flush=True)
    return out


def assert_gates_fresh(gates_path: str, pi_s_expected: str, round_idx: int) -> dict:
    """Refuse to train on gates produced under a different pi_S (design §4.3)."""
    meta_path = f"{os.path.splitext(gates_path)[0]}.meta.json"
    meta = json.load(open(meta_path))
    want = pi_s_fingerprint(pi_s_expected)
    if meta.get("pi_s_sha") != want["pi_s_sha"]:
        raise SystemExit(
            "STALE GATES — refusing to train.\n"
            f"  gate file : {gates_path}\n"
            f"  built under pi_S = {meta.get('pi_s')} (sha {meta.get('pi_s_sha')}, "
            f"round {meta.get('round')})\n"
            f"  this round expects pi_S = {want['pi_s']} (sha {want['pi_s_sha']})\n"
            "Recompute with scripts/stageb_zpd_gates.py under this round's student. "
            "Stale gates are a named drift failure (CLAUDE.md invariant, design §4.3)."
        )
    if meta.get("round") != round_idx:
        print(f"[gates] NOTE sidecar says round {meta.get('round')}, training round "
              f"{round_idx} — fingerprints match, continuing", flush=True)
    return meta


# --------------------------------------------------------------------------
# eval panels
# --------------------------------------------------------------------------

@torch.no_grad()
def control_entropy(model, control: list, topk: int = 512) -> dict:
    """Teacher-forced think-token entropy on a fixed slice — the F3c trajectory.

    Report mean AND median AND p80. Activity 007 finding 8: the median has no
    resolving power at an audit baseline of 0.0278 nats with 56.8% collapse
    mass; the mean is the sensitive one and is what to trip on.
    """
    vals = []
    model.eval()
    for rec in control:
        ids = torch.tensor([rec.ids], device="cuda")
        pos = torch.arange(rec.think_start, rec.think_end, device="cuda")
        if pos.numel() == 0:
            continue
        # bf16 autocast: fp32 SDPA falls back to the math backend and
        # materializes a full (T,T) attention matrix (activity 007 deviation 5).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows = row_logits(model, ids, pos - 1)
        vals.append(topk_entropy(rows.float(), k=topk).cpu().numpy())
    model.train()
    if not vals:
        return {}
    v = np.concatenate(vals)
    return {
        "control_entropy_mean": float(v.mean()),
        "control_entropy_median": float(np.median(v)),
        "control_entropy_p80": float(np.percentile(v, 80)),
        "control_entropy_n": int(v.size),
    }


#: Register-specific leakage markers. Deliberately NOT the full marker set: the
#: English word "case" appears in 10.2% of the corpus's own answers (activity 009
#: finding 1), so a bare substring count would fail F3d on ordinary prose.
LEAK_MARKERS = ("goal:", "chk:", "⇒", "✗")
DENSITY_MARKERS = ("goal:", "chk:", "⇒", "✓", "case", "let ", "→")


@torch.no_grad()
def spot_check(model, tok, probs: list, max_new: int = 2048,
               batch: int = 10) -> dict:
    """Fixed val problems, greedy — the earliest visible assimilation signal.

    Think length should start collapsing from the native ~6k toward corpus scale
    within the first epoch. Also watches answer-segment register leakage, which
    must stay ~0: the register belongs in think only (F3d).

    Generation is **batched with left padding**. One-at-a-time on a 1.7B under
    fp32 weights + autocast runs ~20 tok/s, so 20 problems x 2,048 new tokens is
    half an hour — per eval. Batched it is a couple of minutes. Left padding is
    required for decoder-only generation; right padding silently generates from
    pad tokens.

    ``use_cache`` and gradient checkpointing are toggled around the call and
    restored: the training config has cache off (checkpointing needs it off),
    and generating without a KV cache is quadratic.
    """
    from whetstone.segments import parse_segments

    if not probs:
        return {}
    was_cache = model.config.use_cache
    was_ckpt = getattr(model, "is_gradient_checkpointing", False)
    old_side = tok.padding_side
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.eval()
    if was_ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    tok.padding_side = "left"

    think_lens, answer_lens, g_ok, leak, mark = [], [], 0, 0, []
    try:
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True,
                                    enable_thinking=True, tokenize=False)
            for p in probs
        ]
        for s in range(0, len(texts), batch):
            enc = tok(texts[s: s + batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=pad_id)
            plen = enc["input_ids"].shape[1]
            for i in range(out.shape[0]):
                gen = [t for t in out[i, plen:].tolist() if t != pad_id]
                m = parse_segments(gen, prompt_len=0)
                g_ok += int(m.g == 1)
                think_lens.append(m.think_len)
                answer_lens.append(m.answer_len)
                if m.g == 1:
                    atxt = tok.decode(gen[m.answer_start:m.answer_end])
                    leak += int(any(k in atxt for k in LEAK_MARKERS))
                    ttxt = tok.decode(gen[m.think_start:m.think_end])
                    mark.append(100.0 * sum(ttxt.count(k) for k in DENSITY_MARKERS)
                                / max(len(ttxt), 1))
    finally:
        tok.padding_side = old_side
        model.config.use_cache = was_cache
        if was_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

    return {
        "spot_think_median": float(np.median(think_lens)) if think_lens else None,
        "spot_answer_median": float(np.median(answer_lens)) if answer_lens else None,
        "spot_g_rate": g_ok / max(len(probs), 1),
        "spot_answer_leak_rate": leak / max(g_ok, 1) if g_ok else None,
        "spot_think_marker_density": float(np.mean(mark)) if mark else None,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--init", required=True,
                    help="round 1: Qwen/Qwen3-1.7B (the ORIGINAL checkpoint). "
                         "round 2: the round-1 output dir.")
    ap.add_argument("--pi-s-expected", required=True,
                    help="checkpoint the gates MUST have been scored under; "
                         "for round 1 this equals --init")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-pool", default="/data/whetstone/data/pool/val.jsonl")

    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=GAMMA_INIT)
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT)
    ap.add_argument("--alpha-nov", type=float, default=ALPHA_NOV_DEFAULT)
    ap.add_argument("--s-cap", type=float, default=S_CAP_DEFAULT)
    ap.add_argument("--z-floor-frac", type=float, default=0.25)

    ap.add_argument("--alpha-sed", type=float, default=1.0)
    ap.add_argument("--h-pivot", type=float, default=0.6707)
    ap.add_argument("--delta-max", type=float, default=0.7)
    ap.add_argument("--gamma-e", type=float, default=1.0)
    ap.add_argument("--sed-max-think", type=int, default=1024)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--sync-every", type=int, default=5)

    ap.add_argument("--ce-chunk", type=int, default=512,
                    help="rows per fp32 log-softmax block; bounds the transient")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--n-control", type=int, default=32)
    ap.add_argument("--n-spot", type=int, default=20)
    ap.add_argument("--spot-max-new", type=int, default=2048)
    ap.add_argument("--spot-batch", type=int, default=10)
    ap.add_argument("--spot-every", type=int, default=100,
                    help="optimizer steps between generative spot-checks. Coarser "
                         "than --eval-every because generation is the expensive "
                         "panel: before assimilation every rollout runs to the "
                         "2,048 cap, and that is minutes even batched.")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"; ckpt_dir.mkdir(exist_ok=True)

    gates_meta = assert_gates_fresh(args.gates, args.pi_s_expected, args.round)
    print(f"[gates] fresh: pi_S = {gates_meta['pi_s']} (sha {gates_meta['pi_s_sha']})",
          flush=True)

    recs = load_records(args.train, args.gates, args)
    if args.limit:
        recs = recs[: args.limit]
    n_floored = sum(r.z_floored for r in recs)
    print(f"[data] {len(recs)} sequences | mean masked {100*np.mean([r.masked_frac for r in recs]):.2f}% "
          f"| Z-floor binds on {n_floored} ({100*n_floored/max(len(recs),1):.2f}%)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.init, trust_remote_code=True)
    mclass = marker_class_ids(tok)
    marker_ids = torch.tensor(sorted(frozenset().union(*mclass.values())), device="cuda")

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e9:.3f}B params, "
          "fp32 master weights + bf16 autocast", flush=True)

    # Rule 4: a NEW EMA copy per stage and per round, initialised from THIS
    # round's starting weights. Round 0's phi is dead; round 1's is not reused.
    sed = SEDRegularizer(model, ema_decay=args.ema_decay, sync_every=args.sync_every,
                         tau_range=(1.1, 1.5), topk=512, H_pivot=args.h_pivot,
                         delta_max=args.delta_max, gamma_e=args.gamma_e,
                         shadow_dtype=torch.bfloat16)

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(len(recs) // args.accum, 1)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)
    print(f"[train] {total_steps} optimizer steps ({args.epochs} epochs x "
          f"{steps_per_epoch}/epoch, accum {args.accum}); step 1 runs at LR 0 "
          "(cosine-with-warmup, harmless over this many steps)", flush=True)

    rng = np.random.default_rng(args.seed)
    control = [recs[i] for i in rng.permutation(len(recs))[: args.n_control]]
    val = [json.loads(l) for l in open(args.val_pool)][: args.n_spot] \
        if os.path.exists(args.val_pool) else []
    spot_probs = [v["prompt"] for v in val]
    print(f"[eval] control slice {len(control)} seqs | spot-check {len(spot_probs)} problems",
          flush=True)

    theta0 = torch.cat([p.detach().float().reshape(-1)[::997]
                        for p in model.parameters()]).clone()
    metrics_path = out_dir / "stageb_metrics.jsonl"
    metrics_path.write_text("")
    rows_log: list = []
    t_start = time.time()

    def do_eval(step: int, run: dict, force_spot: bool = False) -> dict:
        t0 = time.time()
        drift = torch.cat([p.detach().float().reshape(-1)[::997]
                           for p in model.parameters()])
        m = {
            "step": step,
            "round": args.round,
            "lr": sched.get_last_lr()[0],
            "ce_weighted": run["ce"] / max(run["n"], 1),
            "sed_k2": run["sed"] / max(run["n"], 1),
            "w_masked_frac": run["masked"] / max(run["n_tok"], 1),
            "w_marker_mean": run["marker_w"] / max(run["marker_n"], 1) if run["marker_n"] else None,
            "w_mean": run["w_sum"] / max(run["n_tok"], 1),
            "z_floor_hits": run["floored"],
            "theta_drift_l2": float((drift - theta0).norm()),
            "theta_drift_rel": float((drift - theta0).norm() / theta0.norm()),
            "n_ema_syncs": sed.n_syncs,
            "ema_syncs_expected": step // args.sync_every,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "elapsed_seconds": time.time() - t_start,
        }
        m.update(control_entropy(model, control))
        if spot_probs and (force_spot or step % args.spot_every == 0):
            m.update(spot_check(model, tok, spot_probs, args.spot_max_new,
                                args.spot_batch))
        m["eval_seconds"] = time.time() - t0
        rows_log.append(m)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")

        if m["n_ema_syncs"] != m["ema_syncs_expected"]:
            print(f"  [!!] EMA cadence wrong: {m['n_ema_syncs']} syncs at step "
                  f"{step}, expected {m['ema_syncs_expected']} — micro-batches "
                  "are being counted as optimizer steps", flush=True)
        if step >= 2 and m["theta_drift_rel"] == 0.0:
            raise SystemExit(
                f"theta_drift_rel = 0 at step {step}: the weights are not moving. "
                "At this LR a bf16 update rounds to zero silently — check that "
                "master weights are fp32 (activity 007 numerics).")

        print(f"[eval {step:>4}] ce={m['ce_weighted']:.4f} sed={m['sed_k2']:.5f} "
              f"| H mean={m.get('control_entropy_mean', float('nan')):.4f} "
              f"med={m.get('control_entropy_median', float('nan')):.4f} "
              f"p80={m.get('control_entropy_p80', float('nan')):.4f} "
              f"| think={m.get('spot_think_median')} ans={m.get('spot_answer_median')} "
              f"g={m.get('spot_g_rate')} leak={m.get('spot_answer_leak_rate')} "
              f"mark={m.get('spot_think_marker_density')} "
              f"| w̄={m['w_mean']:.3f} wR={m['w_marker_mean']} "
              f"drift={m['theta_drift_rel']:.2e} mem={m['peak_mem_gb']:.1f}GB "
              f"({m['eval_seconds']:.0f}s)", flush=True)
        return m

    def save(step: int, tag: str = "") -> None:
        ck = ckpt_dir / (tag or f"step{step:04d}")
        model.save_pretrained(ck, safe_serialization=True, state_dict={
            k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
        tok.save_pretrained(ck)
        print(f"  [ckpt] {ck}", flush=True)

    # --- train ------------------------------------------------------------
    model.train()
    step = micro = 0
    run = collections.defaultdict(float)
    opt.zero_grad(set_to_none=True)
    do_eval(0, run)
    stop = False

    for epoch in range(int(np.ceil(args.epochs))):
        if stop:
            break
        order = rng.permutation(len(recs))       # over problems, seeded
        for idx in order:
            rec = recs[idx]
            ids = torch.tensor([rec.ids], device="cuda")
            p0 = rec.prompt_len
            n = len(rec.ids)

            # CE over ALL completion tokens; SED over think tokens only.
            ce_pos = torch.arange(p0, n, device="cuda")
            w = torch.tensor(rec.w, device="cuda", dtype=torch.float32)
            labels = ids[0, ce_pos]

            # CE stays INSIDE the autocast block and goes through
            # F.cross_entropy rather than log_softmax + gather. cross_entropy is
            # on autocast's fp32 list, so the arithmetic is fp32 either way, but
            # its fused kernel recomputes in backward instead of saving an
            # (N, 151936) fp32 log-softmax -- which on this corpus's longest
            # record is 1.7 GB of saved activation for one sequence.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rlogits = row_logits(model, ids, ce_pos - 1)
                parts = [
                    F.cross_entropy(rlogits[s: s + args.ce_chunk],
                                    labels[s: s + args.ce_chunk], reduction="none")
                    for s in range(0, rlogits.shape[0], args.ce_chunk)
                ]
            ce_tok = torch.cat(parts).float()
            ce = (w * ce_tok).sum() / rec.z

            # SED rows: think positions, capped so one outlier trace cannot
            # spike the fp32 intermediates past the memory budget.
            t_lo, t_hi = rec.think_start - p0, rec.think_end - p0
            sed_off = torch.arange(t_lo, t_hi, device="cuda")
            if sed_off.numel() > args.sed_max_think:
                sel = torch.randperm(sed_off.numel(), device="cuda")[: args.sed_max_think]
                sed_off = sed_off[sel].sort().values
            if sed_off.numel():
                sed_loss = sed.loss_rows(rlogits[sed_off], ids, ce_pos[sed_off])
            else:
                sed_loss = rlogits.sum() * 0.0

            loss = rec.weight * (ce + args.alpha_sed * sed_loss)
            (loss / args.accum).backward()

            run["ce"] += float(ce.detach()); run["sed"] += float(sed_loss.detach())
            run["n"] += 1
            run["n_tok"] += rec.n_completion
            run["w_sum"] += float(rec.w.sum())
            run["masked"] += float((rec.w < 0.1).sum())
            run["floored"] += float(rec.z_floored)
            in_r = torch.isin(labels, marker_ids)
            if bool(in_r.any()):
                run["marker_w"] += float(w[in_r].sum())
                run["marker_n"] += float(in_r.sum())
            micro += 1

            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                sed.maybe_sync(step)          # optimizer steps, never micro-batches

                if step % args.eval_every == 0 or step == total_steps:
                    do_eval(step, run)
                    run = collections.defaultdict(float)
                if step % args.ckpt_every == 0:
                    save(step)
                if step >= total_steps:
                    stop = True
                    break
        if not stop:
            save(step, f"epoch{epoch+1}")

    if step % args.eval_every != 0:
        do_eval(step, run)
    save(step, "final")

    summary = {
        "round": args.round, "steps_run": step, "epochs": args.epochs,
        "lr": args.lr, "accum": args.accum, "warmup": args.warmup,
        "gamma": args.gamma, "kappa": args.kappa, "alpha_nov": args.alpha_nov,
        "s_cap": args.s_cap, "z_floor_frac": args.z_floor_frac,
        "alpha_sed": args.alpha_sed, "h_pivot": args.h_pivot,
        "delta_max": args.delta_max, "gamma_e": args.gamma_e,
        "init": args.init, "train": args.train, "gates": args.gates,
        "gates_pi_s": gates_meta.get("pi_s"), "gates_pi_s_sha": gates_meta.get("pi_s_sha"),
        "n_sequences": len(recs),
        "final": rows_log[-1] if rows_log else None,
        "step0": rows_log[0] if rows_log else None,
    }
    with open(out_dir / "stageb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\n[done] {step} steps -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
