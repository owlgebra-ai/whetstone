"""Round-0 scorer inoculation (packet P4 Part 3; design §2, §7).

Trains the **scorer** — not the student — so compact-register tokens read as a
low hum instead of spikes, while genuine reasoning leaps still spike. The
product is a trustworthy measuring instrument, not a capable model.

    L = sum_{t in R and think} CE_t  +  alpha_sed * L_SED(think)

Answer tokens get no loss. An optional KL-to-pi_0 on non-R tokens exists behind
``--kl-non-r`` and is **off by default** (design §2 marks it optional; reach for
it only if S2 keeps tripping).

**Stopping is a threshold, not a minimum** (design §2). The run stops at the
first of S1 / S2 / S3, and the winning checkpoint is chosen *retroactively* —
every eval checkpoint is kept until the F1 verdict, because the overshoot
signature (verbose-control likelihood falling while register p95 keeps dropping
past tau_spike) is only visible after the crossing. Rollback is expected.

Numerics note (activity 007). Parameters are held in **fp32** with a bf16
autocast forward, not in bf16. At LR 1e-5 an Adam update is ~1e-5 while bf16's
quantum at a typical weight magnitude of 0.02 is ~1.2e-4 — every update would
round to zero and the run would no-op while reporting a falling loss. ``theta_
drift`` is logged at every eval so that failure can never be silent again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0 import build_sequence, load_jsonl, marker_class_ids  # noqa: E402
from whetstone.round0_eval import Pi0Cache, build_eval_sets, evaluate  # noqa: E402
from whetstone.sed import SEDRegularizer, row_logits  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"


def load_r_ids(path: str) -> frozenset:
    d = json.loads(Path(path).read_text())
    return frozenset(int(k) for k in d if not k.startswith("_"))


def plot_curves(rows: List[dict], out_dir: Path, tau_spike: float) -> None:
    """The four §7 monitoring curves. Dashboards are deliverables, not afterthoughts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0][0]
    a.plot(steps, [r["s1_p95_gap"] for r in rows], "o-", label="overall", lw=2)
    for name, c in (("structural", "tab:green"), ("branch", "tab:red")):
        key = f"s1_p95_gap_{name}"
        if any(not math.isnan(r.get(key, math.nan)) for r in rows):
            a.plot(steps, [r.get(key) for r in rows], "s--", color=c, label=name)
    a.axhline(tau_spike, color="k", ls=":", label=f"τ_spike = {tau_spike}")
    a.set_title("S1 — held-out register p95 gap (stop criterion)")
    a.set_xlabel("optimizer step"); a.set_ylabel("p95 d_t (nats)"); a.legend(); a.grid(alpha=.3)

    a = ax[0][1]
    a.plot(steps, [r.get("s2_kl_mean") for r in rows], "o-", color="tab:orange", lw=2)
    a.set_title("S2 — verbose-control drift, mean top-512 KL(π_θ‖π_0)")
    a.set_xlabel("optimizer step"); a.set_ylabel("nats/token"); a.grid(alpha=.3)

    a = ax[1][0]
    a.plot(steps, [r.get("control_entropy_mean") for r in rows], "o-", label="π_θ mean")
    a.plot(steps, [r.get("control_entropy_median") for r in rows], "s-", label="π_θ median")
    if rows and rows[0].get("pi0_entropy_mean") is not None:
        a.axhline(rows[0]["pi0_entropy_mean"], color="tab:blue", ls=":", label="π_0 mean")
        a.axhline(rows[0]["pi0_entropy_median"], color="tab:orange", ls=":", label="π_0 median")
    a.set_title("S3 — control think entropy (SED health)")
    a.set_xlabel("optimizer step"); a.set_ylabel("nats"); a.legend(); a.grid(alpha=.3)

    a = ax[1][1]
    a.plot(steps, [r["hum_R_mean_surprisal"] for r in rows], "o-", color="tab:purple", lw=2)
    a.set_title("Hum — R-token mean surprisal on held-out\n(should fall to a plateau, not to zero)")
    a.set_xlabel("optimizer step"); a.set_ylabel("nats"); a.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_dir / "round0_curves.png", dpi=130)
    plt.close(fig)

    # The overshoot signature gets its own panel: it is a *joint* condition and
    # is invisible in either curve alone (design §7).
    fig, a = plt.subplots(figsize=(7.5, 5))
    a.plot(steps, [r["s1_p95_gap"] for r in rows], "o-", color="tab:blue", label="register p95 gap")
    a.axhline(tau_spike, color="k", ls=":", label=f"τ_spike = {tau_spike}")
    a.set_xlabel("optimizer step"); a.set_ylabel("p95 d_t (nats)", color="tab:blue")
    a2 = a.twinx()
    a2.plot(steps, [r.get("b_logprob_delta_mean") for r in rows], "s-",
            color="tab:red", label="verbose Δlogp vs π_0")
    a2.axhline(0, color="tab:red", ls="--", lw=.8)
    a2.set_ylabel("verbose-control Δ logprob (nats/token)", color="tab:red")
    a.set_title("Overshoot watch: register p95 ↓ past τ_spike\nwhile verbose likelihood ↓ ⇒ roll back")
    a.grid(alpha=.3); a.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "round0_overshoot.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/data/whetstone/corpora/seed_register_qwen")
    ap.add_argument("--r-tokenset", default="/data/whetstone/runs/round0/R_tokenset.json")
    ap.add_argument("--pi0-cache", default="/data/whetstone/runs/round0/pi0_cache.npz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-dir", default="/data/whetstone/runs/round0")
    ap.add_argument("--ckpt-dir", default="/data/whetstone/ckpt/round0")

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--alpha-sed", type=float, default=1.0)
    ap.add_argument("--gamma-e", type=float, default=1.0)
    ap.add_argument("--h-pivot", type=float, default=0.6707)
    ap.add_argument("--delta-max", type=float, default=0.7)
    ap.add_argument("--kl-non-r", type=float, default=0.0,
                    help="optional KL-to-π_0 on ¬R tokens; design §2 marks it optional, default off")

    ap.add_argument("--eval-every", type=int, default=10, help="optimizer steps")
    ap.add_argument("--tau-spike", type=float, default=1.2)
    ap.add_argument("--kappa-max", type=float, default=0.0, help="0 = set after 3 evals")
    ap.add_argument("--entropy-drop-x", type=float, default=0.10)
    ap.add_argument("--n-control", type=int, default=0)
    ap.add_argument("--n-control-positions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sed-max-think", type=int, default=1024,
                    help="cap on SED think tokens per record (memory bound; p99 is 1039)")
    ap.add_argument("--max-steps", type=int, default=0, help="debug cap on optimizer steps")
    ap.add_argument("--no-stop", action="store_true", help="run the full epoch regardless of S1/S2/S3")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(args.model)
    r_ids = load_r_ids(args.r_tokenset)
    class_ids = marker_class_ids(tok)
    print(f"[R] |R| = {len(r_ids)}; marker classes: "
          + ", ".join(f"{k}={len(v)}" for k, v in class_ids.items()))

    # --- data ------------------------------------------------------------
    train = load_jsonl(Path(args.corpus) / "train.jsonl")
    seqs = [
        build_sequence(
            tok, uid=r["_uid"], problem=r["prompt"], think_body=r["compact_think"],
            answer=r["answer"], level=r.get("level", 0), require_gate=True,
        )
        for r in train
    ]
    r_arr = np.array(sorted(r_ids))
    r_counts = []
    for s in seqs:
        ids = np.asarray(s.ids)
        tm = np.asarray(s.masks.think_mask, dtype=bool)
        r_counts.append(int((np.isin(ids, r_arr) & tm).sum()))
    norm_r = float(np.mean(r_counts))
    print(f"[data] {len(seqs)} train sequences, all g=1; "
          f"R∩think tokens per record: mean {norm_r:.2f}, median {np.median(r_counts):.0f}, "
          f"zero-R records {sum(1 for c in r_counts if c == 0)}")

    sets = build_eval_sets(
        tok, args.corpus, n_control=args.n_control,
        n_control_positions=args.n_control_positions, seed=args.seed,
    )
    cache = Pi0Cache(args.pi0_cache) if Path(args.pi0_cache).exists() else None
    if cache is None:
        print(f"[warn] no π_0 cache at {args.pi0_cache} — S2/S3/(b) will be skipped")
    else:
        assert cache.uids == [s.uid for s in sets.control], (
            "π_0 cache uid order does not match the eval sets — rebuild the cache "
            "with the same --seed and --n-control-positions"
        )
    print(f"[eval] heldout {len(sets.heldout)}, control {len(sets.control)} "
          f"({sum(len(p) for p in sets.control_positions):,} scored positions)")

    # --- model -----------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e9:.3f}B params, fp32 weights + bf16 autocast")

    sed = SEDRegularizer(
        model, ema_decay=0.99, sync_every=5, tau_range=(1.1, 1.5), topk=512,
        H_pivot=args.h_pivot, delta_max=args.delta_max, gamma_e=args.gamma_e,
        shadow_dtype=torch.bfloat16,
    )

    steps_per_epoch = len(seqs) // args.accum
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)
    print(f"[train] {total_steps} optimizer steps (accum {args.accum}, "
          f"{args.epochs} epoch cap), eval every {args.eval_every}")

    r_t = torch.tensor(r_arr, device="cuda")
    theta0 = torch.cat([p.detach().float().reshape(-1)[::997] for p in model.parameters()]).clone()

    rows: List[dict] = []
    metrics_path = out_dir / "round0_metrics.jsonl"
    metrics_path.write_text("")
    stop_reason = None
    kappa_max = args.kappa_max or None

    def do_eval(step: int, t_start: float) -> dict:
        t0 = time.time()
        m = evaluate(model, sets, r_ids=r_ids, class_ids=class_ids, cache=cache)
        gaps = m.pop("_gaps_heldout")
        drift = torch.cat([p.detach().float().reshape(-1)[::997] for p in model.parameters()])
        m.update({
            "step": step,
            "theta_drift_l2": float((drift - theta0).norm()),
            "theta_drift_rel": float((drift - theta0).norm() / theta0.norm()),
            "n_ema_syncs": sed.n_syncs,
            "eval_seconds": time.time() - t0,
            "elapsed_seconds": time.time() - t_start,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        })
        rows.append(m)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        np.save(out_dir / f"gaps_heldout_step{step}.npy", np.asarray(gaps, dtype=np.float32))

        ck = ckpt_dir / f"step{step:04d}"
        model.save_pretrained(ck, safe_serialization=True,
                              state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
        tok.save_pretrained(ck)

        print(
            f"[eval step {step:>3}] S1 p95={m['s1_p95_gap']:.3f} "
            f"(struct={m.get('s1_p95_gap_structural', float('nan')):.3f} "
            f"branch={m.get('s1_p95_gap_branch', float('nan')):.3f}) "
            f"hum_R={m['hum_R_mean_surprisal']:.3f} "
            f"S2 KL={m.get('s2_kl_mean', float('nan')):.5f} "
            f"S3 mean_drop={m.get('s3_drop_mean', float('nan')):+.3f} "
            f"Δlogp={m.get('b_logprob_delta_mean', float('nan')):+.4f} "
            f"drift={m['theta_drift_rel']:.2e} "
            f"| {m['eval_seconds']:.0f}s mem={m['peak_mem_gb']:.1f}GB",
            flush=True,
        )
        return m

    t_start = time.time()
    m0 = do_eval(0, t_start)
    print(f"[step 0] </think> sanity entropy: median={m0['close_token_entropy_median']:.2e} "
          f"mean={m0['close_token_entropy_mean']:.2e} nats "
          "(median must be ~1e-4..0.02 per activities 003/005; a large MEDIAN means "
          "the logits/token alignment is off by one)")

    order = np.random.default_rng(args.seed).permutation(len(seqs))
    model.train()
    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    ce_run, sed_run, n_run = 0.0, 0.0, 0

    for idx in order:
        seq = seqs[idx]
        ids = torch.tensor([seq.ids], device="cuda")
        think = torch.tensor(seq.masks.think_mask, device="cuda")

        # Only think tokens carry loss, so lm_head runs on those rows alone —
        # memory then scales with the think segment (median 150) rather than the
        # sequence (median 1003, max 4491). SED think positions are capped at
        # --sed-max-think per record so one outlier trace cannot spike the fp32
        # log-softmax intermediates past the budget; R positions are never
        # dropped (there are ~6 per record and they are the point of the loss).
        tm = think.clone(); tm[0] = 0
        pos = torch.nonzero(tm, as_tuple=False).squeeze(-1)
        labels_all = ids[0, pos]
        in_r = torch.isin(labels_all, r_t)
        keep = pos
        if pos.numel() > args.sed_max_think:
            sel = torch.randperm(pos.numel(), device=pos.device)[: args.sed_max_think]
            keep = torch.unique(torch.cat([pos[sel], pos[in_r]]))
        rows_pos = keep

        with torch.autocast("cuda", dtype=torch.bfloat16):
            rlogits = row_logits(model, ids, rows_pos - 1)

        # --- masked CE on R and think -----------------------------------
        row_labels = ids[0, rows_pos]
        row_in_r = torch.isin(row_labels, r_t)
        ce = rlogits.new_zeros(())
        if row_in_r.any():
            sub = rlogits[row_in_r].float()
            lp = F.log_softmax(sub, dim=-1)
            # Sum, normalized by the corpus mean R-tokens-per-record: every R
            # occurrence carries equal weight (a per-record mean would make one
            # marker in a short trace count as much as ten in a long one), while
            # the gradient scale stays ~O(1).
            ce = -lp.gather(-1, row_labels[row_in_r].unsqueeze(-1)).squeeze(-1).sum() / norm_r

        sed_loss = sed.loss_rows(rlogits, ids, rows_pos)
        loss = ce + args.alpha_sed * sed_loss

        if args.kl_non_r > 0 and cache is not None:
            raise NotImplementedError("--kl-non-r needs a π_0 forward; off by default (design §2)")

        (loss / args.accum).backward()
        ce_run += float(ce.detach()); sed_run += float(sed_loss.detach()); n_run += 1
        micro += 1

        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            sed.maybe_sync(step)          # optimizer steps, never micro-batches

            if step % args.eval_every == 0 or step == total_steps:
                print(f"  [train] step {step}/{total_steps} "
                      f"ce={ce_run/max(n_run,1):.4f} sed={sed_run/max(n_run,1):.5f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                ce_run = sed_run = 0.0; n_run = 0
                m = do_eval(step, t_start)

                if kappa_max is None and len(rows) >= 4:
                    # Packet §7: set kappa_max after the first 3 evals so it
                    # would not have fired in the first third of the run.
                    seen = [r["s2_kl_mean"] for r in rows[1:4]]
                    kappa_max = round(max(seen) * 3.0, 6)
                    print(f"  [S2] κ_max pinned at {kappa_max:.6f} "
                          f"(3× the max of the first three evals: {[round(s,6) for s in seen]})")

                if not args.no_stop:
                    if m["s1_p95_gap"] < args.tau_spike:
                        stop_reason = f"S1 (p95 {m['s1_p95_gap']:.3f} < τ_spike {args.tau_spike})"
                    elif kappa_max and m.get("s2_kl_mean", 0) > kappa_max:
                        stop_reason = f"S2 (KL {m['s2_kl_mean']:.5f} > κ_max {kappa_max:.5f})"
                    elif m.get("s3_drop_mean", 0) > args.entropy_drop_x:
                        stop_reason = (f"S3 (entropy mean drop {m['s3_drop_mean']:.3f} "
                                       f"> x {args.entropy_drop_x})")
                    if stop_reason:
                        print(f"[stop] {stop_reason}")
                        break
            if step >= total_steps:
                break

    if step % args.eval_every != 0 and step > 0:
        do_eval(step, t_start)

    plot_curves(rows, out_dir, args.tau_spike)
    summary = {
        "stop_reason": stop_reason or f"epoch cap ({total_steps} steps)",
        "steps_run": step,
        "tau_spike": args.tau_spike,
        "kappa_max": kappa_max,
        "entropy_drop_x": args.entropy_drop_x,
        "gamma_e": args.gamma_e,
        "h_pivot": args.h_pivot,
        "delta_max": args.delta_max,
        "alpha_sed": args.alpha_sed,
        "lr": args.lr,
        "accum": args.accum,
        "norm_r": norm_r,
        "n_R": len(r_ids),
        "wall_seconds": time.time() - t_start,
        "checkpoints": sorted(p.name for p in ckpt_dir.glob("step*")),
    }
    (out_dir / "round0_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[done] {summary['stop_reason']}  |  {step} steps, "
          f"{summary['wall_seconds']/60:.1f} min  |  {len(rows)} evals")
    print(f"[done] curves → {out_dir}/round0_curves.png, {out_dir}/round0_overshoot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
