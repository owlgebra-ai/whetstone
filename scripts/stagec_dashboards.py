"""Stage-C dashboards from ``train_log.jsonl`` (design §7 playbook; CLAUDE.md).

"Dashboards are first-class deliverables of each stage — build them alongside
the training code, not after." Activity 009 finding 11 is the standing law
behind that: **loss and entropy cannot detect a dead run.** Two Stage-B arms,
one working and one producing nothing parseable, ended at CE 0.5399 vs 0.5524
and entropy 0.6456 vs 0.6381. The curves that *do* separate them are generative
and structural, so those are the ones plotted here.

Panels:
  1. entropy trajectory (TEA health) against the Part-0 entropy card baseline
  2. **think and answer length medians as separate curves** — one combined
     number is how segment drift hides (CLAUDE.md invariant)
  3. reward decomposition: accuracy rate, mean total, penalty magnitudes
  4. **degeneracy watch**: empty-think rate, lenient-only rate, loop penalty
     rate, word-stutter rate — the four ways this student is known to rot
  5. DAPO health: group drop rate (phase exhaustion), clip fractions, ratio
  6. TEA internals: L_TEA both scales, selected-vs-mean entropy, cap hits
  7. optimizer: theta_drift_rel (a no-op run has caught this project twice),
     grad norm, answer KL
  8. **pipeline balance** — the packet's topology verdict, wall-clock per step
     split into rollout / scoring / trainer / sync

matplotlib is turing-only (CLAUDE.md), so run this there.

Usage::

    python scripts/stagec_dashboards.py \\
        --run_dir /data/whetstone/runs/stagec/pilot \\
        --out_dir activity/assets/010
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Baselines from activity 010 run 2 (entropy card on the round-1 student) and
# the frozen baseline card. Plotted as reference lines, never as targets.
ENTROPY_CARD_MEAN = 0.61977
ENTROPY_AUDIT_BASELINE_MEAN = 0.31759
BASELINE_ANSWER_MEDIAN = 288
INIT_THINK_MEDIAN_GSM8K = 218


def _get(rec: dict, path: str, default=None):
    cur = rec
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load(run_dir: str) -> list:
    p = os.path.join(run_dir, "train_log.jsonl")
    rows = []
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "step" in r and "reward" in r:
                rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Stage C")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.run_dir)
    if not rows:
        print("[dash] no completed steps in train_log.jsonl")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    s = [r["step"] for r in rows]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f"{args.title} — {len(rows)} steps", fontsize=14)

    # 1. entropy trajectory
    ax = axes[0][0]
    ax.plot(s, [_get(r, "tea.think_entropy_mean", 0) for r in rows],
            label="think entropy (mean)")
    ax.plot(s, [_get(r, "tea.selected_entropy_mean", 0) for r in rows],
            label="TEA-selected tokens", alpha=0.7)
    ax.axhline(ENTROPY_CARD_MEAN, ls="--", c="g", label="pre-RL card 0.620")
    ax.axhline(ENTROPY_AUDIT_BASELINE_MEAN, ls=":", c="r", label="original ckpt 0.318")
    ax.set_title("1. Entropy trajectory (TEA health)")
    ax.set_xlabel("step"); ax.set_ylabel("nats"); ax.legend(fontsize=7)

    # 2. segment lengths — ALWAYS separate
    ax = axes[0][1]
    ax.plot(s, [_get(r, "reward.think_median", 0) for r in rows], label="think median")
    ax.plot(s, [_get(r, "reward.answer_median", 0) for r in rows], label="answer median")
    ax.plot(s, [_get(r, "reward.budget_B", 0) for r in rows], ls="--", alpha=0.6,
            label="budget B")
    ax.axhline(BASELINE_ANSWER_MEDIAN, ls=":", c="r", label="baseline answer 288")
    ax.set_title("2. Segment lengths (never combined)")
    ax.set_xlabel("step"); ax.set_ylabel("tokens"); ax.legend(fontsize=7)

    # 3. reward decomposition
    ax = axes[1][0]
    ax.plot(s, [_get(r, "reward.acc_rate", 0) for r in rows], label="strict acc rate")
    ax.plot(s, [_get(r, "reward.mean", 0) for r in rows], label="mean total reward",
            alpha=0.7)
    ax.plot(s, [_get(r, "reward.g_rate", 0) for r in rows], label="g rate", alpha=0.6)
    ax.set_title("3. Reward")
    ax.set_xlabel("step"); ax.legend(fontsize=7)

    # 4. degeneracy watch — the four known rot modes
    ax = axes[1][1]
    ax.plot(s, [_get(r, "reward.empty_think_rate", 0) for r in rows],
            label="empty think", lw=2)
    ax.plot(s, [_get(r, "reward.lenient_only_rate", 0) for r in rows],
            label="lenient-only (grading hole)")
    ax.plot(s, [_get(r, "reward.pen_ngram_loop", 0) for r in rows],
            label="loop penalty (mean)")
    ax.plot(s, [_get(r, "reward.word_stutter_rate", 0) for r in rows],
            label="word stutter")
    ax.plot(s, [_get(r, "reward.pen_register_leak", 0) for r in rows],
            label="register leak (mean)", alpha=0.6)
    ax.set_title("4. Degeneracy watch — read rollouts if any of these climb")
    ax.set_xlabel("step"); ax.legend(fontsize=7)

    # 5. DAPO health
    ax = axes[2][0]
    ax.plot(s, [r.get("drop_rate", 0) for r in rows], label="group drop rate")
    ax.plot(s, [_get(r, "clip.low", 0) for r in rows], label="clip frac low")
    ax.plot(s, [_get(r, "clip.high", 0) for r in rows], label="clip frac high")
    ax.set_title("5. DAPO health (drop rate = phase exhaustion)")
    ax.set_xlabel("step"); ax.legend(fontsize=7)

    # 6. TEA internals
    ax = axes[2][1]
    ax.plot(s, [_get(r, "tea.l_tea_mean", 0) for r in rows], label="L_TEA (weighted mean)")
    ax.plot(s, [_get(r, "tea.cap_hit_frac", 0) for r in rows], label="cap hit frac")
    ax2 = ax.twinx()
    ax2.plot(s, [_get(r, "loss.tea_term", 0) for r in rows], c="purple", alpha=0.5,
             label="λ·L_TEA in loss")
    ax2.plot(s, [_get(r, "loss.policy", 0) for r in rows], c="gray", alpha=0.5,
             label="policy loss")
    ax.set_title("6. TEA internals (009's unlogged-SED gap, not repeated)")
    ax.set_xlabel("step"); ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    # 7. optimizer
    ax = axes[3][0]
    ax.plot(s, [_get(r, "opt.theta_drift_rel", 0) for r in rows],
            label="theta_drift_rel", lw=2)
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(s, [_get(r, "opt.grad_norm", 0) for r in rows], c="orange", alpha=0.6,
             label="grad norm")
    ax2.plot(s, [_get(r, "kl.mean", 0) for r in rows], c="green", alpha=0.6,
             label="answer KL")
    ax.set_title("7. Optimizer — flat theta_drift = a no-op run")
    ax.set_xlabel("step"); ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    # 8. pipeline balance — the topology verdict
    ax = axes[3][1]
    ax.stackplot(
        s,
        [_get(r, "wall.rollout", 0) for r in rows],
        [_get(r, "wall.scoring", 0) for r in rows],
        [_get(r, "wall.trainer", 0) for r in rows],
        [_get(r, "wall.sync", 0) for r in rows],
        labels=["rollout (turing)", "scoring (cpu)", "trainer (spark)", "sync"],
    )
    ax.set_title("8. Pipeline balance — the topology verdict")
    ax.set_xlabel("step"); ax.set_ylabel("seconds"); ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(args.out_dir, "stagec_dashboard.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    def _med(key: str) -> float:
        v = [_get(r, key, 0) for r in rows]
        return float(statistics.median(v)) if v else 0.0

    summary = {
        "n_steps": len(rows),
        "wall_median": {k: _med(f"wall.{k}")
                        for k in ("total", "rollout", "scoring", "trainer", "sync")},
        "trainer_over_rollout": (_med("wall.trainer") / _med("wall.rollout")
                                 if _med("wall.rollout") else None),
        "acc_rate_first": _get(rows[0], "reward.acc_rate"),
        "acc_rate_last": _get(rows[-1], "reward.acc_rate"),
        "think_median_first": _get(rows[0], "reward.think_median"),
        "think_median_last": _get(rows[-1], "reward.think_median"),
        "answer_median_first": _get(rows[0], "reward.answer_median"),
        "answer_median_last": _get(rows[-1], "reward.answer_median"),
        "entropy_first": _get(rows[0], "tea.think_entropy_mean"),
        "entropy_last": _get(rows[-1], "tea.think_entropy_mean"),
        "empty_think_max": max(_get(r, "reward.empty_think_rate", 0) for r in rows),
        "lenient_only_max": max(_get(r, "reward.lenient_only_rate", 0) for r in rows),
        "drop_rate_mean": float(statistics.mean(r.get("drop_rate", 0) for r in rows)),
        "theta_drift_last": _get(rows[-1], "opt.theta_drift_rel"),
        "logp_old_mismatch_median": float(statistics.median(
            [r.get("logp_old_mismatch", 0) for r in rows])),
    }
    with open(os.path.join(args.out_dir, "stagec_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[dash] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
