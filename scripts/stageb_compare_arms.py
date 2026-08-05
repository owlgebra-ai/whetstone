"""Side-by-side comparison of Stage-B training arms (packet P6 Parts 4 and 6).

Reads each arm's ``stageb_metrics.jsonl`` and prints the panels that decide
between them, plus a dashboard. Used for two comparisons this packet owes:

* **whitelist floor vs `goal:`-stripped** (activity 009 findings 7 and 9) — the
  user asked for both to be run rather than deciding on the predicted outcome.
* **golden vs unfiltered control** (packet Part 6) — does judge-filtering earn
  its quota, and does trace diversity matter.

The assimilation panel is the generative spot-check, not the loss. Loss curves
across arms trained on *different corpora* are not comparable — the stripped
corpus has 4,514 fewer think tokens and no 40-nat token in it, so its CE is
lower for reasons that have nothing to do with whether assimilation worked.
What is comparable is what the student generates: think length, marker density,
g-rate and answer-segment leakage on the same 20 held-out problems.
"""

from __future__ import annotations

import argparse
import json
import os

CURVES = [
    ("spot_think_median", "think tokens (spot-check)", "log"),
    ("spot_think_marker_density", "register markers / 100 chars", "linear"),
    ("spot_g_rate", "g-rate", "linear"),
    ("control_entropy_mean", "control entropy, mean", "linear"),
    ("ce_weighted", "weighted CE", "linear"),
    ("sed_k2", "SED K2", "linear"),
]


def load(path: str) -> list:
    p = path if path.endswith(".jsonl") else os.path.join(path, "stageb_metrics.jsonl")
    return [json.loads(l) for l in open(p)]


def spot_rows(rows: list) -> list:
    return [r for r in rows if r.get("spot_think_median") is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable, e.g. --arm floor=/data/.../round1_golden")
    ap.add_argument("--baseline-think", type=float, default=1477.0,
                    help="eval-protocol baseline think median (activity 009 Part 0b)")
    ap.add_argument("--corpus-think", type=float, default=251.0,
                    help="corpus think median — the assimilation target")
    ap.add_argument("--corpus-markers", type=float, default=2.0)
    ap.add_argument("--out-plot", default="")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = load(path)

    print("=" * 96)
    print("ASSIMILATION PANEL — generative spot-check, 20 held-out problems, greedy")
    print(f"  target: think -> ~{args.corpus_think:.0f} (corpus), markers -> ~{args.corpus_markers:.1f}")
    print(f"  start : think ~{args.baseline_think:.0f} (baseline), markers ~0.16")
    print("=" * 96)
    for name, rows in arms.items():
        sr = spot_rows(rows)
        print(f"\n--- {name} ({len(rows)} evals, {len(sr)} spot-checks) ---")
        print(f"  {'step':>5} {'think':>8} {'answer':>8} {'g':>6} {'mark':>7} "
              f"{'leak':>7} {'H mean':>8} {'CE':>7} {'SED':>8} {'drift':>9}")
        for r in sr:
            print(f"  {r['step']:>5} {r['spot_think_median']:>8.0f} "
                  f"{r['spot_answer_median']:>8.0f} {r['spot_g_rate']:>6.2f} "
                  f"{(r['spot_think_marker_density'] or 0):>7.3f} "
                  f"{(r['spot_answer_leak_rate'] if r['spot_answer_leak_rate'] is not None else float('nan')):>7.2f} "
                  f"{r.get('control_entropy_mean', float('nan')):>8.4f} "
                  f"{r['ce_weighted']:>7.4f} {r['sed_k2']:>8.5f} "
                  f"{r['theta_drift_rel']:>9.2e}")

    print("\n" + "=" * 96)
    print("FINAL STATE")
    print("=" * 96)
    hdr = f"  {'arm':<14} {'steps':>6} {'think':>8} {'answer':>8} {'g':>6} {'mark':>7} {'leak':>7} {'H mean':>8} {'H p80':>8}"
    print(hdr)
    for name, rows in arms.items():
        sr = spot_rows(rows)
        if not sr:
            print(f"  {name:<14} (no spot-check rows)")
            continue
        r = sr[-1]
        print(f"  {name:<14} {r['step']:>6} {r['spot_think_median']:>8.0f} "
              f"{r['spot_answer_median']:>8.0f} {r['spot_g_rate']:>6.2f} "
              f"{(r['spot_think_marker_density'] or 0):>7.3f} "
              f"{(r['spot_answer_leak_rate'] if r['spot_answer_leak_rate'] is not None else float('nan')):>7.2f} "
              f"{r.get('control_entropy_mean', float('nan')):>8.4f} "
              f"{r.get('control_entropy_p80', float('nan')):>8.4f}")
    print(f"\n  reference: baseline think {args.baseline_think:.0f} | "
          f"F3b bar {args.baseline_think/2:.0f} | corpus {args.corpus_think:.0f}")
    print("  NOTE loss columns are NOT comparable across arms trained on different "
          "corpora;\n       the spot-check columns are.")

    if args.out_plot:
        _plot(arms, args)
    return 0


def _plot(arms: dict, args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib unavailable — skipping plot")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    for ax, (key, label, scale) in zip(axes.flat, CURVES):
        for i, (name, rows) in enumerate(arms.items()):
            pts = [(r["step"], r[key]) for r in rows
                   if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, label=name,
                        color=colors[i % len(colors)])
        if key == "spot_think_median":
            ax.axhline(args.baseline_think, color="k", ls="--", lw=1, label="baseline")
            ax.axhline(args.baseline_think / 2, color="#C44E52", ls=":", lw=1, label="F3b bar")
            ax.axhline(args.corpus_think, color="#55A868", ls=":", lw=1, label="corpus")
        if key == "spot_think_marker_density":
            ax.axhline(args.corpus_markers, color="#55A868", ls=":", lw=1, label="corpus")
        ax.set_yscale(scale)
        ax.set_xlabel("optimizer step")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_plot)), exist_ok=True)
    fig.savefig(args.out_plot, dpi=130)
    print(f"\n[compare] plot -> {args.out_plot}")


if __name__ == "__main__":
    raise SystemExit(main())
