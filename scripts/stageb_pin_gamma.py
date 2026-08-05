"""Pin the ZPD gate threshold gamma from the measured histogram (packet P6 Part 3).

Design §4.1's band-pass weight is

    gate_t  = sigmoid(kappa * (log pi_S(tau_t) - gamma))     # = sigmoid(kappa * (-S_t - gamma))
    nov_t   = 1 + alpha_nov * min(S_t, s_cap)
    w_t     = gate_t * nov_t

with kappa=1, alpha_nov=0.5, s_cap=4 nats and gamma initialised at ln(1e-4) ~ -9.21.
gamma is the one knob the design leaves to measurement, and activity 008's
histogram cannot pin it: that was measured under ``scorer_v1``, which has had 91%
of the register style tax removed, so it is a **lower bound** on masking. This
script re-measures under whatever pi_S actually produced the gate file.

Reported, because each answers a standing question rather than decorating a plot:

* **masked fraction** (gate < 0.1) overall and per level. Sanity band 5-30%.
  Above ~40% in the hard band is activity 006's fear materialising -- the 32B
  teacher reasoning outside a 1.7B student's reach -- and the honest response is
  to report it, not to slide gamma until the number looks better.
* **boosted fraction** (nov_t > 1.2), the share of tokens carrying real novelty
  inside the reachable zone. That is the band-pass's actual payload.
* **register-marker mean w_t at round start.** Under the original checkpoint
  ``goal`` sits ~40 nats out (activity 007 finding 1), so its weight is ~0 and
  the register cannot enter in epoch 1 by brute force. It is meant to arrive as
  pi_S catches up. This number is the step-0 reading of the "is the register
  flowing yet" curve the trainer plots; a flat zero through round 1 is an
  F3-relevant finding, not a reason to hack gamma mid-run.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import MARKER_CLASSES, marker_class_ids, percentile

KAPPA = 1.0
ALPHA_NOV = 0.5
S_CAP = 4.0
GAMMA_INIT = math.log(1e-4)          # ~ -9.2103


def weights(s: np.ndarray, gamma: float, kappa: float = KAPPA,
            alpha_nov: float = ALPHA_NOV, s_cap: float = S_CAP):
    """``(gate, nov, w)`` for an array of surprisals ``S_t = -log pi_S``."""
    gate = 1.0 / (1.0 + np.exp(-kappa * (-s - gamma)))
    nov = 1.0 + alpha_nov * np.minimum(s, s_cap)
    return gate, nov, gate * nov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--gates", required=True, help=".npz from stageb_zpd_gates.py")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dir_plots", default="")
    ap.add_argument("--gamma", type=float, default=GAMMA_INIT)
    ap.add_argument("--gamma_sweep", default="-11.5,-10.5,-9.2103,-8,-7,-6,-5",
                    help="candidate gammas to tabulate alongside the pinned one")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train)]
    npz = np.load(args.gates)
    print(f"[gamma] {len(rows)} records | {len(npz.files)} scored arrays", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    mclass = marker_class_ids(tok)
    marker_ids = frozenset().union(*mclass.values())
    print(f"[gamma] register marker ids: "
          + ", ".join(f"{k} {len(v)}" for k, v in mclass.items()), flush=True)

    think_s: list = []
    by_level: dict = collections.defaultdict(list)
    marker_s: dict = collections.defaultdict(list)
    answer_s: list = []

    for r in rows:
        key = f"{r['_uid']}#{r['trace_idx']}"
        if key not in npz:
            continue
        s = npz[key]
        p0 = r["prompt_len"]
        ts, te = r["think_start"] - p0, r["think_end"] - p0
        as_, ae = r["answer_start"] - p0, r["answer_end"] - p0
        seg = s[ts:te]
        think_s.append(seg)
        by_level[r["level"]].append(seg)
        answer_s.append(s[as_:ae])
        # Marker positions inside the think span, by class.
        ids = r["ids"]
        for cls, cls_ids in mclass.items():
            hit = [i - r["think_start"] for i in range(r["think_start"], r["think_end"])
                   if ids[i] in cls_ids]
            if hit:
                marker_s[cls].append(seg[hit])

    S = np.concatenate(think_s)
    A = np.concatenate(answer_s)
    print(f"[gamma] {len(S):,} think tokens | {len(A):,} answer tokens", flush=True)

    def summarize(s: np.ndarray, gamma: float) -> dict:
        gate, nov, w = weights(s, gamma)
        return {
            "n": int(len(s)),
            "masked_frac": round(float((gate < 0.1).mean()), 5),
            "boosted_frac": round(float((nov > 1.2).mean()), 5),
            "mean_w": round(float(w.mean()), 5),
            "mean_gate": round(float(gate.mean()), 5),
            "S_p50": round(percentile(s.tolist(), 50), 5),
            "S_p90": round(percentile(s.tolist(), 90), 5),
            "S_p99": round(percentile(s.tolist(), 99), 5),
        }

    gammas = [float(x) for x in args.gamma_sweep.split(",")]
    sweep = {f"{g:.4f}": summarize(S, g) for g in gammas}

    pinned = args.gamma
    overall = summarize(S, pinned)
    per_level = {str(k): summarize(np.concatenate(v), pinned)
                 for k, v in sorted(by_level.items())}
    answer = summarize(A, pinned)

    # Register markers: their weight at round start is the "is it flowing" zero point.
    markers = {}
    for cls in MARKER_CLASSES:
        if cls not in marker_s:
            continue
        ms = np.concatenate(marker_s[cls])
        gate, nov, w = weights(ms, pinned)
        markers[cls] = {
            "n": int(len(ms)),
            "share_of_think": round(float(len(ms) / len(S)), 5),
            "S_mean": round(float(ms.mean()), 4),
            "S_p50": round(percentile(ms.tolist(), 50), 4),
            "S_p90": round(percentile(ms.tolist(), 90), 4),
            "S_max": round(float(ms.max()), 4),
            "mean_gate": float(f"{gate.mean():.3e}"),
            "mean_w": float(f"{w.mean():.3e}"),
            "frac_masked": round(float((gate < 0.1).mean()), 5),
        }

    out = {
        "gamma_pinned": pinned,
        "kappa": KAPPA, "alpha_nov": ALPHA_NOV, "s_cap": S_CAP,
        # gate = sigmoid(k(-S - gamma)) < 0.1  <=>  -S - gamma < ln(1/9)
        #                                     <=>  S > -gamma + ln(9)/k
        "gate_lt_0.1_at_S_above": round(-pinned + math.log(9.0) / KAPPA, 4),
        "nov_gt_1.2_at_S_above": round(0.2 / ALPHA_NOV, 4),
        "think_overall": overall,
        "think_per_level": per_level,
        "answer_overall": answer,
        "markers": markers,
        "gamma_sweep": sweep,
        "gates_meta": json.load(open(f"{os.path.splitext(args.gates)[0]}.meta.json")),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"\n[gamma] pinned gamma = {pinned:.4f}  "
          f"(gate<0.1 above S={out['gate_lt_0.1_at_S_above']} nats; "
          f"boost above S={out['nov_gt_1.2_at_S_above']} nats)")
    print(f"[gamma] think overall: masked {100*overall['masked_frac']:.2f}%  "
          f"boosted {100*overall['boosted_frac']:.2f}%  mean_w {overall['mean_w']}")
    print("[gamma] per level:")
    for lv, d in per_level.items():
        print(f"    L{lv}  n={d['n']:>7,}  masked {100*d['masked_frac']:5.2f}%  "
              f"boosted {100*d['boosted_frac']:5.2f}%  mean_w {d['mean_w']:.4f}  "
              f"S_p90 {d['S_p90']:.2f}")
    print("[gamma] register markers at round start:")
    for cls, d in markers.items():
        print(f"    {cls:11s} n={d['n']:>6,} ({100*d['share_of_think']:.2f}% of think)  "
              f"S_mean {d['S_mean']:7.3f}  mean_w {d['mean_w']:.3e}  "
              f"masked {100*d['frac_masked']:.1f}%")
    print("[gamma] gamma sweep (think): " + "  ".join(
        f"{g}->{100*d['masked_frac']:.1f}%" for g, d in sweep.items()))

    if args.out_dir_plots:
        _plots(S, per_level, markers, pinned, args.out_dir_plots)
    print(f"\n[gamma] summary -> {args.out_json}")
    return 0


def _plots(S, per_level, markers, gamma, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[gamma] matplotlib unavailable — skipping plots", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.hist(np.clip(S, 0, 20), bins=120, color="#4C72B0", log=True)
    thr = -gamma + math.log(9.0)
    ax.axvline(thr, color="#C44E52", ls="--",
               label=f"gate<0.1 at S={thr:.2f}")
    ax.axvline(0.4, color="#55A868", ls=":", label="boost starts S=0.4")
    ax.set_xlabel("S_t = -log pi_S (nats), think tokens, clipped at 20")
    ax.set_ylabel("count (log)")
    ax.set_title(f"ZPD histogram under pi_S  (gamma={gamma:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.linspace(0, 20, 400)
    gate, nov, w = weights(xs, gamma)
    ax.plot(xs, gate, label="gate = sigmoid(k(-S-gamma))", color="#C44E52")
    ax.plot(xs, nov / nov.max(), label="novelty (normalised)", color="#55A868")
    ax.plot(xs, w / w.max(), label="w_t (normalised)", color="#4C72B0", lw=2)
    ax.set_xlabel("S_t (nats)")
    ax.set_ylabel("weight")
    ax.set_title("Band-pass shape")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "zpd_gamma.png")
    fig.savefig(p, dpi=130)
    print(f"[gamma] plot -> {p}", flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lv = sorted(per_level, key=int)
    ax.bar(range(len(lv)), [100 * per_level[k]["masked_frac"] for k in lv],
           color="#C44E52", label="masked (gate<0.1)")
    ax.bar(range(len(lv)), [100 * per_level[k]["boosted_frac"] for k in lv],
           bottom=[100 * per_level[k]["masked_frac"] for k in lv],
           color="#55A868", alpha=.75, label="boosted (nov>1.2)")
    ax.axhline(30, color="k", ls=":", lw=1, label="sanity band 5-30% masked")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels([f"L{k}" for k in lv])
    ax.set_ylabel("% of think tokens")
    ax.set_title(f"ZPD masking / boosting by level (gamma={gamma:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "zpd_by_level.png")
    fig.savefig(p, dpi=130)
    print(f"[gamma] plot -> {p}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
