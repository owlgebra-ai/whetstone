"""Draft-level statistics over the Stage-A raw corpus (packet P5 Parts 5 and 6).

The raw corpus is the denominator for everything F2 claims, so this is the view
selection cannot give: what the teacher produced *before* anything was chosen.
Used twice — at the 500-problem calibration checkpoint to pin thresholds, and
over the complete corpus for F2a.

**R_acc is reported over two denominators, deliberately.** The packet's floor
("per-level R_acc below ~90% ⇒ stop and inspect prompts") is a statement about
the *prompt*, so its denominator is drafts that produced a parseable rollout —
a cap-hit draft has no answer segment to verify and says nothing about whether
the teacher can derive the gold. Reporting only "verified / all drafts" would
conflate a budget problem with a reasoning problem; reporting only "verified /
gate-passing" would hide a teacher that has stopped finishing. Both are printed,
and the gate rate alongside them.

The trap this avoids: filtering on ``reject_reason is None`` and then computing
a verify rate over the survivors returns 100.00% by construction, because
``verify_fail`` *is* one of the rejection reasons. That number looks like a
result and is a tautology.

Usage::

    python scripts/stagea_draft_stats.py \\
        --drafts /data/whetstone/corpora/stagea_raw/drafts.jsonl \\
        --out    /data/whetstone/runs/stagea/draft_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import MARKER_CLASSES, percentile

ALL_MARKERS = tuple(m for cls in MARKER_CLASSES.values() for m in cls)
#: A draft that cleared the segment gate and the card-§1.5 boxed check produced
#: a real, parseable rollout. Everything else never got as far as an answer.
GATE_FAILURES = ("cap_think", "cap_answer", "empty_think")


def gate_passed(r: dict) -> bool:
    rr = r.get("reject_reason")
    return rr is None or rr == "verify_fail"


def density(think: str) -> float:
    return 100.0 * sum(think.count(m) for m in ALL_MARKERS) / max(1, len(think))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_r_acc", type=float, default=90.0,
                    help="packet §5 step 3 floor, in percent, per level")
    args = ap.parse_args()

    rows = []
    with open(args.drafts) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise SystemExit(f"[stats] no drafts in {args.drafts}")

    by_level: dict = defaultdict(list)
    for r in rows:
        by_level[r.get("level")].append(r)
    levels = sorted(by_level, key=lambda x: (x is None, x))

    print(f"[in] {len(rows)} drafts over "
          f"{len({r['_uid'] for r in rows})} problems\n")
    print(f"  {'lvl':>3} {'drafts':>7} {'gated':>7} {'gate%':>6} "
          f"{'R_acc/gated':>12} {'R_acc/all':>10} {'kept':>6} {'kept%':>6}")

    per_level, breaches = {}, []
    for lv in levels:
        rs = by_level[lv]
        gated = [r for r in rs if gate_passed(r)]
        kept = [r for r in rs if r.get("reject_reason") is None]
        r_gated = 100.0 * len(kept) / len(gated) if gated else float("nan")
        r_all = 100.0 * len(kept) / len(rs)
        print(f"  {str(lv):>3} {len(rs):>7} {len(gated):>7} "
              f"{100*len(gated)/len(rs):>5.1f}% {r_gated:>11.1f}% "
              f"{r_all:>9.1f}% {len(kept):>6} {100*len(kept)/len(rs):>5.1f}%")
        per_level[str(lv)] = {
            "drafts": len(rs), "gate_passed": len(gated), "kept": len(kept),
            "r_acc_over_gated_pct": round(r_gated, 2),
            "r_acc_over_all_pct": round(r_all, 2),
            "rejects": dict(Counter(r["reject_reason"] for r in rs
                                    if r.get("reject_reason"))),
        }
        if gated and r_gated < args.min_r_acc:
            breaches.append((lv, r_gated))

    rejects = Counter(r["reject_reason"] for r in rows if r.get("reject_reason"))
    flags = Counter(f for r in rows for f in (r.get("clean_flags") or []))
    kept_all = [r for r in rows if r.get("reject_reason") is None]
    gated_all = [r for r in rows if gate_passed(r)]

    print(f"\n  rejects: {dict(rejects.most_common())}")
    print(f"  clean flags (trimmed, not rejected): {dict(flags.most_common())}")
    print(f"  OVERALL  gate {100*len(gated_all)/len(rows):.1f}%   "
          f"R_acc/gated {100*len(kept_all)/max(1,len(gated_all)):.2f}%   "
          f"R_acc/all {100*len(kept_all)/len(rows):.2f}%")

    def dist(vals, name, extra=""):
        vals = [v for v in vals if v]
        if not vals:
            return {}
        d = {"median": percentile(vals, 50), "p25": percentile(vals, 25),
             "p75": percentile(vals, 75), "p95": percentile(vals, 95),
             "p99": percentile(vals, 99), "max": max(vals)}
        print(f"  {name:<16} median {d['median']:.0f}  IQR "
              f"[{d['p25']:.0f}, {d['p75']:.0f}]  p95 {d['p95']:.0f}  "
              f"p99 {d['p99']:.0f}  max {d['max']:.0f}{extra}")
        return d

    print()
    think = dist([r.get("think_tokens") for r in kept_all], "think tokens",
                 "   (B_target 600)")
    answer = dist([r.get("answer_tokens") for r in kept_all], "answer tokens")
    dens = [density(r.get("compact_think", "")) for r in kept_all]
    dens_med = percentile(dens, 50)
    print(f"  {'markers/100ch':<16} median {dens_med:.2f}  "
          f"mean {sum(dens)/len(dens):.2f}   (32B single-draft baseline 2.10)")

    cond = Counter(r.get("conditioned_on") for r in kept_all)
    print(f"  conditioning (kept): {dict(cond)}")

    # Per-problem survivor counts — this is what selection has to work with.
    surv = Counter()
    for r in kept_all:
        surv[r["_uid"]] += 1
    hist = Counter(surv.values())
    attempted = {r["_uid"] for r in rows}
    zero = len(attempted) - len(surv)
    print(f"\n  survivors per problem: "
          f"{dict(sorted(hist.items()))}  (+{zero} with zero)")

    verdict = "PASS" if not breaches else "BREACH"
    print(f"\n  packet §5 step-3 floor (R_acc/gated >= {args.min_r_acc}% "
          f"per level): {verdict}")
    for lv, v in breaches:
        print(f"    !! level {lv}: {v:.1f}%")

    out = {
        "n_drafts": len(rows), "n_problems": len(attempted),
        "gate_pct": round(100 * len(gated_all) / len(rows), 2),
        "r_acc_over_gated_pct": round(100 * len(kept_all) / max(1, len(gated_all)), 2),
        "r_acc_over_all_pct": round(100 * len(kept_all) / len(rows), 2),
        "rejects": dict(rejects), "clean_flags": dict(flags),
        "think_tokens": think, "answer_tokens": answer,
        "marker_density_median": round(dens_med, 3),
        "conditioning_kept": dict(cond),
        "survivors_per_problem": {str(k): v for k, v in sorted(hist.items())},
        "problems_with_zero_survivors": zero,
        "per_level": per_level,
        "r_acc_floor_pct": args.min_r_acc,
        "r_acc_floor_verdict": verdict,
        "r_acc_floor_breaches": [{"level": lv, "pct": round(v, 2)}
                                 for lv, v in breaches],
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
