"""Compute the F3 verdict for a Stage-B student (packet P6 Part 5).

Reads the Part-0b baseline card, the student's eval summary and the student's
re-run entropy audit, and emits all four sub-gates with the numbers each was
decided on. Nothing here re-measures: it compares artifacts, so a verdict can
always be traced to the run that produced it.

    F3a  accuracy   Pass@1 within 1 pt of baseline
    F3b  length     median think <= 50% of baseline; answer stays in band
    F3c  entropy    student's own fresh rollouts above the audit baseline
    F3d  form       g-rate >= 99%, register leakage ~ 0

Two things this script refuses to let slide, both from activity 009:

* **Protocol match.** F3a/F3b compare against the eval-protocol baseline
  (gsm8k_test, T=0.7, 32k cap, think median 1,477); F3c compares against the
  entropy-audit baseline (val_2k, T=0.9, 16k cap, think median 6,099). The two
  differ by 4x and quoting the wrong one sets F3b's bar at ~3,000 instead of
  738. Sampling params are checked against the baseline and a mismatch is a hard
  error, not a footnote.
* **Total output length is reported beside the two segments.** The corpus's
  answers are longer than the baseline's, so the think-only number overstates
  the real reduction by ~2.5x (finding 6). F3b is still decided on think, per the
  packet, but the verdict prints all three so nobody quotes the flattering one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROTOCOL_KEYS = ("K", "temperature", "top_p", "max_tokens", "enable_thinking")


def _suite(summary: dict, name: str) -> dict:
    for s in summary["suites"]:
        if s["suite"] == name:
            return s
    raise SystemExit(f"suite {name} not in {list(x['suite'] for x in summary['suites'])}")


def _check_protocol(base: dict, stud: dict) -> None:
    bad = {k: (base.get(k), stud.get(k)) for k in PROTOCOL_KEYS
           if base.get(k) != stud.get(k)}
    if bad:
        raise SystemExit(
            "PROTOCOL MISMATCH — refusing to emit a verdict.\n"
            + "\n".join(f"  {k}: baseline {b!r} vs student {s!r}" for k, (b, s) in bad.items())
            + "\nAn accuracy or length comparison across protocols is not a comparison."
        )
    if stud.get("limited") or base.get("limited"):
        raise SystemExit("one of these runs is --limit'ed; not a suite number")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="Part-0b summary__*.json")
    ap.add_argument("--student", required=True, help="student's summary__*.json")
    ap.add_argument("--suite", default="gsm8k_test")
    ap.add_argument("--baseline-audit",
                    default="/data/whetstone/runs/entropy_audit/audit.json")
    ap.add_argument("--student-audit", default="",
                    help="student's entropy_audit.py audit.json (F3c). Without it "
                         "F3c is reported UNMEASURED, never PASS.")
    ap.add_argument("--acc-tol-pts", type=float, default=1.0)
    ap.add_argument("--length-frac", type=float, default=0.50)
    ap.add_argument("--answer-band-frac", type=float, default=0.50,
                    help="answer median must stay within +/- this fraction of "
                         "baseline; activity 009 finding 6 expects growth, so the "
                         "band is two-sided by design")
    ap.add_argument("--g-min", type=float, default=0.99)
    ap.add_argument("--leak-max", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    b = _suite(json.load(open(args.baseline)), args.suite)
    s = _suite(json.load(open(args.student)), args.suite)
    _check_protocol(b, s)

    gates: dict = {}

    # --- F3a accuracy -----------------------------------------------------
    d_acc = 100 * (s["pass_at_1_mean"] - b["pass_at_1_mean"])
    gates["F3a"] = {
        "name": "accuracy within 1 pt",
        "baseline": f"{100*b['pass_at_1_mean']:.2f}% ± {100*b['pass_at_1_std']:.2f}",
        "student": f"{100*s['pass_at_1_mean']:.2f}% ± {100*(s['pass_at_1_std'] or 0):.2f}",
        "delta_pts": round(d_acc, 3),
        "threshold": f">= {100*b['pass_at_1_mean'] - args.acc_tol_pts:.2f}%",
        "pass": d_acc >= -args.acc_tol_pts,
    }

    # --- F3b length -------------------------------------------------------
    bt, ba = b["think_tokens_median"], b["answer_tokens_median"]
    st, sa = s["think_tokens_median"], s["answer_tokens_median"]
    think_ok = st <= args.length_frac * bt
    ans_ok = abs(sa - ba) <= args.answer_band_frac * ba
    gates["F3b"] = {
        "name": "think <= 50% of baseline; answer stays in band",
        "baseline_think": bt, "student_think": st,
        "think_ratio": round(st / bt, 4), "think_threshold": round(args.length_frac * bt, 1),
        "baseline_answer": ba, "student_answer": sa,
        "answer_change_pct": round(100 * (sa - ba) / ba, 1),
        "baseline_total": bt + ba, "student_total": st + sa,
        "total_reduction_x": round((bt + ba) / max(st + sa, 1e-9), 2),
        "think_only_reduction_x": round(bt / max(st, 1e-9), 2),
        "pass": bool(think_ok and ans_ok),
        "think_pass": bool(think_ok), "answer_pass": bool(ans_ok),
    }

    # --- F3c entropy ------------------------------------------------------
    ab = json.load(open(args.baseline_audit))["think"]
    if args.student_audit:
        a_s = json.load(open(args.student_audit))["think"]
        gates["F3c"] = {
            "name": "median think entropy above the audit baseline (restoration)",
            "baseline_median": round(ab["p50"], 6), "student_median": round(a_s["p50"], 6),
            "baseline_mean": round(ab["mean"], 6), "student_mean": round(a_s["mean"], 6),
            "baseline_p80": round(ab["p80"], 6), "student_p80": round(a_s["p80"], 6),
            "baseline_collapse_mass": round(ab["collapse_mass_lt_0.1"], 4),
            "student_collapse_mass": round(a_s["collapse_mass_lt_0.1"], 4),
            "pass": a_s["p50"] > ab["p50"],
            "mean_also_up": a_s["mean"] > ab["mean"],
        }
    else:
        gates["F3c"] = {"name": "entropy", "pass": None,
                        "note": "UNMEASURED — run entropy_audit.py on the student "
                                "with the pinned protocol (n=200, seed 0, T=0.9, "
                                "top_p 0.95, max_tokens 16384)"}

    # --- F3d form ---------------------------------------------------------
    leak = s.get("answer_leak_rate")
    gates["F3d"] = {
        "name": "g-rate >= 99%, leakage ~ 0",
        "baseline_g": round(b["g_rate"], 4), "student_g": round(s["g_rate"], 4),
        "cap_hit": round(s["cap_hit_rate"], 4),
        "answer_leak_rate": leak,
        "pass": (s["g_rate"] >= args.g_min
                 and (leak is None or leak <= args.leak_max)),
        "leak_measured": leak is not None,
    }

    decided = [g for g in gates.values() if g["pass"] is not None]
    overall = (len(decided) == len(gates)) and all(g["pass"] for g in decided)

    print("=" * 72)
    print(f"F3 VERDICT — {args.suite}")
    print("=" * 72)
    for k, g in gates.items():
        mark = {True: "PASS", False: "FAIL", None: "UNMEASURED"}[g["pass"]]
        print(f"\n{k}  [{mark}]  {g['name']}")
        for kk, vv in g.items():
            if kk not in ("name", "pass"):
                print(f"      {kk:26s} {vv}")
    print("\n" + "=" * 72)
    if all(g["pass"] is not None for g in gates.values()):
        print(f"**F3 {'PASS' if overall else 'FAIL'}**")
    else:
        print("**F3 INCOMPLETE** — at least one sub-gate is unmeasured; "
              "an unmeasured gate is never a pass.")
    print("=" * 72)

    out = {"suite": args.suite, "gates": gates, "overall_pass": overall,
           "baseline_summary": args.baseline, "student_summary": args.student}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"\n-> {args.out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
