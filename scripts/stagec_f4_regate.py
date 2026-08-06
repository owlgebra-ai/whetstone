"""F4 re-gate (packet P7b §6): K-draw means, paired McNemar, init re-screened.

Replaces the verdict logic of :mod:`scripts.stagec_f4_screen`, which carried
both halves of activity 010 finding 22: it quoted the init from activity 009's
journal instead of re-screening it, and it read ``pass_at_1_strict`` — a
**single-draw** statistic (candidate 0 only) — against it. Two definitions of
"Pass@1" differ by 4 points on this very suite and manufactured a +4.75-point
"gain" that recomputation turned into −2.19. The rules this script hard-codes:

* **The init goes through the identical harness in the same session.** Never a
  journal number.
* **Every Pass@1 is the K-draw mean ± the between-draw std.** Draw *k*'s
  Pass@1 is the mean over problems of candidate *k*'s strict flag; the reported
  number is the mean over the K draws.
* **Paired McNemar per problem per draw** against the init: wins = arm correct
  where init wrong on the same (problem, draw); z = (w−l)/√(w+l).
* **think-per-correct** = total think tokens spent / strict-correct rollouts
  produced — pilot 1's cleanest inverse signal (331 → 367 monotone worsening).

Verdict (packet P7b §6, clause 2 + the EXTEND rule):

* **PASS** — ≥1 checkpoint Pareto-dominates the init: strict Pass@1 up
  (K-draw mean, paired delta positive) at equal-or-less think median.
* **EXTEND** — no Pareto dominance but no degradation either: every checkpoint
  within ±1σ of the init's Pass@1 (σ = the init's between-draw std) and think
  median not above the init's. Flat-but-healthy is not FAIL.
* **FAIL** — degradation on any axis (Pass@1 below −1σ, or think median up).

Clause 1 (training-sampler trajectory) is judged from ``train_log.jsonl``:
pass ``--train_log`` to print `missing_think_close` / H / g by 10-step window.

Usage (turing)::

    python scripts/stagec_f4_regate.py \\
        --init /data/whetstone/ckpt/stageb/golden/round1/final \\
        --checkpoints /data/whetstone/runs/stagec/pilot2_armA/ckpt/step00{25,50,75,100} \\
        --out_dir /data/whetstone/runs/stagec/pilot2_armA/f4 \\
        --train_log /data/whetstone/runs/stagec/pilot2_armA/train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl

SCREEN_UIDS = "/data/whetstone/runs/stagec_buckets/screen200_uids.json"
SCREEN_POOL = "/data/whetstone/eval/gsm8k_test.jsonl"


def run_screen(ckpt: str, out_dir: str, k: int, force: bool, tag: str) -> str:
    """Run the gate-protocol screen; returns the buckets.jsonl path."""
    d = os.path.join(out_dir, tag)
    per_problem = os.path.join(d, "buckets.jsonl")
    if not os.path.exists(per_problem) or force:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "stagec_bucket.py"),
            "--model", ckpt, "--uids", SCREEN_UIDS, "--pool", SCREEN_POOL,
            "--out_dir", d, "--K", str(k),
            # gate protocol (T=0.7 / top-p 0.95 / cap 8192) — NOT the training sampler
            "--temperature", "0.7", "--top_p", "0.95", "--max_tokens", "8192",
            "--max_model_len", "9216",
        ]
        if force:
            cmd.append("--force")
        print(f"[regate] screening {tag}: {ckpt}", flush=True)
        subprocess.run(cmd, check=True)
    return per_problem


def screen_stats(per_problem_path: str) -> dict:
    """K-draw statistics from one screen's buckets.jsonl."""
    rows = read_jsonl(per_problem_path)
    K = min(len(r["candidates"]) for r in rows)
    strict_by_draw = [
        statistics.mean(r["candidates"][k]["strict"] for r in rows)
        for k in range(K)
    ]
    as_by_draw = [
        statistics.mean(
            bool(r["candidates"][k]["as_scored"]) and r["candidates"][k]["g"] == 1
            for r in rows)
        for k in range(K)
    ]
    all_c = [c for r in rows for c in r["candidates"]]
    n_correct = sum(c["strict"] for c in all_c)
    total_think = sum(c["think_tokens"] for c in all_c)
    return {
        "n_problems": len(rows), "K": K,
        "strict_p1_mean": statistics.mean(strict_by_draw),
        "strict_p1_std": statistics.stdev(strict_by_draw) if K > 1 else 0.0,
        "as_scored_p1_mean": statistics.mean(as_by_draw),
        "as_scored_p1_std": statistics.stdev(as_by_draw) if K > 1 else 0.0,
        "pass_at_k_strict": statistics.mean(r["n_strict"] > 0 for r in rows),
        "think_median": statistics.median(
            [c["think_tokens"] for c in all_c if c["g"] == 1] or [0]),
        "answer_median": statistics.median(
            [c["answer_tokens"] for c in all_c if c["g"] == 1] or [0]),
        "g_rate": statistics.mean(c["g"] for c in all_c),
        "cap_hit_rate": statistics.mean(c["finish_reason"] == "length" for c in all_c),
        "lenient_only_rate": statistics.mean(
            bool(c["as_scored"]) and not c["strict"] and c["g"] == 1 for c in all_c),
        "think_per_correct": total_think / n_correct if n_correct else float("inf"),
        # keyed verdicts for pairing
        "_verdicts": {r["_uid"]: [bool(c["strict"]) for c in r["candidates"][:K]]
                      for r in rows},
    }


def mcnemar(init: dict, arm: dict) -> dict:
    """Paired per problem per draw. Requires identical uid sets and K."""
    wins = losses = 0
    common = set(init["_verdicts"]) & set(arm["_verdicts"])
    if len(common) != len(init["_verdicts"]) or len(common) != len(arm["_verdicts"]):
        print(f"[regate] WARNING: pairing on {len(common)} common problems "
              f"(init {len(init['_verdicts'])}, arm {len(arm['_verdicts'])})",
              flush=True)
    K = min(init["K"], arm["K"])
    for uid in common:
        for k in range(K):
            i, a = init["_verdicts"][uid][k], arm["_verdicts"][uid][k]
            if a and not i:
                wins += 1
            elif i and not a:
                losses += 1
    n = wins + losses
    z = (wins - losses) / math.sqrt(n) if n else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))          # two-sided normal approx
    return {"wins": wins, "losses": losses, "z": z, "p": p}


def clause1_table(train_log: str, window: int = 10) -> list:
    """Training-sampler trajectory by step window (F4 clause 1's instrument)."""
    recs = [r for r in read_jsonl(train_log) if "reward" in r]
    out = []
    for lo in range(0, len(recs), window):
        w = recs[lo:lo + window]
        if not w:
            continue
        def _m(path):
            vals = []
            for r in w:
                v = r
                for part in path:
                    v = v.get(part) if isinstance(v, dict) else None
                    if v is None:
                        break
                if v is not None:
                    vals.append(v)
            return statistics.mean(vals) if vals else None
        out.append({
            "steps": f"{w[0]['step']}-{w[-1]['step']}",
            "missing_think_close": _m(("format", "missing_think_close_rate")),
            "g_rate_all": _m(("format", "g_rate_all_cands")),
            "g_rate_kept": _m(("reward", "g_rate")),
            "H_think": _m(("tea", "think_entropy_mean")),
            "acc": _m(("reward", "acc_rate")),
            "batch_p_hat": _m(("batch_p_hat_mean",)),
            "word_stutter": _m(("reward", "word_stutter_rate")),
            "clip_high": _m(("clip", "high")), "clip_low": _m(("clip", "low")),
            "answer_median": _m(("reward", "answer_median")),
            "theta_drift": _m(("opt", "theta_drift_rel")),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True,
                    help="the init checkpoint — RE-SCREENED here, never quoted")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_log", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    init_pp = run_screen(args.init, args.out_dir, args.K, args.force, "init")
    init_s = screen_stats(init_pp)

    arms = []
    for c in args.checkpoints:
        tag = os.path.basename(os.path.normpath(c))
        pp = run_screen(c, args.out_dir, args.K, args.force, tag)
        s = screen_stats(pp)
        s["name"] = tag
        s["mcnemar"] = mcnemar(init_s, s)
        s["delta_p1_pts"] = 100 * (s["strict_p1_mean"] - init_s["strict_p1_mean"])
        s["delta_think"] = s["think_median"] - init_s["think_median"]
        s["delta_think_per_correct"] = (
            s["think_per_correct"] - init_s["think_per_correct"])
        s["pareto"] = (s["strict_p1_mean"] > init_s["strict_p1_mean"]
                       and s["think_median"] <= init_s["think_median"])
        arms.append(s)

    sigma = 100 * init_s["strict_p1_std"]
    print(f"\n{'arm':<12} {'strict P@1 (8-draw)':>22} {'think':>6} {'ans':>5} "
          f"{'p@8':>7} {'g':>7} {'thk/corr':>9} {'ΔP@1':>7} {'McNemar z,p':>14} "
          f"{'Pareto':>7}")
    row = init_s
    print(f"{'init':<12} {100*row['strict_p1_mean']:9.2f}% ± {100*row['strict_p1_std']:4.2f}"
          f"{'':>6} {row['think_median']:6.0f} {row['answer_median']:5.0f} "
          f"{100*row['pass_at_k_strict']:6.2f}% {100*row['g_rate']:6.2f}% "
          f"{row['think_per_correct']:9.1f} {'—':>7} {'—':>14} {'—':>7}")
    for s in arms:
        m = s["mcnemar"]
        print(f"{s['name']:<12} {100*s['strict_p1_mean']:9.2f}% ± {100*s['strict_p1_std']:4.2f}"
              f"{'':>6} {s['think_median']:6.0f} {s['answer_median']:5.0f} "
              f"{100*s['pass_at_k_strict']:6.2f}% {100*s['g_rate']:6.2f}% "
              f"{s['think_per_correct']:9.1f} {s['delta_p1_pts']:+7.2f} "
              f"{m['z']:+5.2f},{m['p']:5.3f} "
              f"{'**YES**' if s['pareto'] else 'no':>7}")

    dominating = [s["name"] for s in arms if s["pareto"]]
    degraded = [s["name"] for s in arms
                if s["delta_p1_pts"] < -sigma or s["delta_think"] > 0]
    # PASS: any checkpoint Pareto-dominates. Otherwise the arm is judged by its
    # ENDPOINT (a mid-run dip that later recovers is not degradation): endpoint
    # degrading on either axis = FAIL; endpoint flat within ±1σ at
    # equal-or-less think = EXTEND (flat-but-healthy is not FAIL — pilot 1
    # moved θ by only ~1e-4 relative; run the same arm to 200 steps first).
    if dominating:
        verdict = "PASS"
    elif arms and arms[-1]["name"] in degraded:
        verdict = "FAIL"
    else:
        verdict = "EXTEND"

    out = {
        "protocol": {"uids": SCREEN_UIDS, "pool": SCREEN_POOL, "K": args.K,
                     "temperature": 0.7, "top_p": 0.95, "max_tokens": 8192,
                     "rule": "K-draw mean ± between-draw std; init re-screened "
                             "same harness same session (010 f22)"},
        "init": {k: v for k, v in init_s.items() if not k.startswith("_")},
        "checkpoints": [{k: v for k, v in s.items() if not k.startswith("_")}
                        for s in arms],
        "pareto_dominating": dominating,
        "degraded": degraded,
        "sigma_pts": sigma,
        "clause2_verdict": verdict,
    }
    if args.train_log and os.path.exists(args.train_log):
        out["clause1_windows"] = clause1_table(args.train_log)
        print("\nclause 1 — training-sampler trajectory (10-step windows):")
        print(f"{'steps':<10} {'mtc':>7} {'g(all)':>7} {'H':>7} {'acc':>6} "
              f"{'p̂':>6} {'stutter':>8} {'ans_med':>8} {'drift':>9}")
        for w in out["clause1_windows"]:
            def _f(x, fmt):
                return fmt.format(x) if x is not None else "  —"
            print(f"{w['steps']:<10} {_f(w['missing_think_close'], '{:7.3f}')} "
                  f"{_f(w['g_rate_all'], '{:7.3f}')} {_f(w['H_think'], '{:7.3f}')} "
                  f"{_f(w['acc'], '{:6.2f}')} {_f(w['batch_p_hat'], '{:6.2f}')} "
                  f"{_f(w['word_stutter'], '{:8.4f}')} "
                  f"{_f(w['answer_median'], '{:8.0f}')} "
                  f"{_f(w['theta_drift'], '{:9.2e}')}")

    with open(os.path.join(args.out_dir, "f4_regate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[regate] Pareto-dominating: {dominating or 'NONE'}")
    print(f"[regate] clause 2 verdict: **{verdict}**"
          + (" (flat-but-healthy → extend to 200 steps before judging)"
             if verdict == "EXTEND" else ""))
    print("[regate] clause 1 (monotone-worsening named failure) is judged from "
          "the windows above + rollout reading; the verdict in the journal "
          "combines both clauses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
