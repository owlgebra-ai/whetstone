"""F4 screen: does any Stage-C checkpoint Pareto-dominate the init? (packet P7 §7.2)

F4's second clause is "≥1 checkpoint **Pareto-dominating the init** on the easy
suite (200-screen: strict Pass@1 up at equal-or-less think median)". This runs
the screen over a list of checkpoints and prints the Pareto table.

Protocol is the **gate/screen** row of the packet's sampling table — **T=0.7,
top-p 0.95** — not the training sampler. Training runs at T=1.0 for
policy-gradient correctness; comparability with 009's published numbers wins for
gates, and the packet is explicit that you never train against the eval protocol.

Cap is **8,192**, matching 009 Run 12 exactly. Activity 010 finding 4 raised the
*training* cap to 12,288 because the Phase-1 pool is half DeepMath — but this
screen is `gsm8k_test`, i.e. GSM8K only, which is the distribution 009 measured
0/377 truncation on. Same suite, same protocol, same cap ⇒ the numbers are
directly comparable to the baselines below.

Reference points (activity 009 Run 12/13, same 200 problems, same protocol):

    round-1 init : strict Pass@1 64.25%  as-scored 66.50%  think median 218
    original ckpt: strict Pass@1 ~90.2%  (the 4B-away baseline, for context)

Usage (turing)::

    python scripts/stagec_f4_screen.py \\
        --checkpoints /data/whetstone/runs/stagec/pilot/ckpt/step0020 \\
                      /data/whetstone/runs/stagec/pilot/ckpt/step0040 \\
        --out_dir /data/whetstone/runs/stagec/pilot/f4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Activity 009 Run 12/13 on the identical 200 problems and protocol.
INIT_STRICT_P1 = 0.6425
INIT_AS_SCORED_P1 = 0.6650
INIT_THINK_MEDIAN = 218.0

SCREEN_UIDS = "/data/whetstone/runs/stagec_buckets/screen200_uids.json"
SCREEN_POOL = "/data/whetstone/eval/gsm8k_test.jsonl"


def run_screen(ckpt: str, out_dir: str, k: int, force: bool) -> dict:
    d = os.path.join(out_dir, os.path.basename(os.path.normpath(ckpt)))
    summary = os.path.join(d, "buckets_summary.json")
    if not os.path.exists(summary) or force:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "stagec_bucket.py"),
            "--model", ckpt, "--uids", SCREEN_UIDS, "--pool", SCREEN_POOL,
            "--out_dir", d, "--K", str(k),
            # gate protocol, NOT the training sampler
            "--temperature", "0.7", "--top_p", "0.95", "--max_tokens", "8192",
            "--max_model_len", "9216",
        ]
        if force:
            cmd.append("--force")
        print(f"[f4] screening {ckpt}", flush=True)
        subprocess.run(cmd, check=True)
    with open(summary) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [{
        "name": "init (round-1 student, 009 Run 12/13)",
        "strict_p1": INIT_STRICT_P1, "as_scored_p1": INIT_AS_SCORED_P1,
        "think_median": INIT_THINK_MEDIAN, "pass_at_k": 0.8950,
        "cap_hit": 0.0269, "g_rate": 0.9594, "is_init": True,
    }]
    for c in args.checkpoints:
        s = run_screen(c, args.out_dir, args.K, args.force)
        o, cs = s["overall"], s["candidate_stats"]
        rows.append({
            "name": os.path.basename(os.path.normpath(c)),
            "strict_p1": o["pass_at_1_strict"],
            "as_scored_p1": o["pass_at_1_as_scored"],
            "think_median": cs["think_median"],
            "answer_median": cs["answer_median"],
            "pass_at_k": o["pass_at_k_strict"],
            "cap_hit": cs["cap_hit_rate"], "g_rate": cs["g_rate"],
            "lenient_only": cs["lenient_only_rate"],
            "mixed_frac": o["mixed_frac"],
            "is_init": False,
        })

    print(f"\n{'checkpoint':<40} {'strictP@1':>10} {'as-scored':>10} "
          f"{'think':>7} {'pass@K':>8} {'g':>7} {'Pareto?':>9}")
    dominating = []
    for r in rows:
        tag = "— init —" if r["is_init"] else ""
        if not r["is_init"]:
            dom = (r["strict_p1"] > INIT_STRICT_P1
                   and r["think_median"] <= INIT_THINK_MEDIAN)
            tag = "**YES**" if dom else "no"
            if dom:
                dominating.append(r["name"])
        print(f"{r['name']:<40} {100*r['strict_p1']:9.2f}% {100*r['as_scored_p1']:9.2f}% "
              f"{r['think_median']:7.0f} {100*r['pass_at_k']:7.2f}% "
              f"{100*r['g_rate']:6.2f}% {tag:>9}")

    verdict = {
        "criterion": ("strict Pass@1 > init AND think median <= init "
                      "(packet P7 §7.2)"),
        "init": {"strict_p1": INIT_STRICT_P1, "think_median": INIT_THINK_MEDIAN},
        "checkpoints": rows[1:],
        "pareto_dominating": dominating,
        "f4_clause2_pass": bool(dominating),
    }
    with open(os.path.join(args.out_dir, "f4_screen.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\n[f4] Pareto-dominating checkpoints: {dominating or 'NONE'}")
    print(f"[f4] F4 clause 2 (Pareto): {'PASS' if dominating else 'FAIL'}")
    print("[f4] clause 1 (50 steps, no critical rollout flag) is judged from the "
          "dashboards + rollout reading, not from this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
