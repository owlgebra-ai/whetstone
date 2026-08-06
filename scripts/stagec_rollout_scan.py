"""Per-checkpoint rollout investigation, made reusable (v1 §7.6–7.7; 010 Run 11).

Activity 009 finding 11 is the standing law: **only generative inspection
catches death; losses will look fine.** This script does the mechanical half —
scan every training rollout for the named rot patterns, windowed by step so a
*worsening* pattern is visible — and stages the human half by dumping verbatim
samples per category for reading.

The categories are 010 Run 11's table plus the reward-integrity views the
pilot used (lenient-only, penalty firings). Detectors are imported from
:mod:`whetstone.reward.stagec` — the scan and the reward must see the same
thing, or the scan validates an instrument that is not the one training.

Usage (either box; CPU only)::

    python scripts/stagec_rollout_scan.py \\
        --run_dir /data/whetstone/runs/stagec/pilot2_armA \\
        --dump_dir /data/whetstone/runs/stagec/pilot2_armA/scan \\
        --window 20 --samples 5
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.reward.extract import split_think_close
from whetstone.reward.stagec import (
    detect_answer_repeat,
    detect_contradiction,
    detect_ngram_loop,
    detect_register_leak,
    detect_word_stutter,
)
from whetstone.reward.strict import verify_strict

_CASE_RE = re.compile(r"^case \d+:", re.MULTILINE)
_CHK_RE = re.compile(r"^chk:", re.MULTILINE)


def scan_candidate(c: dict, gold: str) -> dict:
    """All named patterns for one rollout. Returns flag dict."""
    text = c["text"]
    split = split_think_close(text)
    think, post = split.think, split.post_think
    v = verify_strict(text, gold)
    flags = {
        "empty_think": c["g"] == 1 and c["think_len"] < 16,
        "missing_think_close": c["gate_reason"] == "missing_think_close"
        and c["finish_reason"] != "length",
        "missing_think_open": c["gate_reason"] == "missing_think_open",
        "cap_hit": c["finish_reason"] == "length",
        "ngram_loop": bool(detect_ngram_loop(think)["fired"]),
        "case_enum_20": len(_CASE_RE.findall(think)) >= 20,
        "chk_chain_15": len(_CHK_RE.findall(think)) >= 15,
        "register_leak": c["g"] == 1 and bool(detect_register_leak(post)["fired"]),
        "answer_repeat": c["g"] == 1 and bool(detect_answer_repeat(post)["fired"]),
        "contradiction": c["g"] == 1 and bool(detect_contradiction(split)["fired"]),
        "word_stutter": detect_word_stutter(text)["rate"] > 0.01,
        "lenient_only": bool(v.as_scored) and not (bool(v.strict) and c["g"] == 1)
        and c["g"] == 1,
        "strict_correct": bool(v.strict) and c["g"] == 1,
    }
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dump_dir", default=None,
                    help="write N verbatim samples per flagged category here")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args(argv)

    resp_files = sorted(
        glob.glob(os.path.join(args.run_dir, "resp", "step_*.jsonl")),
        key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not resp_files:
        raise SystemExit(f"no resp/step_*.jsonl under {args.run_dir}")

    windows: dict = collections.OrderedDict()
    dumps: dict = collections.defaultdict(list)
    n_total = 0
    for path in resp_files:
        step = int(re.search(r"step_(\d+)", path).group(1))
        wkey = f"{(step - 1) // args.window * args.window + 1}-" \
               f"{(step - 1) // args.window * args.window + args.window}"
        w = windows.setdefault(wkey, collections.Counter())
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                for c in row["candidates"]:
                    n_total += 1
                    w["n"] += 1
                    flags = scan_candidate(c, row.get("ground_truth", ""))
                    for k, v in flags.items():
                        if v:
                            w[k] += 1
                            if (args.dump_dir and k not in
                                    ("strict_correct",)
                                    and len(dumps[k]) < args.samples):
                                dumps[k].append({
                                    "step": step, "uid": row["uid"],
                                    "level": row.get("level"),
                                    "gold": row.get("ground_truth", ""),
                                    "finish_reason": c["finish_reason"],
                                    "text": c["text"]})

    cats = ["empty_think", "missing_think_close", "missing_think_open",
            "cap_hit", "ngram_loop", "case_enum_20", "chk_chain_15",
            "register_leak", "answer_repeat", "contradiction", "word_stutter",
            "lenient_only", "strict_correct"]
    print(f"{'window':<10}{'n':>6}" + "".join(f"{c[:12]:>13}" for c in cats))
    for wkey, w in windows.items():
        print(f"{wkey:<10}{w['n']:>6}" + "".join(
            f"{100 * w[c] / max(1, w['n']):>12.2f}%" for c in cats))

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for k, items in dumps.items():
            with open(os.path.join(args.dump_dir, f"{k}.jsonl"), "w") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
        print(f"\n[scan] verbatim samples -> {args.dump_dir}/<category>.jsonl "
              f"({args.samples} max per category) — READ THEM; the table above "
              "is the mechanical half only")
    print(f"[scan] {n_total} rollouts scanned across {len(resp_files)} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
