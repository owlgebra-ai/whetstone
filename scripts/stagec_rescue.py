"""Pedagogy rescue for 0/K problems (design §5.2; packet P7 §10).

The problems DAPO cannot learn from. A group with zero correct rollouts has no
within-group advantage, so dynamic sampling drops it and the curriculum never
batches it — those problems would sit untouched for the whole phase. Rescue is
the path back in: the **privileged** 32B teacher, holding the gold answer (and
the verbose trace where one exists), writes candidate solutions; survivors of a
hard filter are assimilated with the Stage-B loss; the problems then re-enter
the next K=8 refresh and can rejoin the curriculum on their own merits.

Runs **at phase boundaries, not continuously** (packet §10).

Two deviations from design §5.2, both attested and both from measurement:

* **The filter is not `G_spike`-thresholded.** Activity 008 finding 10b measured
  G_spike's faithful-vs-wrong AUC falling 0.800 (L1) → 0.633 (L6) → 0.555 (L8)
  → **0.541 (L9)** — a coin flip exactly in the hard band, which is where every
  0/K problem lives. A G_spike threshold here would apply real pressure on easy
  problems and noise on hard ones, the reverse of what is wanted. Replaced by:
  **strict verify + g=1 + in-register + GLM faithfulness**.
* **Conditioning is `gold+trace` wherever a trace exists.** Given the answer but
  not the reasoning on a hard problem, the teacher invents a derivation —
  activity 008 finding 13 measured gold-only at **73.7% wrong** in the hard band
  against `gold+trace`'s 17.7%. Problems with no trace are still generated but
  are flagged, and their yield is reported separately.

This script is a **driver**: it selects the clientele, delegates generation to
``teacher_generate.py`` and assimilation to ``stageb_train.py`` (both already
built and validated in activities 008/009), and owns the filtering and the
bookkeeping in between. It prints the two delegated commands rather than
guessing at their environment — the 32B server placement and the GLM quota are
operator decisions.

Usage::

    # 1. select the clientele and write the teacher's input subset
    python scripts/stagec_rescue.py select \\
        --buckets /data/whetstone/runs/stagec_buckets/phase1_init/buckets.jsonl \\
        --stagea_subset /data/whetstone/corpora/stagea/subset_stagea.jsonl \\
        --out_dir /data/whetstone/corpora/rescue/phase1

    # 2. (delegated) run teacher_generate.py with the printed command, M=4

    # 3. filter the drafts and emit the assimilation corpus
    python scripts/stagec_rescue.py filter \\
        --drafts /data/whetstone/corpora/rescue/phase1/drafts.jsonl \\
        --out_dir /data/whetstone/corpora/rescue/phase1
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_jsonl
from whetstone.reward.strict import verify_strict

#: In-register test, same shape as F3d's and 009 finding 1's: line-initial
#: markers plus two symbols, never bare substrings (`case` is an English word
#: in 10.2% of honest answers).
_MARKER_RE = re.compile(r"(?m)^\s*(?:goal|chk|sub|let|case)\s*:|[⇒✗✓]")
MIN_MARKERS = 3


def register_markers(think: str) -> int:
    return len(_MARKER_RE.findall(think or ""))


def cmd_select(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_jsonl(args.buckets)
    zero = [r for r in rows if r["bucket"] == "0/K"]
    by_level = collections.Counter(r["level"] for r in zero)
    by_seen = collections.Counter(bool(r.get("seen")) for r in zero)

    # Carry the teacher's conditioning fields across from Stage A's subset so
    # `gold+trace` problems keep their traces (008 finding 13).
    stagea = {r["_uid"]: r for r in read_jsonl(args.stagea_subset)}
    out, n_trace = [], 0
    for r in zero:
        src = stagea.get(r["_uid"])
        if src is None:
            continue
        out.append(src)
        n_trace += bool(src.get("has_trace"))

    path = os.path.join(args.out_dir, "rescue_subset.jsonl")
    write_jsonl(path, out)
    summary = {
        "n_zero_bucket": len(zero),
        "n_written": len(out),
        "n_missing_from_stagea_subset": len(zero) - len(out),
        "n_with_trace": n_trace,
        "frac_with_trace": n_trace / max(1, len(out)),
        "by_level": {str(k): v for k, v in sorted(by_level.items())},
        "by_seen": {str(k): v for k, v in by_seen.items()},
    }
    with open(os.path.join(args.out_dir, "select_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[rescue] wrote {len(out)} problems -> {path}")
    print(f"[rescue] {n_trace} have a verbose trace ({100*n_trace/max(1,len(out)):.1f}%); "
          "the rest are gold-only and 008 f13 says expect ~74% confabulation there — "
          "their yield is reported separately by `filter`.\n")
    print("[rescue] next (delegated to P5's generator, M=4 at its pinned regime):\n")
    print(f"""  python scripts/teacher_generate.py \\
      --subset {path} \\
      --output {os.path.join(args.out_dir, 'drafts.jsonl')} \\
      --server http://127.0.0.1:8000/v1 --model nvidia/Qwen3-32B-NVFP4 \\
      --k 4 --temperature 0.8 --top_p 0.95 --concurrency 16
""")
    return 0


def cmd_filter(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    drafts = read_jsonl(args.drafts)
    counts = collections.Counter()
    best: dict = {}

    for d in drafts:
        counts["drafts"] += 1
        uid = d["_uid"]
        if int(d.get("g", 0)) != 1:
            counts["reject_malformed"] += 1
            continue
        v = verify_strict(d.get("raw_text", ""), d.get("ground_truth", ""))
        if not v.strict:
            counts["reject_wrong_strict"] += 1
            continue
        n_mark = register_markers(d.get("compact_think", ""))
        if n_mark < MIN_MARKERS:
            counts["reject_out_of_register"] += 1
            continue
        counts["survivors"] += 1
        # One trace per problem; prefer the most register-dense survivor.
        if uid not in best or n_mark > best[uid]["_markers"]:
            rec = dict(d)
            rec["_markers"] = n_mark
            best[uid] = rec

    rows = []
    for uid, d in best.items():
        rows.append({
            "_uid": uid, "prompt": d["prompt"], "ground_truth": d["ground_truth"],
            "level": d.get("level"), "source": d.get("source"),
            "compact_think": d["compact_think"], "answer": d.get("answer", ""),
            "completion": d.get("raw_text", ""),
            "conditioned_on": d.get("conditioned_on"),
            "register_markers": d["_markers"],
            "think_tokens": d.get("think_tokens"),
            "answer_tokens": d.get("answer_tokens"),
        })
    path = os.path.join(args.out_dir, "rescue_corpus.jsonl")
    write_jsonl(path, rows)

    by_cond = collections.Counter(r["conditioned_on"] for r in rows)
    n_problems = len({d["_uid"] for d in drafts})
    summary = {
        "counts": dict(counts),
        "n_problems_attempted": n_problems,
        "n_problems_rescued": len(rows),
        "problem_yield": len(rows) / max(1, n_problems),
        "draft_yield": counts["survivors"] / max(1, counts["drafts"]),
        "by_conditioning": {str(k): v for k, v in by_cond.items()},
        "by_level": {str(k): v for k, v in
                     sorted(collections.Counter(r["level"] for r in rows).items())},
        "note": ("G_spike is deliberately NOT a filter here — 008 f10b measured "
                 "its faithful-vs-wrong AUC at 0.541 at level 9, a coin flip in "
                 "exactly the band every 0/K problem lives in."),
    }
    with open(os.path.join(args.out_dir, "filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[rescue] {len(rows)} problems rescued -> {path}")
    print("[rescue] STILL REQUIRED before assimilation: the GLM faithfulness pass")
    print("         (packet §10 filter is strict-verify AND g=1 AND in-register AND")
    print("          GLM-faithful). Budget ~100 judgments per rescue round:\n")
    print(f"""  python scripts/faithfulness_audit.py \\
      --corpus {path} \\
      --output {os.path.join(args.out_dir, 'faithfulness.jsonl')} \\
      --n 100 --concurrency 8
""")
    print("[rescue] then assimilate with the Stage-B loss — whitelist floor ON,")
    print("         LR 5e-6, <=1 epoch, FRESH EMA (never carried over, design §12.4):\n")
    print("""  python scripts/stageb_train.py --corpus <faithful subset> \\
      --lr 5e-6 --epochs 1 --whitelist_floor 1.0   # confirm flag names against
                                                   # the script's own --help
""")
    print("[rescue] finally: re-bucket the rescued uids under the updated policy and")
    print("         report how many actually moved out of 0/K. A rescue that does not")
    print("         move buckets is a corpus, not a rescue (packet §10).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="pick the 0/K clientele")
    s.add_argument("--buckets", required=True)
    s.add_argument("--stagea_subset",
                   default="/data/whetstone/corpora/stagea/subset_stagea.jsonl")
    s.add_argument("--out_dir", required=True)
    s.set_defaults(fn=cmd_select)

    f = sub.add_parser("filter", help="filter teacher drafts into a rescue corpus")
    f.add_argument("--drafts", required=True)
    f.add_argument("--out_dir", required=True)
    f.set_defaults(fn=cmd_filter)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
