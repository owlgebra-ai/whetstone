"""Apply the Δlogp gate and emit the seed register corpus (packet P3 Part 2, "Output").

Takes the Δlogp-annotated compression output and writes the accepted corpus with
the packet's field list, after asserting the invariants that make it safe to
train on. Every check here is a **hard failure**, not a warning — each one
corresponds to a documented way this corpus can silently poison a later stage:

  * ``verify_ok`` — the answer segment was copied through untouched, so it must
    still verify exactly as its source trace did. A failure means the compressor
    leaked past the think boundary (P3 gotcha 3);
  * ``think_has_boxed`` — a boxed answer inside the think segment is a card §1.5
    violation and exactly the contamination Stage C's answer-segment KL exists
    to prevent (activity 005 finding 5);
  * **both think versions present** — Stage-A teacher conditioning wants
    (gold + *verbose* trace) while Round 0 wants the *compact* one. Storing
    compact-only is unrecoverable (P3 gotcha 2).

``chunks`` is dropped: in one-shot mode it is a verbatim copy of
``verbose_think``, so keeping it doubles the corpus on disk for nothing.

Usage::

    python scripts/finalize_seed_register.py \\
        --scored  /data/whetstone/corpora/seed_register/seed_register_scored.jsonl \\
        --output  /data/whetstone/corpora/seed_register/seed_register.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_jsonl, write_meta

DROP_FIELDS = ("chunks", "compacts_per_chunk", "n_chunks")
REQUIRED = ("_uid", "prompt", "verbose_think", "compact_think", "answer",
            "delta_logp", "level")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="keep iff delta_logp > threshold (v1 §3.6)")
    args = ap.parse_args()

    rows = read_jsonl(args.scored)
    print(f"[in] {len(rows)} scored records")

    hard: list[str] = []
    n_bad_verify = sum(1 for r in rows if not r.get("verify_ok"))
    n_boxed = sum(1 for r in rows if r.get("think_has_boxed"))
    if n_bad_verify:
        hard.append(f"{n_bad_verify} records fail verify_response — the "
                    "compressor leaked past the think boundary (P3 gotcha 3)")
    if n_boxed:
        hard.append(f"{n_boxed} compact think segments contain \\boxed{{}} — "
                    "card §1.5 violation (activity 005 finding 5)")

    kept = []
    for r in rows:
        d = r.get("delta_logp")
        if not isinstance(d, (int, float)) or d != d or d <= args.threshold:
            continue
        if not (r.get("verbose_think") and r.get("compact_think")):
            hard.append(f"{r.get('_uid')}: missing a think version — Stage A "
                        "needs verbose, Round 0 needs compact (P3 gotcha 2)")
            continue
        kept.append({k: v for k, v in r.items() if k not in DROP_FIELDS})

    missing = {f for r in kept for f in REQUIRED if f not in r}
    if missing:
        hard.append(f"required fields absent from output: {sorted(missing)}")

    if hard:
        print("[finalize] *** REFUSING TO WRITE:", file=sys.stderr)
        for h in hard[:10]:
            print(f"  - {h}", file=sys.stderr)
        raise SystemExit(1)

    write_jsonl(args.output, kept)
    lv = Counter(str(r.get("level")) for r in kept)
    lv_all = Counter(str(r.get("level")) for r in rows)
    ratios = sorted(r["compression_ratio"] for r in kept if "compression_ratio" in r)
    ct = sorted(r["compact_think_tokens"] for r in kept if "compact_think_tokens" in r)
    meta = {
        "builder": "scripts/finalize_seed_register.py",
        "packet": "P3 Part 2",
        "scored_input": args.scored,
        "threshold": args.threshold,
        "n_scored": len(rows),
        "n_accepted": len(kept),
        "acceptance_rate": round(len(kept) / max(1, len(rows)), 4),
        "by_level_scored": {k: lv_all[k] for k in sorted(lv_all)},
        "by_level_accepted": {k: lv[k] for k in sorted(lv)},
        "by_level_acceptance": {k: round(lv[k] / lv_all[k], 3)
                                for k in sorted(lv_all) if lv_all[k]},
        "compression_ratio_median": ratios[len(ratios) // 2] if ratios else None,
        "compact_think_tokens_median": ct[len(ct) // 2] if ct else None,
        "card_git_sha": kept[0].get("card_git_sha") if kept else None,
        "rendered_prompt_sha1": kept[0].get("rendered_prompt_sha1") if kept else None,
    }
    write_meta(args.output, meta)

    print(f"[out] {len(kept)}/{len(rows)} accepted "
          f"({meta['acceptance_rate']:.1%}) -> {args.output}")
    print("      per-level acceptance: " + ", ".join(
        f"{k}:{lv[k]}/{lv_all[k]}" for k in sorted(lv_all)))
    if ct:
        print(f"      compact think tokens median {meta['compact_think_tokens_median']}, "
              f"compression ratio median {meta['compression_ratio_median']:.4f}")
    if not 300 <= len(kept) <= 1000:
        print(f"[finalize] NOTE: {len(kept)} accepted is outside the packet's "
              "300-1,000 target band.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
