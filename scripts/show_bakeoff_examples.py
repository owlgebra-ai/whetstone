"""Print bake-off records for hand inspection (packet P3a, chunk-alignment check + M5).

Two modes:
  --mode chunks   ORIGINAL CHUNK k vs COMPACT CHUNK k, per trace — the
                  chunk-alignment check that must pass before any bulk run.
  --mode faithful verbose think vs compact think, side by side — M5's
                  fused-steps / dropped-values / hallucinated-shortcut review.

Usage::

    python scripts/show_bakeoff_examples.py --files A=/…/bakeoff_A.jsonl \\
        B=/…/bakeoff_B.jsonl --mode faithful --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="LABEL=path …")
    ap.add_argument("--mode", choices=["chunks", "faithful"], default="faithful")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--uids", nargs="*", default=None)
    ap.add_argument("--max_chars", type=int, default=2500)
    args = ap.parse_args()

    corpora = {}
    for spec in args.files:
        label, path = spec.split("=", 1)
        corpora[label] = {r["_uid"]: r for r in read_jsonl(path)}

    labels = list(corpora)
    uids = args.uids or sorted(set.intersection(*(set(c) for c in corpora.values())))[: args.n]

    for uid in uids:
        print("#" * 78)
        for label in labels:
            r = corpora[label].get(uid)
            if r is None:
                continue
            print(f"### ARM {label} — {uid}  level {r.get('level')}  "
                  f"verbose {r['verbose_think_tokens']} -> compact "
                  f"{r['compact_think_tokens']} tok  "
                  f"(ratio {r['compression_ratio']:.3f})  verify={r['verify_ok']}")
            if args.mode == "chunks":
                for i, (c, k) in enumerate(zip(r["chunks"], r["compacts_per_chunk"])):
                    print(f"--- ORIGINAL {i + 1} ---\n{c[:args.max_chars]}")
                    print(f"--- COMPACT  {i + 1} ---\n{k[:args.max_chars]}")
            else:
                if label == labels[0]:
                    print(f"--- VERBOSE THINK ---\n{r['verbose_think'][:args.max_chars]}")
                print(f"--- COMPACT THINK ({label}) ---\n{r['compact_think'][:args.max_chars]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
