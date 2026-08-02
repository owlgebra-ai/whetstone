"""Select the P3 seed-harvest subset (packet P3 Part 1; design §1 precondition 3).

v1 harvested the whole pool for GPU-days. v2's harvest is a **seed**: the
training corpus comes from the Stage-A teacher, not from here (design §10), so
this script takes a fixed 15% slice of the train pool and freezes it.

Selection rule (packet P3 Part 1):
  * ``--frac`` of ``train_30k.jsonl`` (default 0.15 ≈ 4,500 problems);
  * **proportional** level stratification via
    :func:`whetstone.poolutil.stratified_sample` — equal-count strata are
    impossible here, levels 2/3/10 hold 38/767/13 rows against level 6's 7,488
    (activity 002 note 1);
  * fixed seed.

**Resume invariance:** the subset is defined *once, by file*. ``subset_uids.json``
is written before anything reads it and is never re-sampled — a re-run of this
script with the same ``--frac``/``--seed`` reproduces it, but the harvest reads
``subset.jsonl``, so even a change to the sampler cannot silently move the
harvest's problem set underneath a partially-finished run (v1 §2.5).

Usage (turing or spark — pure CPU)::

    python scripts/select_seed_subset.py \\
        --pool    /data/whetstone/data/pool/train_30k.jsonl \\
        --out_dir /data/whetstone/corpora/seed --frac 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, stratified_sample, write_jsonl, write_meta

KEEP = ("_uid", "prompt", "ground_truth", "level", "source")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl")
    ap.add_argument("--out_dir", default="/data/whetstone/corpora/seed")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.pool)
    n = int(round(args.frac * len(rows)))
    print(f"[in] {len(rows)} pool rows -> target subset {n} ({args.frac:.0%})")

    sample = stratified_sample(rows, lambda r: str(r.get("level", "_")),
                               n, random.Random(args.seed))
    sample.sort(key=lambda r: r["_uid"])          # deterministic file order
    sample = [{k: r[k] for k in KEEP if k in r} for r in sample]

    os.makedirs(args.out_dir, exist_ok=True)
    uids_path = os.path.join(args.out_dir, "subset_uids.json")
    subset_path = os.path.join(args.out_dir, "subset.jsonl")

    # uid list first: it is the contract the harvest resumes against.
    with open(uids_path, "w") as f:
        json.dump([r["_uid"] for r in sample], f, indent=1)
    write_jsonl(subset_path, sample)

    lv_all = Counter(str(r.get("level")) for r in rows)
    lv_sel = Counter(str(r.get("level")) for r in sample)
    src_sel = Counter(str(r.get("source")) for r in sample)
    write_meta(subset_path, {
        "builder": "scripts/select_seed_subset.py",
        "packet": "P3 Part 1",
        "pool": args.pool,
        "pool_n": len(rows),
        "frac": args.frac,
        "seed": args.seed,
        "n": len(sample),
        "by_level": {k: lv_sel[k] for k in sorted(lv_sel, key=lambda x: int(x))},
        "by_source": dict(src_sel),
        "stratification": "proportional (poolutil.stratified_sample)",
    })

    print(f"[out] {len(sample)} problems -> {subset_path}")
    print(f"      {uids_path}")
    print("      level  pool -> selected: " + ", ".join(
        f"{k}:{lv_all[k]}->{lv_sel[k]}" for k in sorted(lv_all, key=lambda x: int(x))))
    print(f"      source: {dict(src_sel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
