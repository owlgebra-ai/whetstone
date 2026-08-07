"""Build the WHETSTONE v2 train + val problem pool (design §12.7, packet P1).

v2 replaces v1's openr1/nemotron mix with:

  * **DeepMath-103K** (`zwhe99/DeepMath-103K`) — main pool. Verified golds plus a
    `difficulty` label in [1,10] that becomes `level`, which every later stage
    uses for stratified probes, curriculum bands and pass-rate stratification.
  * **GSM8K** (`openai/gsm8k`, config `main`, split `train`) — the easy tier,
    ~20% of the pool, all at `level: 1`.

Output schema (one JSONL line per problem, no header line):

    {
        "_uid": "<source>:<sha8-of-normalized-prompt>",
        "prompt": "<problem statement>",
        "ground_truth": "<gold answer string>",
        "level": <int>,
        "source": "deepmath" | "gsm8k",
        "difficulty": <float, DeepMath only>,
        "topic": "<DeepMath topic, when present>"
    }

`_uid / prompt / ground_truth / level` are the non-negotiable four (v1 §1);
`source / difficulty / topic` are additive and safe for readers that ignore them.
Pinned dataset revisions and row counts go to the sidecar `*.meta.json` and to
`pool_stats.json` — never into a JSONL header line (readers json-load every line).

Gold handling:
  * GSM8K: the text after ``#### ``, stripped of ``$``/commas/units, then run
    through ``whetstone.verify._normalize`` and stored in normalized form.
  * DeepMath: stored **verbatim**. The golds are LaTeX (``\\frac{3}{4}``,
    intervals, sets); reformatting them here would silently shift verifier yield
    everywhere downstream. The verifier normalizes at compare time.

Part 2 of packet P1 (``--sca_out_dir``) additionally emits the three-stage
SCA-comparison curriculum from the same deduplicated source pools.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from typing import Iterable

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import (
    dedup_key,
    norm_text,
    stratified_sample,
    uid_for,
    write_jsonl,
    write_meta,
)
from whetstone.verify import _normalize as verify_normalize

# Pinned dataset revisions (resolved 2026-08-01; see activity/002-data-pools.md).
DEEPMATH_REPO = "zwhe99/DeepMath-103K"
DEEPMATH_REV = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"

GSM8K_ANS_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
# Difficulty bands used by the SCA-arm stage 3 ("500 low + 500 high").
SCA_LOW_MAX = 4.0
SCA_HIGH_MIN = 7.0


def _level_from_difficulty(d: float) -> int:
    """DeepMath difficulty is a float in 0.5 steps → integer level band.

    Round-half-up (not Python's banker's rounding) so 4.5 → 5 deterministically.
    """
    return int(math.floor(float(d) + 0.5))


def _gsm8k_gold(answer_field: str) -> str | None:
    """`#### 1,234` → `1234`, normalized through the deterministic verifier."""
    m = GSM8K_ANS_RE.search(answer_field or "")
    if not m:
        return None
    raw = m.group(1).strip()
    raw = raw.replace("$", "").replace(",", "").strip()
    raw = raw.rstrip(".").strip()
    gold = verify_normalize(raw)
    return gold or None


def load_deepmath(revision: str, limit: int | None = None) -> Iterable[dict]:
    from datasets import load_dataset

    # Not streamed: the 10 parquet shards total ~2.1 GB and land in the shared
    # HF cache, so re-runs and the SCA arm cost nothing.
    ds = load_dataset(DEEPMATH_REPO, split="train", revision=revision)
    # Drop the three r1_solution traces before iterating — they are ~90% of the
    # bytes and Stage 1 blindness means we must not carry them into the pool.
    ds = ds.select_columns([c for c in ("question", "final_answer", "difficulty", "topic")
                            if c in ds.column_names])
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        prompt = norm_text(rec.get("question"))
        gold = (rec.get("final_answer") or "").strip()  # verbatim, no normalization
        diff = rec.get("difficulty")
        if not prompt or not gold or diff is None:
            continue
        # DeepMath ships a handful of sentinel difficulties outside [1,10]
        # (observed: a single -1.0). They would create a bogus level stratum, so
        # they are dropped rather than clamped.
        if not (1.0 <= float(diff) <= 10.0):
            continue
        yield {
            "_uid": uid_for("deepmath", prompt),
            "prompt": prompt,
            "ground_truth": gold,
            "level": _level_from_difficulty(diff),
            "source": "deepmath",
            "difficulty": float(diff),
            "topic": rec.get("topic") or "",
            "_dedup": dedup_key(prompt),
        }


def load_gsm8k(revision: str, limit: int | None = None) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset(GSM8K_REPO, "main", split="train", revision=revision)
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        prompt = norm_text(rec.get("question"))
        gold = _gsm8k_gold(rec.get("answer") or "")
        if not prompt or not gold:
            continue
        yield {
            "_uid": uid_for("gsm8k", prompt),
            "prompt": prompt,
            "ground_truth": gold,
            "level": 1,
            "source": "gsm8k",
            "_dedup": dedup_key(prompt),
        }


def _dedup(rows: Iterable[dict], seen: set[str], tag: str) -> list[dict]:
    kept, n_in, n_dup = [], 0, 0
    for r in rows:
        n_in += 1
        if r["_dedup"] in seen:
            n_dup += 1
            continue
        seen.add(r["_dedup"])
        kept.append(r)
        if n_in % 20000 == 0:
            print(f"  [{tag}] {n_in} read, {len(kept)} kept", flush=True)
    print(f"[load] {tag}: {n_in} read → {len(kept)} unique ({n_dup} dup)", flush=True)
    return kept


def _dist(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_level": {str(k): v for k, v in sorted(Counter(r["level"] for r in rows).items())},
    }


def _build_sca_arm(dm: list[dict], gsm: list[dict], out_dir: str, seed: int) -> dict:
    """Part 2: SCA-comparison curriculum — three disjoint stages, seed 0.

    stage1: 2000 GSM8K
    stage2: 1400 GSM8K + 600 DeepMath difficulty <= 4
    stage3: 1000 GSM8K + 500 DeepMath low (<=4) + 500 DeepMath high (>=7)
    """
    rng = random.Random(seed)
    gsm_pool = list(gsm)
    rng.shuffle(gsm_pool)
    low = [r for r in dm if r["difficulty"] <= SCA_LOW_MAX]
    high = [r for r in dm if r["difficulty"] >= SCA_HIGH_MIN]
    rng.shuffle(low)
    rng.shuffle(high)

    need_gsm = 2000 + 1400 + 1000
    if len(gsm_pool) < need_gsm:
        raise SystemExit(f"SCA arm needs {need_gsm} GSM8K rows, have {len(gsm_pool)}")
    if len(low) < 1100 or len(high) < 500:
        raise SystemExit(f"SCA arm needs 1100 low / 500 high DeepMath, have {len(low)}/{len(high)}")

    g1, g2, g3 = gsm_pool[:2000], gsm_pool[2000:3400], gsm_pool[3400:4400]
    l2, l3 = low[:600], low[600:1100]
    h3 = high[:500]

    stages = {
        "sca_stage1": g1,
        "sca_stage2": g2 + l2,
        "sca_stage3": g3 + l3 + h3,
    }
    summary = {}
    for name, rows in stages.items():
        rows = list(rows)
        rng.shuffle(rows)
        path = os.path.join(out_dir, f"{name}.jsonl")
        n = write_jsonl(path, rows)
        summary[name] = _dist(rows)
        summary[name]["path"] = path
        print(f"[sca] {name}: {n} → {path}", flush=True)

    # Disjointness assertion across the three stages.
    uids = [{r["_uid"] for r in v} for v in stages.values()]
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            overlap = uids[i] & uids[j]
            if overlap:
                raise SystemExit(f"SCA stages {i+1}/{j+1} overlap on {len(overlap)} uids")
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build WHETSTONE v2 train+val pool")
    ap.add_argument("--out_dir", required=True, help="e.g. /data/whetstone/data/pool")
    ap.add_argument("--sca_out_dir", default="", help="e.g. /data/whetstone/data/sca_arm")
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--gsm8k_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deepmath_rev", default=DEEPMATH_REV)
    ap.add_argument("--gsm8k_rev", default=GSM8K_REV)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows read per source")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)
    limit = args.limit or None

    seen: set[str] = set()
    print(f"[load] {GSM8K_REPO}@{args.gsm8k_rev[:8]} ...", flush=True)
    gsm = _dedup(load_gsm8k(args.gsm8k_rev, limit), seen, "gsm8k")
    print(f"[load] {DEEPMATH_REPO}@{args.deepmath_rev[:8]} ...", flush=True)
    dm = _dedup(load_deepmath(args.deepmath_rev, limit), seen, "deepmath")

    n_total = args.n_train + args.n_val
    n_gsm = min(int(round(n_total * args.gsm8k_frac)), len(gsm))
    n_dm = min(n_total - n_gsm, len(dm))
    print(f"[plan] target {n_total} = {n_gsm} gsm8k + {n_dm} deepmath", flush=True)

    gsm_s = stratified_sample(gsm, lambda r: "1", n_gsm, rng)
    dm_s = stratified_sample(dm, lambda r: str(r["level"]), n_dm, rng)
    pool = gsm_s + dm_s
    rng.shuffle(pool)

    # Val is stratified over source×level so it mirrors train exactly.
    val = stratified_sample(pool, lambda r: f"{r['source']}|{r['level']}", args.n_val, rng)
    val_uids = {r["_uid"] for r in val}
    train = [r for r in pool if r["_uid"] not in val_uids][: args.n_train]

    train_path = os.path.join(args.out_dir, "train_30k.jsonl")
    val_path = os.path.join(args.out_dir, "val_2k.jsonl")
    n_tr = write_jsonl(train_path, train)
    n_va = write_jsonl(val_path, val)
    print(f"[done] train={n_tr} → {train_path}", flush=True)
    print(f"[done] val  ={n_va} → {val_path}", flush=True)

    meta = {
        "builder": "scripts/build_train_pool.py",
        "seed": args.seed,
        "sources": {
            "deepmath": {"repo": DEEPMATH_REPO, "revision": args.deepmath_rev,
                         "split": "train", "unique_rows": len(dm)},
            "gsm8k": {"repo": GSM8K_REPO, "config": "main", "revision": args.gsm8k_rev,
                      "split": "train", "unique_rows": len(gsm)},
        },
        "gsm8k_frac": args.gsm8k_frac,
        "uid_recipe": "<source>:sha1(whitespace-collapsed prompt)[:8]",
        "gold_policy": {
            "deepmath": "verbatim (LaTeX preserved)",
            "gsm8k": "post-#### text, $/comma stripped, whetstone.verify._normalize applied",
        },
    }
    write_meta(train_path, {**meta, "file": train_path, "rows": n_tr, **_dist(train)})
    write_meta(val_path, {**meta, "file": val_path, "rows": n_va, **_dist(val)})

    stats = {
        **meta,
        "train": {"path": train_path, **_dist(train)},
        "val": {"path": val_path, **_dist(val)},
    }

    if args.sca_out_dir:
        stats["sca_arm"] = _build_sca_arm(dm, gsm, args.sca_out_dir, args.seed)

    stats_path = os.path.join(args.out_dir, "pool_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[done] stats → {stats_path}", flush=True)
    print("[dist] train:", json.dumps(_dist(train)), flush=True)
    print("[dist] val:  ", json.dumps(_dist(val)), flush=True)


if __name__ == "__main__":
    main()
