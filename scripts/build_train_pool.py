"""Build the WHETSTONE train + val problem pool from three math corpora.

Output schema (one JSONL line per problem):
    {
        "_uid": "<source>:<sha8>",
        "prompt": "<problem statement>",
        "ground_truth": "<gold answer string>",
        "source": "openr1-math" | "nemotron-sft-math" | "nemotron-math-proofs",
        "subject": "<sub-category, when available>",
        "level":   "<difficulty, when available>",
    }

This format is the input to Stage 1 blind harvest (`scripts/harvest.py`).
The original solutions are NOT preserved here — Stage 1 needs only the
problem statement and the gold answer (blindness invariant, §2).

Sampling:
  * OpenR1-Math-220k is stratified across its `source` field (each original
    MATH chapter / synthetic source contributes proportionally, so the 30k
    pool is representative across problem types, not dominated by one source).
  * Nemotron-SFT-Math-v4 and Nemotron-Math-Proofs-v2 are sampled uniformly.
  * The three sources are weighted equally by default (~1/3 each); pass
    --weights to rebalance.
  * Cross-source dedup on a normalized problem-text hash.

Train/val split is stratified by `source` so eval distribution mirrors train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from typing import Iterable

WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return WS.sub(" ", (text or "")).strip()


def _dedup_key(prompt: str) -> str:
    return hashlib.sha1(_norm(prompt).encode("utf-8")).hexdigest()[:16]


def _uid(source: str, prompt: str, idx: int) -> str:
    return f"{source}:{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]}:{idx}"


def _pick(rec: dict, *names: str) -> str | None:
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return str(rec[n])
    return None


def _normalize(rec: dict, source: str, idx: int) -> dict | None:
    prompt = _pick(rec, "problem", "question", "prompt", "input")
    gold = _pick(rec, "ground_truth", "answer", "gold_answer", "expected_answer",
                 "verification")
    if not prompt or not gold:
        return None
    subject = _pick(rec, "subject", "source_chapter", "source", "category", "topic",
                    "type", "domain")
    level = _pick(rec, "level", "difficulty", "grade")
    return {
        "_uid": _uid(source, prompt, idx),
        "prompt": _norm(prompt),
        "ground_truth": _norm(gold),
        "source": source,
        "subject": subject or "",
        "level": level or "",
        "_dedup": _dedup_key(prompt),
    }


def _load_openr1() -> Iterable[dict]:
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    for i, rec in enumerate(ds):
        out = _normalize(rec, "openr1-math", i)
        if out is not None:
            yield out


def _load_nemotron_sft() -> Iterable[dict]:
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-SFT-Math-v4", split="train")
    for i, rec in enumerate(ds):
        out = _normalize(rec, "nemotron-sft-math", i)
        if out is not None:
            yield out


def _load_nemotron_proofs() -> Iterable[dict]:
    from datasets import load_dataset
    ds = load_dataset("nvidia/Nemotron-Math-Proofs-v2", split="train")
    for i, rec in enumerate(ds):
        out = _normalize(rec, "nemotron-math-proofs", i)
        if out is not None:
            yield out


LOADERS = {
    "openr1-math": _load_openr1,
    "nemotron-sft-math": _load_nemotron_sft,
    "nemotron-math-proofs": _load_nemotron_proofs,
}


def _stratified_sample(items: list[dict], key_fn, n: int,
                       rng: random.Random) -> list[dict]:
    """Sample n items, preserving the per-key distribution of key_fn(items)."""
    if n >= len(items):
        return list(items)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    out: list[dict] = []
    total = len(items)
    # First pass: proportional allocation, floor.
    quotas: dict[str, int] = {}
    for k, bucket in buckets.items():
        quotas[k] = (n * len(bucket)) // total
    # Sort keys for determinism.
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
        out.extend(buckets[k][: quotas[k]])
    # Top up: round-robin from remaining items, largest bucket first.
    remaining = {k: buckets[k][quotas[k]:] for k in keys}
    i = 0
    while len(out) < n and any(remaining.values()):
        for k in sorted(keys, key=lambda x: -len(remaining[x])):
            if not remaining[k]:
                continue
            out.append(remaining[k].pop(0))
            if len(out) >= n:
                break
            i += 1
    return out[:n]


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build WHETSTONE train+val pool from 3 datasets")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--weights", default="openr1-math:1,nemotron-sft-math:1,nemotron-math-proofs:1",
                    help="Comma list source:weight — controls per-source allocation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=",".join(LOADERS.keys()),
                    help="Comma list of sources to include")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)

    weights = {k: float(v) for k, v in (s.split(":") for s in args.weights.split(","))}
    active_sources = [s for s in args.sources.split(",") if s in LOADERS]
    if not active_sources:
        raise SystemExit(f"no valid sources in {args.sources!r}")
    total_w = sum(weights.get(s, 0.0) for s in active_sources)
    if total_w <= 0:
        raise SystemExit(f"all weights zero for {active_sources}")

    # Pull each source, dedup within-source.
    by_source: dict[str, list[dict]] = {}
    seen_global: set[str] = set()
    for src in active_sources:
        print(f"[load] streaming {src}...", flush=True)
        kept: list[dict] = []
        seen_local: set[str] = set()
        n_in = n_dup = 0
        for rec in LOADERS[src]():
            n_in += 1
            if rec["_dedup"] in seen_local:
                n_dup += 1
                continue
            if rec["_dedup"] in seen_global:
                n_dup += 1
                continue
            seen_local.add(rec["_dedup"])
            seen_global.add(rec["_dedup"])
            kept.append(rec)
            if n_in % 5000 == 0:
                print(f"  [{src}] {n_in} read, {len(kept)} kept", flush=True)
        print(f"[load] {src}: {n_in} read → {len(kept)} unique ({n_dup} dup)", flush=True)
        by_source[src] = kept

    total_pool = sum(len(v) for v in by_source.values())
    n_total = args.n_train + args.n_val
    if total_pool < n_total:
        print(f"[WARN] only {total_pool} problems available; targeting {n_total}",
              flush=True)

    # Per-source target count proportional to weight, clamped to availability.
    targets: dict[str, int] = {}
    for s in active_sources:
        alloc = int(round(n_total * weights.get(s, 0.0) / total_w))
        targets[s] = min(alloc, len(by_source[s]))
    # Top up from the largest undersampled source if rounding left seats empty.
    while sum(targets.values()) < min(n_total, total_pool):
        room = [(s, len(by_source[s]) - targets[s]) for s in active_sources]
        room.sort(key=lambda x: -x[1])
        for s, r in room:
            if r > 0:
                targets[s] += 1
                break

    print("[plan] per-source sample targets:", dict(targets), flush=True)

    # Stratify within each source by `subject` (problem type), to preserve
    # representation. Fall back to the whole bucket when subject missing.
    sampled: list[dict] = []
    for s in active_sources:
        sub = _stratified_sample(
            by_source[s],
            key_fn=lambda r: r.get("subject") or "_",
            n=targets[s],
            rng=rng,
        )
        sampled.extend(sub)
        print(f"[sample] {s}: {len(sub)} (from {len(by_source[s])})", flush=True)

    rng.shuffle(sampled)

    # Final train/val split, stratified by source.
    val = _stratified_sample(sampled, key_fn=lambda r: r["source"],
                             n=min(args.n_val, len(sampled) // 10), rng=rng)
    val_keys = {r["_uid"] for r in val}
    train = [r for r in sampled if r["_uid"] not in val_keys]
    if len(train) > args.n_train:
        train = train[: args.n_train]

    train_path = os.path.join(args.out_dir, "train_pool.jsonl")
    val_path = os.path.join(args.out_dir, "val_pool.jsonl")
    n_tr = _write_jsonl(train_path, train)
    n_va = _write_jsonl(val_path, val)
    print(f"[done] train={n_tr} -> {train_path}", flush=True)
    print(f"[done] val  ={n_va} -> {val_path}", flush=True)

    # Per-source distribution report
    from collections import Counter
    tr_counts = Counter(r["source"] for r in train)
    va_counts = Counter(r["source"] for r in val)
    print("[dist] train:", dict(tr_counts), flush=True)
    print("[dist] val:  ", dict(va_counts), flush=True)


if __name__ == "__main__":
    main()
