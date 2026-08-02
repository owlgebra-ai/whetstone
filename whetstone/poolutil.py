"""Shared helpers for the v2 data-pool / eval-suite builders (design §12.7).

Everything that must stay byte-identical across `scripts/build_train_pool.py`,
`scripts/build_eval_sets.py` and `scripts/check_contamination.py` lives here —
above all the `_uid` recipe, which the v1 resume invariants (v1 §2.5) key on.

`_uid` recipe (v2, packet P1): ``"<source>:<sha8-of-normalized-prompt>"`` where
"normalized" means whitespace runs collapsed and the string stripped — **case is
preserved**. The v1 recipe hashed the *raw* prompt and appended the row index;
that made ids unstable under any re-shuffle of the source dataset. Since the v2
pool is built from scratch (no v1 resume file survives the source change), the
packet's stable recipe is used. Do not change it again.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from typing import Callable, Iterable, Sequence

WS = re.compile(r"\s+")
WORD = re.compile(r"\w+")


def norm_text(text: str | None) -> str:
    """Collapse whitespace runs, strip. Case preserved (packet P1 gotcha 4)."""
    return WS.sub(" ", (text or "")).strip()


def uid_for(source: str, prompt: str) -> str:
    """`<source>:<sha8 of the normalized prompt>` — stable across rebuilds."""
    h = hashlib.sha1(norm_text(prompt).encode("utf-8")).hexdigest()[:8]
    return f"{source}:{h}"


def dedup_key(prompt: str) -> str:
    """Longer hash of the normalized prompt, used for exact-duplicate removal."""
    return hashlib.sha1(norm_text(prompt).encode("utf-8")).hexdigest()[:16]


def match_key(prompt: str) -> str:
    """Contamination key: normalized prompt, case-folded (§P1 part 5)."""
    return norm_text(prompt).lower()


def ngrams(text: str, n: int = 8, first_chars: int = 400) -> set[str]:
    """Word n-gram set over the first `first_chars` chars of the match key."""
    toks = WORD.findall(match_key(text)[:first_chars])
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def stratified_sample(
    items: Sequence[dict],
    key_fn: Callable[[dict], str],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n items preserving the per-key distribution of key_fn (v1 machinery)."""
    if n >= len(items):
        return list(items)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        buckets[key_fn(it)].append(it)
    out: list[dict] = []
    total = len(items)
    quotas = {k: (n * len(b)) // total for k, b in buckets.items()}
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
        out.extend(buckets[k][: quotas[k]])
    remaining = {k: buckets[k][quotas[k] :] for k in keys}
    while len(out) < n and any(remaining.values()):
        for k in sorted(keys, key=lambda x: -len(remaining[x])):
            if not remaining[k]:
                continue
            out.append(remaining[k].pop(0))
            if len(out) >= n:
                break
    return out[:n]


def write_jsonl(path: str, rows: Iterable[dict], keep_private: bool = False) -> int:
    """Write pure-record JSONL (no header line — metadata goes to `<path>.meta.json`).

    Keys starting with `_` are dropped unless whitelisted (`_uid` always kept),
    so scratch fields like `_dedup` never reach disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            if not keep_private:
                r = {k: v for k, v in r.items() if k == "_uid" or not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_meta(path: str, meta: dict) -> str:
    """Sidecar metadata for a JSONL (pinned revisions, row counts, grading mode).

    Deliberately *not* a header line inside the JSONL: the P1 acceptance check
    requires `head -1` of every JSONL to be a schema-conforming record, and every
    existing reader (`harvest.py`, `run_eval.py`, …) json-loads every line.
    """
    meta_path = f"{os.path.splitext(path)[0]}.meta.json"
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return meta_path


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = [
    "norm_text",
    "uid_for",
    "dedup_key",
    "match_key",
    "ngrams",
    "jaccard",
    "stratified_sample",
    "write_jsonl",
    "write_meta",
    "read_jsonl",
]
