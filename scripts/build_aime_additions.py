"""AIME 1983–2023 pool additions (activity 011, phase-2 amendment).

User direction 2026-08-07: add `gneubig/aime-1983-2024` to the phase-2 RL
pool, **excluding 2024** — the `aime24` eval suite is AIME 2024, and `aime25`
is out of the dataset's range. Defense in depth, same as the other additions:

1. hard **year filter** (< 2024, from the Year column or the ID),
2. exact normalized-text gate vs every eval suite AND the already-built pool,
3. the P1 8-gram near-duplicate gate (`check_contamination.py --apply`),
   which the caller runs after this script.

AIME answers are integers 0–999 — no normalizer risk. Rows: source
``aime_hist``, level 8 (reporting-only; DeepMath's L8 band is the nearest
difficulty analogue — the curriculum tilts on measured p̂, not labels).

Usage (turing, CPU)::

    python scripts/build_aime_additions.py \\
        --existing /data/whetstone/data/pool/train_30k.jsonl \\
                   /data/whetstone/data/pool/phase2_additions.jsonl \\
        --eval_dir /data/whetstone/eval \\
        --out /data/whetstone/data/pool/aime_additions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existing", nargs="+", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_year", type=int, default=2023,
                    help="keep problems from years <= this (2024 excluded: "
                         "aime24 eval)")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    known = set()
    for path in args.existing:
        with open(path) as f:
            for line in f:
                known.add(_norm_text(json.loads(line)["prompt"]))
    eval_norm = set()
    for fn in os.listdir(args.eval_dir):
        if fn.endswith(".jsonl"):
            with open(os.path.join(args.eval_dir, fn)) as f:
                for line in f:
                    row = json.loads(line)
                    text = (row.get("prompt") or row.get("problem")
                            or row.get("question") or "")
                    eval_norm.add(_norm_text(text))

    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    cols = set(ds.column_names)
    print(f"[aime] columns: {sorted(cols)}")

    def _year(r) -> int:
        for k in ("Year", "year"):
            if k in cols and r.get(k) is not None:
                return int(r[k])
        rid = str(r.get("ID") or r.get("id") or "")
        m = re.match(r"(\d{4})", rid)
        return int(m.group(1)) if m else 0

    def _field(r, *names):
        for n in names:
            if n in cols and r.get(n):
                return str(r[n])
        return ""

    rows, stats = [], {"year_excluded": 0, "eval_exact": 0, "dup": 0,
                       "no_answer": 0, "seen": 0}
    seen = set()
    for r in ds:
        y = _year(r)
        if y == 0 or y > args.max_year:
            stats["year_excluded"] += 1
            continue
        q = _field(r, "Question", "question", "Problem", "problem")
        a = _field(r, "Answer", "answer").strip()
        n = _norm_text(q)
        if not q or not a:
            stats["no_answer"] += 1
            continue
        if n in eval_norm:
            stats["eval_exact"] += 1
            continue
        if n in known:
            stats["dup"] += 1
            continue
        if n in seen:
            stats["seen"] += 1
            continue
        seen.add(n)
        rows.append({"_uid": f"aimeh:{hashlib.sha1(n.encode()).hexdigest()[:8]}",
                     "prompt": q.strip(), "ground_truth": a,
                     "level": 8, "source": "aime_hist", "year": y})

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    years = sorted({r["year"] for r in rows})
    print(f"[aime] wrote {len(rows)} rows ({years[0]}–{years[-1]}) -> {args.out}")
    print(f"[aime] filtered: {stats}")
    print("[aime] NOW RUN scripts/check_contamination.py --apply on the output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
