"""Phase-2 pool additions: hendrycks MATH (train) + AIMO AMC (activity 011).

User direction 2026-08-07: continue RL past the global-400 endpoint (target
~1000) and widen the pool with `EleutherAI/hendrycks_math` and
`AI-MO/aimo-validation-amc` **in addition to** the existing 4,000-problem set.

Contamination is the reason this script exists rather than a one-liner:

* ``aimo-validation-amc`` mixes AMC 2022 **and 2023** — and the project's
  ``amc23`` eval suite *is* AMC 2023. Every row whose text or answer overlaps
  any eval suite is dropped here, and the survivor list is re-screened by
  ``scripts/check_contamination.py`` (P1's 8-gram/first-chars checker) as a
  second, independent gate. Training on an eval twin would quietly convert
  the benchmark into a memorization read.
* ``hendrycks_math``'s **test** split is MATH-500's source. Only the
  **train** split is taken, and it still goes through the same two gates
  (MATH-500 ∈ the eval dir).
* DeepMath-103K partially derives from MATH — new rows whose normalized text
  collides with the existing ``train_30k.jsonl`` are dropped as duplicates
  (double-weighting, not contamination, but still unwanted).

Output rows use the pool schema exactly (`_uid / prompt / ground_truth /
level / source`). MATH keeps its native 1–5 level under source
``hendrycks_math``; AMC rows get level 6 under source ``aimo_amc`` (AMC12
difficulty sits near DeepMath's mid-high band; the curriculum is p̂-driven so
the label is reporting-only). Draw policy: **all** of MATH L4–L5 and AMC
(hard problems are where mixed groups live at a 75% Pass@1 policy), plus
``--n_l3`` from L3 and ``--n_easy`` from L1–L2, seeded.

Usage (turing, CPU)::

    python scripts/build_phase2_additions.py \\
        --existing /data/whetstone/data/pool/train_30k.jsonl \\
        --eval_dir /data/whetstone/eval \\
        --out /data/whetstone/data/pool/phase2_additions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _uid(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def _last_boxed(solution: str):
    matches = BOXED_RE.findall(solution or "")
    return matches[-1].strip() if matches else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existing", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_l3", type=int, default=600)
    ap.add_argument("--n_easy", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    rng = random.Random(args.seed)

    existing_norm = set()
    with open(args.existing) as f:
        for line in f:
            existing_norm.add(_norm_text(json.loads(line)["prompt"]))

    # Eval-suite texts for the first (exact/normalized) contamination gate.
    eval_norm = set()
    for fn in os.listdir(args.eval_dir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(args.eval_dir, fn)) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("prompt") or row.get("problem") or row.get("question") or ""
                eval_norm.add(_norm_text(text))

    rows, stats = [], {"dup_existing": 0, "dup_internal": 0, "eval_exact": 0,
                       "no_answer": 0}
    seen_norm = set()

    def _add(prompt: str, gold: str, level: int, source: str, prefix: str) -> None:
        n = _norm_text(prompt)
        if not gold:
            stats["no_answer"] += 1
            return
        if n in eval_norm:
            stats["eval_exact"] += 1
            return
        if n in existing_norm:
            stats["dup_existing"] += 1
            return
        if n in seen_norm:
            stats["dup_internal"] += 1
            return
        seen_norm.add(n)
        rows.append({"_uid": _uid(prefix, n), "prompt": prompt.strip(),
                     "ground_truth": gold, "level": level, "source": source})

    # --- hendrycks MATH, train split only ------------------------------------
    by_level: dict = {1: [], 2: [], 3: [], 4: [], 5: []}
    ds = load_dataset("EleutherAI/hendrycks_math", "all", split="train")
    for r in ds:
        m = re.search(r"(\d)", r.get("level") or "")
        lv = int(m.group(1)) if m else 0
        if lv in by_level:
            by_level[lv].append(r)
    for lv in (4, 5):
        for r in by_level[lv]:
            _add(r["problem"], _last_boxed(r["solution"]), lv, "hendrycks_math", "math")
    l3 = by_level[3][:]
    rng.shuffle(l3)
    for r in l3[:args.n_l3]:
        _add(r["problem"], _last_boxed(r["solution"]), 3, "hendrycks_math", "math")
    easy = by_level[1] + by_level[2]
    rng.shuffle(easy)
    for r in easy[:args.n_easy]:
        m = re.search(r"(\d)", r.get("level") or "")
        _add(r["problem"], _last_boxed(r["solution"]),
             int(m.group(1)) if m else 1, "hendrycks_math", "math")

    # --- AIMO validation AMC --------------------------------------------------
    amc = load_dataset("AI-MO/aimo-validation-amc", split="train")
    n_amc = 0
    for r in amc:
        gold = str(r.get("answer", "")).strip()
        before = len(rows)
        _add(r["problem"], gold, 6, "aimo_amc", "amc")
        n_amc += len(rows) - before

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    comp = Counter((r["source"], r["level"]) for r in rows)
    print(f"[phase2] wrote {len(rows)} rows -> {args.out}")
    print(f"[phase2] AMC survivors of the exact gate: {n_amc}")
    for (src, lv), n in sorted(comp.items()):
        print(f"  {src:>15} L{lv}: {n}")
    print(f"[phase2] filtered: {stats}")
    print("[phase2] NOW RUN the 8-gram gate: scripts/check_contamination.py "
          f"--train {args.out} --eval_dir {args.eval_dir} --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
