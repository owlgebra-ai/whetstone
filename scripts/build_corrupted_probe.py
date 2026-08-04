"""Build the corrupted/clean twin pairs for meter test (c) — packet P4 §8.

Test (c) is **decisive**: a scorer that fails it is invalid regardless of how
well it passes (a) and (b). It asks whether the inoculated scorer still spikes
on a genuine unsupported leap after the register's style tax has been calibrated
away. So the corruptions have to be leaps of *reasoning*, and every surface
artifact that could spike for a cheaper reason has to be engineered out.

Two corruption types over ``probe_pool`` (120 traces, never trained on):

1. **Chunk deletion** — remove one intermediate numbered derivation step, so the
   next line uses a result that now comes from nowhere.
2. **Value substitution** — replace one intermediate numeric result with a
   plausible wrong value (one digit, +/-1), leaving the later steps that used
   the correct value as non-sequiturs.

Two confounds are removed deliberately, both of the kind activity 005's method
note warns about (five inverted findings from naive string handling):

* **Renumbering after deletion.** Deleting step 3 and leaving ``1. 2. 4. 5.``
  makes the *numbering* discontinuous, and the model spikes on the missing
  ``3.`` — a surface artifact, not an unsupported leap. Steps are renumbered so
  the only anomaly left is the absent derivation.
* **Step numbers are never the substituted value.** The leading ``N.`` is
  stripped before numerals are collected, so a substitution can never land on a
  step index. Counting step numbers as content is the single confound that has
  produced the most false findings on this corpus.

The output is a **fixed artifact on disk**: the same twins are scored at every
checkpoint, so (c) is comparable across the winner and its neighbours.

Usage:
    python scripts/build_corrupted_probe.py \
        --probe /data/whetstone/corpora/seed_register_qwen/probe_pool.jsonl \
        --out /data/whetstone/runs/round0/corrupted_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0 import load_jsonl  # noqa: E402

STEP_RE = re.compile(r"^(\s*)(\d+)\.(\s*)(.*)$")
# Numerals in a step *body*: integers and decimals, not step indices.
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def split_steps(body: str) -> Tuple[List[str], List[int]]:
    """``(lines, indices_of_numbered_lines)``."""
    lines = body.split("\n")
    idx = [i for i, ln in enumerate(lines) if STEP_RE.match(ln)]
    return lines, idx


def corrupt_delete(body: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    """Delete one intermediate numbered step and renumber the rest."""
    lines, idx = split_steps(body)
    if len(idx) < 3:
        return None
    # Never the first (nothing depends on it yet) or the last (that is usually
    # the conclusion, and deleting it removes the answer rather than its support).
    victim = rng.choice(idx[1:-1])
    removed = lines[victim]
    out = lines[:victim] + lines[victim + 1 :]

    n = 0
    for i, ln in enumerate(out):
        m = STEP_RE.match(ln)
        if m:
            n += 1
            out[i] = f"{m.group(1)}{n}.{m.group(3)}{m.group(4)}"
    return "\n".join(out), removed.strip()


def corrupt_substitute(body: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    """Perturb one digit of one numeric result in an intermediate step."""
    lines, idx = split_steps(body)
    if len(idx) < 2:
        return None
    for cand in rng.sample(idx[:-1], k=len(idx[:-1])):
        m = STEP_RE.match(lines[cand])
        head, body_txt = lines[cand][: m.start(4)], m.group(4)
        nums = [mm for mm in NUM_RE.finditer(body_txt)]
        # Prefer a value the later steps actually consume: corrupting it makes
        # them non-sequiturs, which is the leap the probe is meant to create.
        later = "\n".join(lines[cand + 1 :])
        used = [mm for mm in nums if mm.group(1) in later] or nums
        if not used:
            continue
        mm = rng.choice(used)
        old = mm.group(1)
        digits = [i for i, c in enumerate(old) if c.isdigit()]
        if not digits:
            continue
        p = rng.choice(digits)
        d = int(old[p])
        nd = d + rng.choice([-1, 1])
        if nd < 0:
            nd = 1
        if nd > 9:
            nd = 8
        new = old[:p] + str(nd) + old[p + 1 :]
        if new == old:
            continue
        nb = body_txt[: mm.start(1)] + new + body_txt[mm.end(1) :]
        out = list(lines)
        out[cand] = head + nb
        return "\n".join(out), f"{old} -> {new}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = load_jsonl(args.probe)
    print(f"[load] {len(recs)} probe traces from {args.probe}")

    rows, counts, skipped = [], {"delete": 0, "substitute": 0}, 0
    for i, r in enumerate(recs):
        body = r["compact_think"]
        # Alternate so the two types are drawn from the same trace distribution;
        # fall back to the other type when a trace is not eligible.
        order = ["delete", "substitute"] if i % 2 == 0 else ["substitute", "delete"]
        made = None
        for kind in order:
            fn = corrupt_delete if kind == "delete" else corrupt_substitute
            got = fn(body, rng)
            if got is not None:
                made = (kind, got[0], got[1])
                break
        if made is None:
            skipped += 1
            continue
        kind, corrupted, detail = made
        counts[kind] += 1
        rows.append({
            "_uid": r["_uid"],
            "level": r.get("level", 0),
            "prompt": r["prompt"],
            "answer": r["answer"],
            "clean_think": body,
            "corrupted_think": corrupted,
            "corruption_type": kind,
            "corruption_detail": detail,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {out}  {len(rows)} twin pairs "
          f"(delete {counts['delete']}, substitute {counts['substitute']}, skipped {skipped})")

    for kind in ("delete", "substitute"):
        ex = next((r for r in rows if r["corruption_type"] == kind), None)
        if ex:
            print(f"\n=== example: {kind}  ({ex['_uid']}, level {ex['level']}) "
                  f"detail: {ex['corruption_detail']}")
            print("--- clean ---");     print(ex["clean_think"][:600])
            print("--- corrupted ---"); print(ex["corrupted_think"][:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
