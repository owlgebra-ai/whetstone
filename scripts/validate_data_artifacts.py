"""Acceptance checks for the P1 data artifacts (packet P1 definition of done).

Run after any rebuild of the pool / SCA arm / eval suites:

  * every JSONL parses on **every** line (not just `head -1`) and carries the
    `_uid / prompt / ground_truth / level` schema with the right types;
  * `_uid`s are unique within a file;
  * **gold round-trip**: for N random rows per file,
    `verify_response("</think>\\n\\boxed{<gold>}", gold)` must be True. This is
    the check that catches gold-format mistakes (LaTeX mangling, stray units,
    `####` leftovers) the moment they are introduced rather than three stages
    later as a mysteriously low reward yield.

Suites whose records declare a non-`whetstone.verify` verifier (HumanEval →
`code-exec`) are schema-checked but excluded from the round-trip.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

from whetstone.poolutil import read_jsonl
from whetstone.verify import verify_response

REQUIRED = {"_uid": str, "prompt": str, "ground_truth": str, "level": int}


def check_file(path: str, n_sample: int, seed: int) -> dict:
    res: dict = {"file": path, "errors": [], "warnings": []}
    rows = read_jsonl(path)
    res["rows"] = len(rows)
    if not rows:
        res["errors"].append("empty file")
        return res

    for i, r in enumerate(rows):
        for k, t in REQUIRED.items():
            if k not in r:
                res["errors"].append(f"row {i}: missing {k}")
            elif not isinstance(r[k], t):
                res["errors"].append(f"row {i}: {k} is {type(r[k]).__name__}, want {t.__name__}")
        if not r.get("prompt", "").strip():
            res["errors"].append(f"row {i}: empty prompt")
        if not str(r.get("ground_truth", "")).strip():
            res["errors"].append(f"row {i}: empty ground_truth")
        if len(res["errors"]) > 20:
            res["errors"].append("... truncated")
            break

    uids = [r.get("_uid") for r in rows]
    if len(set(uids)) != len(uids):
        res["errors"].append(f"duplicate _uid: {len(uids) - len(set(uids))} collisions")

    # Gold round-trip, grouped by `source` (pool) or the whole file (eval suite).
    verifier = rows[0].get("verifier", "whetstone.verify")
    if verifier != "whetstone.verify":
        res["roundtrip"] = f"skipped (verifier={verifier})"
        return res

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("source") or r.get("suite") or "all", []).append(r)
    rng = random.Random(seed)
    rt: dict[str, str] = {}
    for g, grows in sorted(groups.items()):
        sample = rng.sample(grows, min(n_sample, len(grows)))
        bad = [r for r in sample
               if not verify_response("</think>\n\\boxed{" + r["ground_truth"] + "}",
                                      r["ground_truth"])]
        rt[g] = f"{len(sample) - len(bad)}/{len(sample)}"
        if bad:
            res["warnings"].append(
                f"{g}: {len(bad)} gold(s) fail the boxed round-trip, e.g. "
                + json.dumps([b["ground_truth"] for b in bad[:3]], ensure_ascii=False))
    res["roundtrip"] = rt
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate P1 data artifacts")
    ap.add_argument("--paths", nargs="+", required=True,
                    help="JSONL files or directories of JSONLs")
    ap.add_argument("--n_sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            files.append(p)

    n_err = 0
    for f in files:
        r = check_file(f, args.n_sample, args.seed)
        status = "FAIL" if r["errors"] else ("WARN" if r["warnings"] else "ok")
        n_err += len(r["errors"])
        print(f"[{status:4s}] {os.path.basename(f):28s} rows={r['rows']:6d} "
              f"roundtrip={r.get('roundtrip')}")
        for e in r["errors"]:
            print(f"        ERROR {e}")
        for w in r["warnings"]:
            print(f"        warn  {w}")
    print("\nschema errors:", n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
