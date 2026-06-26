"""Stage 1 §2.8 — Deterministic verification gate.

Filters a Stage 1 harvest JSONL by the deterministic verifier
(whetstone.verify.verify_response). Keeps only (uid, candidate_idx) pairs
whose final answer matches gold. Expected yield ≈ 30–60% on mid-difficulty
pools, ≈ 10–20% on competition-difficulty pools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1 §2.8 verification filter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default=None,
                    help="If set, prepended to sys.path so whetstone.verify resolves")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        sys.path.insert(0, args.data_root)

    from whetstone.verify import verify_response

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_in = n_keep = 0
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            r = json.loads(line)
            n_in += 1
            completion = r.get("completion", "")
            gold = r.get("ground_truth", "")
            if verify_response(completion, gold):
                f_out.write(line)
                n_keep += 1
            if n_in % 1000 == 0:
                rate = n_keep / n_in
                print(f"[verify] {n_in} read, {n_keep} kept ({rate:.2%})", flush=True)

    rate = n_keep / max(1, n_in)
    print(f"[verify] DONE. {n_keep}/{n_in} = {rate:.2%} kept", flush=True)
    if rate < 0.10:
        print("[verify] WARN: yield < 10%. Recheck the §2.1 calibration metrics before "
              "scaling up.", file=sys.stderr)


if __name__ == "__main__":
    main()
