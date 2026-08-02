"""Pick which verified seed traces get compressed (packet P3 Part 2, "Input").

The verifier gate alone is not enough to feed the compressor. A trace also has
to be **structurally well-formed** (:func:`whetstone.segments.parse_segments`,
``g == 1``): Part 2 rewrites the think segment and copies the answer segment
through untouched, so a rollout with no locatable ``</think>`` has nothing to
copy and would silently produce a record whose "answer" is really truncated
reasoning. This is the same eligibility rule the bake-off used
(``select_bakeoff_subset.py``), applied at seed-corpus scale.

Selection rule:
  1. verifier-correct (input is already ``seed_verified.jsonl``, re-asserted);
  2. parser gate ``g == 1``;
  3. think segment short enough that ``scaffold + problem + think + output cap``
     fits the compressor's ``--max-model-len`` (see ``--max-think-tokens``);
  4. **one candidate per problem** — lowest ``candidate_idx`` among those that
     pass. Two compressions of the same problem are near-duplicates and would
     let one problem contribute twice to a corpus whose whole point is
     coverage;
  5. proportional level-stratified sample of ``--n``.

The think/answer split comes from the **token-level** parser on vLLM's own
``completion_token_ids``, never from a string split of the decoded text — the
boundary token merges differently with its neighbours on a re-tokenize, and
every downstream index shifts (design §12.1).

Sizing: the seed register corpus targets 300–1,000 accepted traces after the
Δlogp gate, which ran at 66% in the bake-off. ``--n 1200`` is the default so a
pass rate anywhere in 30–80% still lands inside the target band.

Usage::

    python scripts/select_compression_inputs.py \\
        --input   /data/whetstone/corpora/seed/seed_verified.jsonl \\
        --out_dir /data/whetstone/corpora/seed_register --n 1200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import stratified_sample, write_jsonl, write_meta
from whetstone.segments import blank_token_ids_for, parse_segments
from whetstone.verify import verify_response


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="seed_verified.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-think-tokens", type=int, default=28000,
                    help="Skip traces whose think segment cannot fit the "
                         "compressor's context alongside the card and the "
                         "output cap. 0 disables the check.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    blank = blank_token_ids_for(tok)

    best: dict[str, dict] = {}
    n_in = n_gate = n_wrong = n_long = n_noids = 0

    with open(args.input) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            uid = r["_uid"]
            cidx = r.get("candidate_idx", 0)
            if uid in best and best[uid]["src_candidate_idx"] <= cidx:
                continue                       # already have an earlier candidate
            if not verify_response(r.get("completion", ""), r.get("ground_truth", "")):
                n_wrong += 1
                continue
            ids = r.get("completion_token_ids") or []
            if not ids:
                n_noids += 1
                continue
            m = parse_segments(ids, blank_token_ids=blank)
            if m.g != 1:
                n_gate += 1
                continue
            if args.max_think_tokens and m.think_len > args.max_think_tokens:
                n_long += 1
                continue
            think = tok.decode(ids[m.think_start:m.think_end]).strip()
            answer = tok.decode(ids[m.answer_start:m.answer_end]).strip()
            if not think or not answer:
                continue
            best[uid] = {
                "_uid": uid,
                "src_candidate_idx": cidx,
                "level": r.get("level"),
                "source": r.get("source"),
                "prompt": r.get("prompt", ""),
                "ground_truth": r.get("ground_truth", ""),
                "verbose_think": think,
                "answer": answer,
                "verbose_think_tokens": int(m.think_len),
                "answer_tokens": int(m.answer_len),
            }

    eligible = list(best.values())
    print(f"[in] {n_in} verified rollouts -> {len(eligible)} eligible problems "
          f"(gate-fail {n_gate}, verifier-wrong {n_wrong}, "
          f"too-long {n_long}, no-token-ids {n_noids})")
    if len(eligible) < args.n:
        print(f"[warn] only {len(eligible)} eligible for n={args.n}; taking all")

    sample = stratified_sample(eligible, lambda r: str(r.get("level", "_")),
                               args.n, random.Random(args.seed))
    sample.sort(key=lambda r: r["_uid"])          # deterministic file order

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "compression_inputs.jsonl")
    write_jsonl(out_path, sample)

    lv_all = Counter(str(r.get("level")) for r in eligible)
    lv_sel = Counter(str(r.get("level")) for r in sample)
    tt = sorted(r["verbose_think_tokens"] for r in sample)
    write_meta(out_path, {
        "builder": "scripts/select_compression_inputs.py",
        "packet": "P3 Part 2",
        "input": args.input,
        "n_verified_rollouts": n_in,
        "n_eligible_problems": len(eligible),
        "n_selected": len(sample),
        "seed": args.seed,
        "max_think_tokens": args.max_think_tokens,
        "rejects": {"gate_fail": n_gate, "verifier_wrong": n_wrong,
                    "think_too_long": n_long, "no_token_ids": n_noids},
        "by_level_eligible": {k: lv_all[k] for k in sorted(lv_all)},
        "by_level_selected": {k: lv_sel[k] for k in sorted(lv_sel)},
        "verbose_think_tokens": {
            "median": tt[len(tt) // 2] if tt else 0,
            "p90": tt[int(0.9 * len(tt))] if tt else 0,
            "max": tt[-1] if tt else 0,
        },
    })

    print(f"[out] {len(sample)} traces -> {out_path}")
    print("      level  eligible -> selected: " + ", ".join(
        f"{k}:{lv_all[k]}->{lv_sel[k]}" for k in sorted(lv_all)))
    if tt:
        print(f"      verbose think tokens: median {tt[len(tt)//2]}, "
              f"p90 {tt[int(0.9*len(tt))]}, max {tt[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
