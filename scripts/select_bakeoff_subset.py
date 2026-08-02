"""Select the register bake-off subset (packet P3a, "Inputs").

Both arms of the bake-off must compress **the same 50 traces**, chosen once and
recorded by `_uid`, so the comparison isolates the register card. This script
picks them and emits a compression-ready corpus.

Selection rule (packet P3a):
  1. start from the P2 entropy-audit rollouts (200 traces, Qwen3-1.7B, 16k);
  2. keep only traces that are **verifier-correct AND segment-parser
     gate-passing** (`g == 1`) — a trace whose reasoning never reaches the gold
     answer is not register material (v1 §3.7), and a cap-hit trace has no
     answer segment to copy through;
  3. proportional level-stratified sample of 50 (`poolutil.stratified_sample`;
     equal-count strata are impossible — the level histogram is peaked at 5–8,
     activity 002 note 1), fixed seed.

The think/answer split is taken from the **token-level parser** on vLLM's own
`completion_token_ids`, never from a string split of the decoded text
(CLAUDE.md invariant). The decoded segments are what the compressor rewrites
and copies through respectively.

The audit rollouts predate `entropy_audit.py` carrying `ground_truth`, so golds
are joined back from `probe.jsonl` by `_uid`.

Usage (turing or spark — CPU only apart from the tokenizer)::

    python scripts/select_bakeoff_subset.py \\
        --rollouts /data/whetstone/runs/entropy_audit/rollouts.jsonl \\
        --probe    /data/whetstone/runs/entropy_audit/probe.jsonl \\
        --out_dir  /data/whetstone/runs/register_bakeoff --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, stratified_sample, write_jsonl
from whetstone.segments import blank_token_ids_for, parse_segments
from whetstone.verify import verify_response


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", default="/data/whetstone/runs/entropy_audit/rollouts.jsonl")
    ap.add_argument("--probe", default="/data/whetstone/runs/entropy_audit/probe.jsonl")
    ap.add_argument("--out_dir", default="/data/whetstone/runs/register_bakeoff")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    blank = blank_token_ids_for(tok)

    gold = {r["_uid"]: r.get("ground_truth", "") for r in read_jsonl(args.probe)}
    rows = read_jsonl(args.rollouts)
    print(f"[in] {len(rows)} rollouts, {len(gold)} probe golds")

    eligible: list[dict] = []
    n_gate_fail = n_wrong = n_no_gold = 0
    for r in rows:
        gt = gold.get(r["_uid"], "")
        if not gt:
            n_no_gold += 1
            continue
        m = parse_segments(r["completion_token_ids"], blank_token_ids=blank)
        if m.g != 1:
            n_gate_fail += 1
            continue
        if not verify_response(r["completion"], gt):
            n_wrong += 1
            continue
        ids = r["completion_token_ids"]
        think = tok.decode([t for t, k in zip(ids, m.think_mask) if k]).strip()
        answer = tok.decode([t for t, k in zip(ids, m.answer_mask) if k]).strip()
        if not think or not answer:
            continue
        eligible.append({
            "_uid": r["_uid"],
            "level": r.get("level"),
            "prompt": r["prompt"],
            "ground_truth": gt,
            "verbose_think": think,
            "answer": answer,
            "verbose_think_tokens": int(m.think_len),
            "answer_tokens": int(m.answer_len),
            "src": os.path.basename(args.rollouts),
        })

    print(f"[filter] eligible {len(eligible)}  "
          f"(gate-fail {n_gate_fail}, verifier-wrong {n_wrong}, no-gold {n_no_gold})")
    if len(eligible) < args.n:
        print(f"[warn] only {len(eligible)} eligible traces for n={args.n}")

    sample = stratified_sample(eligible, lambda r: str(r.get("level", "_")),
                               args.n, random.Random(args.seed))
    sample.sort(key=lambda r: r["_uid"])          # deterministic file order

    os.makedirs(args.out_dir, exist_ok=True)
    subset_path = os.path.join(args.out_dir, "subset.jsonl")
    uids_path = os.path.join(args.out_dir, "subset_uids.json")
    write_jsonl(subset_path, sample)
    with open(uids_path, "w") as f:
        json.dump([r["_uid"] for r in sample], f, indent=1)

    lv_all = Counter(str(r.get("level")) for r in eligible)
    lv_sel = Counter(str(r.get("level")) for r in sample)
    tt = sorted(r["verbose_think_tokens"] for r in sample)
    print(f"[out] {len(sample)} traces -> {subset_path}")
    print(f"      {uids_path}")
    print(f"      level  eligible -> selected: "
          + ", ".join(f"{k}:{lv_all[k]}->{lv_sel[k]}" for k in sorted(lv_all)))
    print(f"      verbose think tokens: median {tt[len(tt)//2]}, "
          f"min {tt[0]}, max {tt[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
