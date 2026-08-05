"""Assemble the Stage-B training set from a Stage-A corpus (design §4, packet P6 Part 1).

The student sees a **plain, unprivileged prompt**: no register card, no gold, no
verbose trace, no system message. The register enters the weights here and only
here (design §4), so anything privileged leaking into this prompt would make the
whole stage measure the wrong thing.

Every sequence is built by :func:`whetstone.round0.build_sequence` — the shared
construction of packet P4 §4. That is not a convenience: the ZPD gate pass
(Part 2) scores these exact token ids and the trainer trains on them, and a
record built one way and scored another is the silently-inverted meter this
project keeps re-learning about. Ids are written to disk so no consumer ever
re-tokenizes.

Output record (one per trace):

    {"_uid", "level", "weight", "trace_idx",
     "prompt_len", "ids",
     "think_start", "think_end", "answer_start", "answer_end"}

Spans are absolute positions into ``ids`` and half-open, matching
:class:`whetstone.segments.SegmentMasks`; the trainer rebuilds masks from them
rather than storing two mask arrays per record.

**Weights are per PROBLEM, never per trace** (activity 008, binding). The golden
corpus has one trace per problem so every weight is 1.0. The unfiltered control
arm has 1-3 traces per problem, and ``n_kept`` is the teacher's sampling luck,
not the problem's value — a 3-keep problem left at weight 1.0 each would get
triple the curriculum share for no reason anyone chose.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import BOUNDARY_IDS, build_completion_text, build_sequence, load_jsonl
from whetstone.verify import verify_response


def _assert_no_boundary_tokens(tokenizer, uid: str, field: str, text: str) -> None:
    """Refuse text of ours that encodes a `<think>`/`</think>`/`<|im_*|>` token.

    ``build_sequence`` already guards the problem statement. The teacher wrote
    ``compact_think`` and ``answer``, and a stray boundary token there would
    parse as a real segment edge — duplicated-open/close, or a think block that
    ends early — corrupting every mask downstream instead of failing loudly.
    """
    stray = BOUNDARY_IDS.intersection(tokenizer.encode(text, add_special_tokens=False))
    if stray:
        raise ValueError(f"{uid}: {field} encodes boundary tokens {sorted(stray)}")


def build(args) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = load_jsonl(args.corpus)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[stageb] {len(rows)} corpus records from {args.corpus}", flush=True)

    # Per-problem weighting: one share per problem, split across its traces.
    n_kept = collections.Counter(r["_uid"] for r in rows)
    multi = sum(1 for v in n_kept.values() if v > 1)
    print(f"[stageb] {len(n_kept)} problems; {multi} with >1 trace "
          f"(max {max(n_kept.values())}) -> weight = 1/n_kept", flush=True)

    seen = collections.Counter()
    out, failures = [], collections.Counter()
    think_toks = answer_toks = 0

    for r in rows:
        uid = r["_uid"]
        try:
            _assert_no_boundary_tokens(tokenizer, uid, "compact_think", r["compact_think"])
            _assert_no_boundary_tokens(tokenizer, uid, "answer", r["answer"])

            seq = build_sequence(
                tokenizer,
                uid=uid,
                problem=r["prompt"],
                think_body=r["compact_think"],
                answer=r["answer"],
                level=int(r.get("level", 0)),
                require_gate=True,          # g=0 is a build error here, not a filter
            )

            # The corpus is certified; a verifier failure means THIS assembly is
            # wrong (wrong field, wrong join), not that the corpus is bad.
            completion = build_completion_text(r["compact_think"], r["answer"])
            if not verify_response(completion, r["ground_truth"]):
                raise ValueError("verify_response failed on a certified record")
        except (ValueError, KeyError) as e:
            failures[f"{type(e).__name__}: {str(e)[:100]}"] += 1
            continue

        m = seq.masks
        seen[uid] += 1
        out.append({
            "_uid": uid,
            "level": seq.level,
            "weight": 1.0 / n_kept[uid],
            "trace_idx": seen[uid] - 1,
            "prompt_len": seq.prompt_len,
            "ids": list(seq.ids),
            "think_start": m.think_start,
            "think_end": m.think_end,
            "answer_start": m.answer_start,
            "answer_end": m.answer_end,
        })
        think_toks += m.think_len
        answer_toks += m.answer_len

    if failures:
        print(f"[stageb] {sum(failures.values())} records FAILED assembly:", flush=True)
        for k, v in failures.most_common(10):
            print(f"    {v:5d}  {k}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_level = collections.Counter(r["level"] for r in out)
    tok_by_level: dict = collections.Counter()
    for r in out:
        tok_by_level[r["level"]] += r["think_end"] - r["think_start"]

    meta = {
        "corpus": args.corpus,
        "tokenizer": args.tokenizer,
        "records": len(out),
        "problems": len(seen),
        "failed": sum(failures.values()),
        "failure_reasons": dict(failures),
        "think_tokens_total": think_toks,
        "answer_tokens_total": answer_toks,
        "weight_sum": round(sum(r["weight"] for r in out), 4),
        "records_by_level": dict(sorted(by_level.items())),
        "think_tokens_by_level": dict(sorted(tok_by_level.items())),
        "construction": "whetstone.round0.build_sequence (packet P4 §4)",
        "prompt": "user turn only; no system message, no register card, no gold, "
                  "no verbose trace; enable_thinking=True",
    }
    with open(f"{os.path.splitext(args.out)[0]}.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[stageb] wrote {len(out)} records -> {args.out}", flush=True)
    print(f"[stageb] think tokens {think_toks:,} | answer tokens {answer_toks:,} "
          f"| weight sum {meta['weight_sum']}", flush=True)
    print("[stageb] think-token share by level: "
          + ", ".join(f"L{k} {100*v/max(think_toks,1):.1f}%"
                      for k, v in sorted(tok_by_level.items())), flush=True)
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build the Stage-B training set")
    ap.add_argument("--corpus", required=True,
                    help="Stage-A corpus JSONL (golden_faithfulness.jsonl or selected.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B",
                    help="ORIGINAL checkpoint's tokenizer — never scorer_v1's")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    return ap.parse_args(argv)


def main(argv=None):
    build(parse_args(argv))


if __name__ == "__main__":
    main()
