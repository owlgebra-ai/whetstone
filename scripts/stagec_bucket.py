"""K=8 curriculum bucketing for Stage C (packet P7 Part 0.4 / Part 4 / Part 9).

Produces the table Stage C's curriculum is built from: for every problem, how
many of K samples are correct under the **current** policy. Three buckets:

  * ``0/8``   — out of the batches entirely; pedagogy rescue's clientele (Part 5)
  * ``1–7/8`` — the mixed groups; the *only* source of within-group advantage,
                and therefore the only problems DAPO can learn from
  * ``8/8``   — saturated; dropped by dynamic sampling

The same script runs three times in the packet's life: once on the init
checkpoint (Phase 1's curriculum), once at the Phase-1 endpoint (Phase 2's
input), and again at Phase 2's endpoint (the Phase-3 decision). *The re-bucket,
not the calendar, decides when RL is done.*

Sampling must match the RL rollout sampler exactly (**T=1.0, top-p 1.0**) or the
buckets mis-predict group composition — see activity 010's packet-correction
note, which resolves P7 Part 0.4's ``T=0.7`` against the Part-1 table's ``T=1.0``
in favour of the table.

Grading is **strict** (:mod:`whetstone.reward.strict`); the as-scored verdict is
recorded alongside every candidate so the two are always reportable side by side
and the leniency gap can be watched for widening.

Usage (turing)::

    python scripts/stagec_bucket.py \\
        --model /data/whetstone/ckpt/stageb/golden/round1/final \\
        --uids /data/whetstone/corpora/stagea/subset_stagea_uids.json \\
        --seen_uids /data/whetstone/corpora/stagea_golden/golden_faithfulness.jsonl \\
        --out_dir /data/whetstone/runs/stagec_buckets/init
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_jsonl
from whetstone.reward.strict import verify_strict
from whetstone.segments import blank_token_ids_for, parse_segments


def _seed_for(uid: str, base_seed: int) -> int:
    """Deterministic per-problem seed (packet §11: seed-per-rollout).

    vLLM draws ``n`` independent samples from one request, so the seed is
    per-problem rather than per-rollout; the duplicate-rate metric below is what
    actually verifies the group is not degenerate. A group of byte-identical
    members has zero within-group advantage and DAPO silently learns nothing
    from it, so the rate is reported, not assumed.
    """
    h = hashlib.sha1(f"{uid}:{base_seed}".encode()).hexdigest()
    return int(h[:8], 16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--uids", required=True, help="JSON list of _uids to bucket")
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl")
    ap.add_argument("--seen_uids", default=None,
                    help="JSONL whose _uid field marks SFT-seen problems "
                         "(the memorization control's 'seen' side)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=12288,
                    help="activity 010 finding 4: 8,192 truncates 5.6%% of "
                         "well-formed generations on this pool's DeepMath half")
    ap.add_argument("--max_model_len", type=int, default=13312)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    per_problem_path = os.path.join(args.out_dir, "buckets.jsonl")
    summary_path = os.path.join(args.out_dir, "buckets_summary.json")
    if os.path.exists(summary_path) and not args.force:
        print(f"[bucket] {summary_path} exists; --force to redo")
        return 0

    uids = json.load(open(args.uids))
    if args.limit:
        uids = uids[: args.limit]
    uid_set = set(uids)
    pool = {r["_uid"]: r for r in read_jsonl(args.pool) if r["_uid"] in uid_set}
    missing = uid_set - set(pool)
    if missing:
        raise SystemExit(f"{len(missing)} uids not found in {args.pool}")

    seen: set = set()
    if args.seen_uids:
        seen = {r["_uid"] for r in read_jsonl(args.seen_uids)}
    print(f"[bucket] {len(uids)} problems | seen={len(uid_set & seen)} "
          f"unseen={len(uid_set - seen)} | K={args.K} T={args.temperature} "
          f"top_p={args.top_p} cap={args.max_tokens}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    blank_ids = blank_token_ids_for(tok)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, trust_remote_code=True)

    rows = [pool[u] for u in uids]
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
        for r in rows
    ]
    # One request per problem so each carries its own deterministic seed.
    sps = [
        SamplingParams(temperature=args.temperature, top_p=args.top_p,
                       max_tokens=args.max_tokens, n=args.K,
                       seed=_seed_for(r["_uid"], args.seed))
        for r in rows
    ]

    t0 = time.time()
    outs = llm.generate(prompts, sps)
    dur = time.time() - t0
    print(f"[bucket] generation done in {dur/60:.1f} min", flush=True)

    out_rows = []
    n_dup_groups = 0
    for rec, out in zip(rows, outs):
        gold = rec.get("ground_truth", "")
        cands = []
        texts = []
        for c in out.outputs:
            ids = list(c.token_ids)
            m = parse_segments(ids, prompt_len=0, blank_token_ids=blank_ids)
            v = verify_strict(c.text, gold)
            # Packet §6: malformed (g=0, incl. cap-hits) scores R_acc = 0
            # regardless of what the text contains.
            strict_ok = bool(v.strict) and m.g == 1
            cands.append({
                "strict": strict_ok,
                "as_scored": bool(v.as_scored),
                "strict_reason": v.reason,
                "g": m.g,
                "gate_reason": m.reason,
                "think_tokens": m.think_len,
                "answer_tokens": m.answer_len,
                "total_tokens": len(ids),
                "finish_reason": c.finish_reason,
            })
            texts.append(c.text)

        n_strict = sum(c["strict"] for c in cands)
        n_as = sum(c["as_scored"] and c["g"] == 1 for c in cands)
        n_distinct = len(set(texts))
        if n_distinct < len(texts):
            n_dup_groups += 1
        bucket = "0/K" if n_strict == 0 else ("K/K" if n_strict == len(cands) else "mixed")
        out_rows.append({
            "_uid": rec["_uid"],
            "level": int(rec.get("level", 0)),
            "source": rec.get("source", ""),
            "seen": rec["_uid"] in seen,
            "n_strict": n_strict,
            "n_as_scored": n_as,
            "K": len(cands),
            "bucket": bucket,
            "p_hat": n_strict / max(1, len(cands)),
            "n_distinct_completions": n_distinct,
            "think_median": statistics.median(
                [c["think_tokens"] for c in cands if c["g"] == 1] or [0]),
            "candidates": cands,
        })

    write_jsonl(per_problem_path, out_rows)

    # --- summary: by bucket, by level, by seen/unseen, and cross-tabulated ---
    def _tab(rows_):
        c = collections.Counter(r["bucket"] for r in rows_)
        n = max(1, len(rows_))
        return {"n": len(rows_), "0/K": c["0/K"], "mixed": c["mixed"], "K/K": c["K/K"],
                "mixed_frac": c["mixed"] / n, "zero_frac": c["0/K"] / n,
                "sat_frac": c["K/K"] / n,
                "pass_at_1_strict": statistics.mean(
                    [r["candidates"][0]["strict"] for r in rows_]) if rows_ else None,
                "pass_at_k_strict": statistics.mean(
                    [r["n_strict"] > 0 for r in rows_]) if rows_ else None,
                "pass_at_1_as_scored": statistics.mean(
                    [r["candidates"][0]["as_scored"] and r["candidates"][0]["g"] == 1
                     for r in rows_]) if rows_ else None}

    all_c = [c for r in out_rows for c in r["candidates"]]
    summary = {
        "config": vars(args),
        "model": args.model,
        "n_problems": len(out_rows),
        "generation_minutes": dur / 60,
        "overall": _tab(out_rows),
        "by_seen": {str(k): _tab([r for r in out_rows if r["seen"] is k])
                    for k in (True, False)},
        "by_level": {str(lv): _tab([r for r in out_rows if r["level"] == lv])
                     for lv in sorted({r["level"] for r in out_rows})},
        "by_level_seen": {
            f"L{lv}_{'seen' if s else 'unseen'}": _tab(
                [r for r in out_rows if r["level"] == lv and r["seen"] is s])
            for lv in sorted({r["level"] for r in out_rows}) for s in (True, False)
        },
        "candidate_stats": {
            "n": len(all_c),
            "g_rate": statistics.mean([c["g"] for c in all_c]),
            "cap_hit_rate": statistics.mean(
                [c["finish_reason"] == "length" for c in all_c]),
            "lenient_only_rate": statistics.mean(
                [bool(c["as_scored"]) and not c["strict"] and c["g"] == 1
                 for c in all_c]),
            "think_median": statistics.median(
                [c["think_tokens"] for c in all_c if c["g"] == 1] or [0]),
            "answer_median": statistics.median(
                [c["answer_tokens"] for c in all_c if c["g"] == 1] or [0]),
        },
        "groups_with_duplicate_completions": n_dup_groups,
        "duplicate_group_rate": n_dup_groups / max(1, len(out_rows)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    o = summary["overall"]
    print(f"[bucket] 0/K {o['zero_frac']:.1%} | mixed {o['mixed_frac']:.1%} | "
          f"K/K {o['sat_frac']:.1%} | strict Pass@1 {o['pass_at_1_strict']:.2%} | "
          f"pass@{args.K} {o['pass_at_k_strict']:.2%}")
    print(f"[bucket] dup groups {summary['duplicate_group_rate']:.2%} | "
          f"cap-hit {summary['candidate_stats']['cap_hit_rate']:.2%} | "
          f"g {summary['candidate_stats']['g_rate']:.2%}")
    print(f"[bucket] -> {per_problem_path}\n[bucket] -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
