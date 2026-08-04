"""Part 1 of packet P4 — build the register-token set R (design §12.3, §2).

R is the mask of the inoculation loss: ``L = Σ_{t ∈ R ∩ think} CE_t + α·L_SED``.
Get R wrong and Round 0 either trains nothing (R too small) or teaches the
scorer the corpus's *content* rather than its *style* (R polluted), which is the
overtrained state — register becomes argmax, residual prose reads spiky.

The selection rule (design §12.3) is Light-IF's type-aggregation logic with an
inverted purpose — select what to *install*, not what to protect:

    R_stats = { types : mean surprisal > p75 across eligible types
                        AND across-occurrence std < median std }
    R       = R_stats ∪ structural whitelist (card §2)

The two conditions do different jobs. High mean says the type is expensive under
π_0. Low across-occurrence std says it is expensive *consistently* — priced the
same wherever it appears, which is what a style token looks like. A content
token is expensive in one place and cheap in another, so it has high std and is
excluded. Eligibility is ≥ 10 occurrences: with fewer, the std estimate is
noise and the filter admits whatever happens to be rare.

The whitelist enters by fiat, below the occurrence floor, because that is its
purpose: activity 005 finding 7 measured ``case``/``✗`` at ~0% of this corpus,
so the branch vocabulary can never be selected statistically from it — and the
32B teacher's branch-keeping traces (activity 006) are written in exactly that
vocabulary.

Scoring runs against a resident vLLM server (one ``/v1/completions`` prefill per
record, ``prompt_logprobs=2``) or an in-process HF forward. Both paths go
through :mod:`whetstone.round0` so the sequence construction is identical to
every other part of the packet.

Usage (turing, with a Qwen3-1.7B server up):
    python scripts/build_register_tokenset.py \
        --train /data/whetstone/corpora/seed_register_qwen/train.jsonl \
        --server http://127.0.0.1:8000/v1 \
        --out /data/whetstone/runs/round0/R_tokenset.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0 import (  # noqa: E402
    build_sequence,
    load_jsonl,
    percentile,
    read_whitelist_strings,
    scores_from_prompt_logprobs,
    whitelist_dropped,
    whitelist_token_ids,
)

MODEL = "Qwen/Qwen3-1.7B"


def score_via_server(seqs, base_url: str, model: str, concurrency: int):
    """One prefill per sequence against a resident vLLM server.

    ``max_tokens=1`` because we want the prompt pass only; ``prompt_logprobs=2``
    returns the top-2 plus the actual token whenever it falls outside them,
    which is exactly what d_t needs (design §12.2).

    Token ids are posted directly rather than text: re-tokenizing decoded text
    does not round-trip at the ``<think>`` boundary, and the masks were computed
    on these ids.
    """
    import asyncio

    import aiohttp

    async def run():
        out: Dict[str, object] = {}
        sem = asyncio.Semaphore(concurrency)
        timeout = aiohttp.ClientTimeout(total=1800)

        async def one(session, seq):
            async with sem:
                payload = {
                    "model": model,
                    "prompt": list(seq.ids),
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 2,
                }
                async with session.post(f"{base_url}/completions", json=payload) as r:
                    r.raise_for_status()
                    body = await r.json()
                return seq.uid, body["choices"][0]["prompt_logprobs"]

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [asyncio.create_task(one(session, s)) for s in seqs]
            done = 0
            for fut in asyncio.as_completed(tasks):
                uid, pl = await fut
                out[uid] = pl
                done += 1
                if done % 50 == 0:
                    print(f"  scored {done}/{len(seqs)}", flush=True)
        return out

    return asyncio.run(run())


def score_via_hf(seqs, model_path: str, topk: int = 2):
    """In-process HF forward fallback — same outputs, no server needed."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    out: Dict[str, object] = {}
    for n, seq in enumerate(seqs, 1):
        ids = torch.tensor([seq.ids], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        entries: List[object] = [None]
        for t in range(1, len(seq.ids)):
            # logits at t-1 predict token t (packet §4 alignment rule).
            row = logprobs[t - 1]
            top = torch.topk(row, topk)
            d = {}
            for rank, (lp, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
                d[tid] = {"logprob": lp, "rank": rank}
            actual = seq.ids[t]
            if actual not in d:
                rank = int((row > row[actual]).sum().item()) + 1
                d[actual] = {"logprob": row[actual].item(), "rank": rank}
            entries.append({str(k): v for k, v in d.items()})
        out[seq.uid] = entries
        if n % 25 == 0:
            print(f"  scored {n}/{len(seqs)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--server", default=None, help="vLLM base url, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--served-model-name", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--mean-pct", type=float, default=75.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--whitelist-all-pieces", action="store_true",
                    help="packet-literal whitelist expansion; admits digits/whitespace "
                         "(26.9%% of think tokens) — see whetstone.round0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores-out", default=None, help="optional raw per-type dump")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    records = load_jsonl(args.train)
    if args.limit:
        records = records[: args.limit]
    print(f"[load] {len(records)} records from {args.train}")

    seqs = []
    for r in records:
        seqs.append(
            build_sequence(
                tok,
                uid=r["_uid"],
                problem=r["prompt"],
                think_body=r["compact_think"],
                answer=r["answer"],
                level=r.get("level", 0),
                require_gate=True,          # packet §4: assert g == 1 for every record
            )
        )
    think_tok = sum(len(s.think_positions) for s in seqs)
    print(f"[build] {len(seqs)} sequences, all g=1; {think_tok:,} think tokens")

    t0 = time.time()
    if args.server:
        print(f"[score] server {args.server} (concurrency {args.concurrency})")
        raw = score_via_server(seqs, args.server, args.served_model_name, args.concurrency)
        backend = f"vllm-server:{args.server}"
    else:
        print("[score] in-process HF forward")
        raw = score_via_hf(seqs, args.model)
        backend = "hf-forward"
    print(f"[score] done in {time.time() - t0:.0f}s")

    # --- type aggregation over think-segment positions only ----------------
    per_type: Dict[int, List[float]] = defaultdict(list)
    for seq in seqs:
        scores = scores_from_prompt_logprobs(seq.ids, raw[seq.uid])
        for pos in seq.think_positions:
            if pos == 0:
                continue
            per_type[seq.ids[pos]].append(scores.surprisal[pos])

    eligible = {tid: v for tid, v in per_type.items() if len(v) >= args.min_count}
    print(f"[types] {len(per_type):,} distinct; {len(eligible):,} with ≥{args.min_count} occurrences")

    means = {tid: statistics.fmean(v) for tid, v in eligible.items()}
    stds = {tid: (statistics.pstdev(v) if len(v) > 1 else 0.0) for tid, v in eligible.items()}
    mean_thr = percentile(list(means.values()), args.mean_pct)
    std_thr = statistics.median(stds.values())
    print(f"[thresholds] mean surprisal p{args.mean_pct:g} = {mean_thr:.4f} nats; "
          f"median across-occurrence std = {std_thr:.4f}")

    r_stats = {
        tid for tid in eligible if means[tid] > mean_thr and stds[tid] < std_thr
    }
    print(f"[R_stats] {len(r_stats)} types (high mean AND low std)")

    wl_strings = read_whitelist_strings(args.card)
    wl = whitelist_token_ids(tok, wl_strings, single_token_only=not args.whitelist_all_pieces)
    dropped = whitelist_dropped(tok, wl_strings)
    print(f"[whitelist] card §2: {len(wl_strings)} strings → {len(wl)} token ids")
    if not args.whitelist_all_pieces and dropped:
        n_drop_tok = sum(len(per_type.get(t, [])) for v in dropped.values() for t in v)
        print(f"[whitelist] dropped {len(dropped)} multi-piece variants "
              f"(their pieces cover {n_drop_tok:,} think tokens = "
              f"{100 * n_drop_tok / max(think_tok, 1):.1f}% of the corpus):")
        for variant, pieces in sorted(dropped.items()):
            surf = [tok.decode([p]) for p in pieces]
            print(f"    {variant!r:<8} → {pieces} {surf}")

    entry: Dict[str, dict] = {}
    for tid in sorted(r_stats | set(wl)):
        occ = per_type.get(tid, [])
        entry[str(tid)] = {
            "surface": tok.decode([tid]),
            "mean": round(statistics.fmean(occ), 6) if occ else None,
            "std": round(statistics.pstdev(occ), 6) if len(occ) > 1 else (0.0 if occ else None),
            "count": len(occ),
            "source": "stats" if tid in r_stats else "whitelist",
        }
    both = sum(1 for tid in r_stats if tid in wl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": {
            "train": args.train,
            "n_records": len(seqs),
            "think_tokens": think_tok,
            "backend": backend,
            "model": args.model,
            "card": args.card,
            "min_count": args.min_count,
            "mean_pct": args.mean_pct,
            "mean_threshold": mean_thr,
            "std_threshold": std_thr,
            "n_eligible_types": len(eligible),
            "n_stats": len(r_stats),
            "n_whitelist": len(wl),
            "n_overlap": both,
            "n_total": len(entry),
        }
    }
    out_path.write_text(json.dumps({**meta, **entry}, ensure_ascii=False, indent=1))
    print(f"[write] {out_path}  |R| = {len(entry)} "
          f"(stats {len(r_stats)}, whitelist {len(wl)}, overlap {both})")

    if args.scores_out:
        Path(args.scores_out).write_text(
            json.dumps(
                {
                    str(t): {"mean": means.get(t), "std": stds.get(t),
                             "count": len(per_type[t]), "surface": tok.decode([t])}
                    for t in eligible
                },
                ensure_ascii=False,
            )
        )

    # --- the eyeball (packet §5 step 5) ------------------------------------
    ranked = sorted(
        (tid for tid in entry if entry[tid]["mean"] is not None),
        key=lambda t: entry[t]["mean"],
        reverse=True,
    )
    print("\ntop-50 R surfaces by mean surprisal "
          "(if ordinary English words dominate, the p75 threshold is wrong):")
    print(f"{'id':>7} {'surface':<16} {'mean':>8} {'std':>7} {'count':>7}  src")
    for tid in ranked[:50]:
        e = entry[tid]
        print(f"{tid:>7} {e['surface']!r:<16} {e['mean']:>8.3f} {e['std']:>7.3f} "
              f"{e['count']:>7}  {e['source']}")

    print("\nwhitelist coverage in corpus (branch class is expected to be thin):")
    for s in wl_strings:
        ids = whitelist_token_ids(tok, [s], single_token_only=not args.whitelist_all_pieces)
        c = sum(len(per_type.get(t, [])) for t in ids)
        print(f"  {s!r:<8} ids={sorted(ids)} occurrences={c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
