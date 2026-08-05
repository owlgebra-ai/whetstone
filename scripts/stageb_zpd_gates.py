"""ZPD gate precompute: teacher-forced scoring of the Stage-B corpus (packet P6 Part 2).

The ZPD band-pass weight of design §4.1

    w_t = sigmoid(kappa * (log pi_S(tau_t) - gamma)) * (1 + alpha_nov * min(S_t, s_cap))

needs ``log pi_S(tau_t)`` for every completion token under **the student as it is
at the start of the round**. Design §12.2 precomputes that offline: the scorer is
frozen within a round, so one teacher-forced pass per round is exact, and the
online alternative would double every training step's cost for the same numbers.

Round 1 scores under the ORIGINAL checkpoint; round 2 re-scores the same corpus
under the round-1 student. **Round 2 with round-1 gates is invalid** (design
§4.3, CLAUDE.md invariant: stale gates are a named drift failure), so the sidecar
records which pi_S produced these numbers, along with a content hash of that
checkpoint, and the trainer refuses to start when they disagree with its own
round. The check belongs in code, not in a checklist.

Scoring is the P0 contract path — token ids posted directly (the masks were
computed on these ids and re-tokenizing does not round-trip at the ``<think>``
boundary), ``max_tokens=1`` for a prefill-only pass, ``prompt_logprobs=2``, and
:func:`whetstone.round0.scores_from_prompt_logprobs` to convert. Position 0 is
unconditioned and carries NaN; it is never a completion position, so it never
reaches a weight.

Output: ``<out>.npz`` with one float32 array per record (``S_t`` over completion
positions, i.e. ``-log pi_S``) plus ``<out>.meta.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import percentile, scores_from_prompt_logprobs


def pi_s_fingerprint(model: str) -> dict:
    """Identify the checkpoint these gates were produced under.

    For a local directory: sha256 over ``config.json`` plus the (name, size) of
    every weight shard. That changes whenever the weights change and costs
    milliseconds, unlike hashing multi-GB shards. For an HF repo id: the id
    itself (its revision is pinned by the HF cache the server loaded from).
    """
    if os.path.isdir(model):
        h = hashlib.sha256()
        cfg = os.path.join(model, "config.json")
        if os.path.exists(cfg):
            h.update(open(cfg, "rb").read())
        for name in sorted(os.listdir(model)):
            if name.endswith((".safetensors", ".bin", ".index.json")):
                h.update(name.encode())
                h.update(str(os.path.getsize(os.path.join(model, name))).encode())
        return {"pi_s": os.path.abspath(model), "pi_s_sha": h.hexdigest()[:16],
                "pi_s_kind": "local_dir"}
    return {"pi_s": model, "pi_s_sha": hashlib.sha256(model.encode()).hexdigest()[:16],
            "pi_s_kind": "hf_repo"}


async def score_all(records, base_url: str, model: str, concurrency: int,
                    timeout_s: int, progress_every: int = 200) -> dict:
    """``{key: (prompt_logprobs | None, error | None)}``, one prefill per record."""
    import aiohttp

    out: dict = {}
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}
    t0 = time.time()

    async def one(session, key, ids):
        async with sem:
            payload = {"model": model, "prompt": list(ids), "max_tokens": 1,
                       "temperature": 0.0, "prompt_logprobs": 2}
            for attempt in range(3):
                try:
                    async with session.post(f"{base_url}/completions", json=payload) as r:
                        if r.status != 200:
                            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
                        body = await r.json()
                    res = (key, body["choices"][0]["prompt_logprobs"], None)
                    break
                except Exception as exc:                       # noqa: BLE001
                    if attempt == 2:
                        res = (key, None, f"{type(exc).__name__}: {exc}")
                        break
                    await asyncio.sleep(2 ** attempt)
            done["n"] += 1
            if done["n"] % progress_every == 0:
                el = time.time() - t0
                print(f"[zpd] {done['n']}/{len(records)} scored "
                      f"({done['n']/max(el,1e-9):.1f}/s, {el:.0f}s)", flush=True)
            return res

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(one(session, k, ids)) for k, ids in records]
        for fut in asyncio.as_completed(tasks):
            key, pl, err = await fut
            out[key] = (pl, err)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="Part-1 train.jsonl (carries ids)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--server", default="http://127.0.0.1:8101/v1",
                    help="pi-of-round server. NOT :8100 — that is frozen scorer_v1 "
                         "and scoring the corpus under it would reproduce activity "
                         "008's lower-bound histogram, not this round's gates.")
    ap.add_argument("--model", required=True,
                    help="--served-model-name of the pi-of-round server")
    ap.add_argument("--pi_s", required=True,
                    help="checkpoint path/id the server is serving; recorded and "
                         "hashed into the sidecar for the trainer's staleness assert")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train)]
    if args.limit:
        rows = rows[: args.limit]
    keyed = [(f"{r['_uid']}#{r['trace_idx']}", r) for r in rows]
    print(f"[zpd] {len(rows)} records | pi_S = {args.pi_s} (round {args.round})", flush=True)

    t0 = time.time()
    raw = asyncio.run(score_all(
        [(k, r["ids"]) for k, r in keyed], args.server, args.model,
        args.concurrency, args.timeout_s))
    print(f"[zpd] scoring pass: {time.time() - t0:.0f}s", flush=True)

    arrays: dict = {}
    errors: dict = {}
    all_think, all_answer = [], []

    for key, r in keyed:
        pl, err = raw.get(key, (None, "missing"))
        if pl is None:
            errors[key] = err
            continue
        try:
            sc = scores_from_prompt_logprobs(r["ids"], pl)
        except ValueError as e:
            errors[key] = f"contract: {e}"
            continue
        p0 = r["prompt_len"]
        # Completion positions only. Position 0 is unconditioned (NaN) and is
        # always inside the prompt, so it can never land in this slice.
        s = np.asarray(sc.surprisal[p0:], dtype=np.float32)
        if np.isnan(s).any():
            errors[key] = "NaN inside completion span"
            continue
        arrays[key] = s
        ts, te = r["think_start"] - p0, r["think_end"] - p0
        as_, ae = r["answer_start"] - p0, r["answer_end"] - p0
        all_think.extend(s[ts:te].tolist())
        all_answer.extend(s[as_:ae].tolist())

    if errors:
        print(f"[zpd] {len(errors)} records FAILED:", flush=True)
        for k, v in list(errors.items())[:10]:
            print(f"    {k}: {v}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    def _stats(xs, label):
        if not xs:
            return {}
        return {
            f"{label}_n": len(xs),
            f"{label}_mean": round(float(np.mean(xs)), 6),
            f"{label}_p50": round(percentile(xs, 50), 6),
            f"{label}_p90": round(percentile(xs, 90), 6),
            f"{label}_p99": round(percentile(xs, 99), 6),
            f"{label}_max": round(float(np.max(xs)), 6),
        }

    meta = {
        "train": os.path.abspath(args.train),
        "round": args.round,
        "server": args.server,
        "served_model_name": args.model,
        **pi_s_fingerprint(args.pi_s),
        "records_scored": len(arrays),
        "records_failed": len(errors),
        "errors": {k: v for k, v in list(errors.items())[:50]},
        "units": "S_t = -log pi_S(tau_t), nats, completion positions only",
        **_stats(all_think, "think_surprisal"),
        **_stats(all_answer, "answer_surprisal"),
    }
    with open(f"{os.path.splitext(args.out)[0]}.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[zpd] {len(arrays)} arrays -> {args.out}", flush=True)
    if all_think:
        print(f"[zpd] think S_t: mean {meta['think_surprisal_mean']} "
              f"p50 {meta['think_surprisal_p50']} p90 {meta['think_surprisal_p90']} "
              f"p99 {meta['think_surprisal_p99']} max {meta['think_surprisal_max']}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
