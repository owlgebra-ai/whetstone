"""Stage-A draft scoring and structural annotation (packet P5 Part 2).

Drains the append-only raw corpus against the frozen ``scorer_v1`` on
**spark:8100**, one teacher-forced prefill per surviving draft, and writes a
scores sidecar keyed by ``(_uid, candidate_idx)``. Runs concurrently with
generation: ``--follow`` polls the raw corpus for new drafts.

**The unprivileged prompt is what gets scored.** G_spike asks "how followable is
this trace *to the student*", and the student never sees the gold answer, the
register card, or the verbose trace. Scoring the teacher's privileged prompt
would measure followability-given-the-answer, which is high for every draft
including the ones that assert the answer instead of deriving it — the
inverted-meter class this project keeps guarding against. So the sequence is
rebuilt from scratch through :mod:`whetstone.round0`: student tokenizer, student
chat template, no system prompt, ``enable_thinking=True``. That module is the
mandatory shared path — a record scored under one construction and thresholded
under another is exactly the silent inversion (CLAUDE.md).

**Structural annotation is source-conditional.** ``verify_kept``/``branch_kept``
only mean something against the trace's own verbose source: "kept the
verification its source had". With no source they degrade to "contains a
``chk:``" — an unconditional notation requirement, which is why
``structural_gate.py`` leaves ``--require_branch`` off by default. Drafts with no
verbose trace are therefore flagged ``no_source`` and get compact-side presence
flags only, and selection is written to know the difference between an earned
True and a vacuous one.

λ=1 with G_spike computed at **both** β=5 and β=10 (both stored, selection uses
β=10 — packet §3). Pinned scorer constants from activity 007: τ_spike 2.25,
τ_leap 3.175.

Usage::

    # on spark, alongside the resident scorer
    python scripts/score_drafts.py --follow \\
        --drafts /data/whetstone/corpora/stagea_raw/drafts.jsonl \\
        --subset /data/whetstone/corpora/stagea/subset_stagea.jsonl \\
        --output /data/whetstone/corpora/stagea_raw/scores.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import (
    build_sequence, g_spike, percentile, scores_from_prompt_logprobs,
)
from whetstone.runio import repair_tail, scan_seen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structural_gate import (  # noqa: E402
    COMPACT_BRANCH, COMPACT_VERIFY, features,
)

STUDENT_MODEL = "Qwen/Qwen3-1.7B"
#: activity 007 finding 6 — pinned, recorded on every record so a later
#: threshold change is visible against the scores it was applied to.
TAU_SPIKE = 2.25
TAU_LEAP = 3.175
B_TARGET = 600

#: Right edges of the think-token surprisal histogram, in nats (last bin is
#: open). Stored per draft so activity 006's open item 2 — "the ZPD masked
#: fraction should be measured on the 32B corpus before Stage B is sized" — can
#: be answered for *any* γ without re-scoring 32,000 drafts. Stage B's band-pass
#: gate is σ(κ(log π_S(τ_t) − γ)), i.e. a threshold on exactly this quantity,
#: and the corpus-wide histogram is what says how much of the teacher's text
#: sits outside the student's reachable zone.
SURPRISAL_BINS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def histogram(values, edges=SURPRISAL_BINS) -> list:
    """Counts per bin; ``len(edges) + 1`` entries, last bin open-ended."""
    out = [0] * (len(edges) + 1)
    for v in values:
        for i, e in enumerate(edges):
            if v < e:
                out[i] += 1
                break
        else:
            out[-1] += 1
    return out


def g_budget(think_tokens: int, b_target: int = B_TARGET) -> float:
    """``exp[−μ·max(0, T_think − B)/B]``, μ=1 (design §3.2, soft tail).

    A soft tail, not a cliff: v1's hard length penalty produced a reward with a
    discontinuity the model learned to sit just under. With a frozen teacher
    this is a *selection* criterion rather than an annealed schedule (activity
    006), so B is pinned at B_target and there is no freeze rule to run.
    """
    return math.exp(-max(0, think_tokens - b_target) / b_target)


def read_new(path: str, seen: set) -> list[dict]:
    """Surviving, not-yet-scored drafts from the raw corpus."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn tail; the writer will finish the line
            if r.get("reject_reason") is not None:
                continue          # rejected drafts are never scored
            if (r["_uid"], r["candidate_idx"], r.get("gen_round", 1)) in seen:
                continue
            out.append(r)
    return out


def annotate(draft: dict, verbose: str = "") -> dict:
    """Structural annotation, honest about whether a source exists.

    ``verbose`` comes from the subset file, not from the draft: the raw corpus
    deliberately does not carry a copy of the verbose trace per draft (it would
    be 8 copies of a ~4.5k-token trace per problem). Passing it in is therefore
    load-bearing — with an empty string every draft reads ``no_source`` and the
    whole structural half of selection silently turns off.
    """
    compact = draft.get("compact_think", "")
    has_branch = bool(COMPACT_BRANCH.search(compact))
    has_verify = bool(COMPACT_VERIFY.search(compact))
    out = {
        "compact_has_branch": has_branch,
        "compact_has_verify": has_verify,
        "compact_lines": len([l for l in compact.splitlines() if l.strip()]),
    }
    if not verbose:
        # No source ⇒ no claim about what was *kept*. Leaving verify_kept=True
        # here (the vacuous value features() returns) would let a draft that
        # never had a verification to keep outrank one that kept a real one.
        out.update({"no_source": True, "src_has_branch": None,
                    "src_has_verify": None, "branch_kept": None,
                    "verify_kept": None, "value_coverage": None})
        return out
    f = features({"verbose_think": verbose, "compact_think": compact})
    out.update({
        "no_source": False,
        # Whether the TEACHER saw this trace. A trace over the 12,288-token
        # conditioning cap still exists and still defines what "kept" means,
        # but the teacher wrote without it — keep the distinction so a later
        # analysis can condition on it instead of rediscovering it.
        "source_seen_by_teacher": draft.get("conditioned_on") == "gold+trace",
        "src_has_branch": f["src_has_branch"],
        "src_has_verify": f["src_has_verify"],
        "branch_kept": f["branch_kept"],
        "verify_kept": f["verify_kept"],
        "value_coverage": f["value_coverage"],
        "n_anchors": f["n_anchors"],
        "invented_frac": f["invented_frac"],
        "lines_per_step": f["lines_per_step"],
        "verbose_steps": f["verbose_steps"],
    })
    return out


async def score_batch(seqs, base_url: str, model: str, concurrency: int,
                      timeout_s: int) -> dict:
    """One prefill per sequence; ``max_tokens=1`` (prompt pass only),
    ``prompt_logprobs=2`` (top-2 plus the actual token — design §12.2).
    Token ids are posted directly: re-tokenizing decoded text does not
    round-trip at the boundary, and the masks were computed on these ids."""
    import aiohttp

    out: dict = {}
    sem = asyncio.Semaphore(concurrency)

    async def one(session, key, seq):
        async with sem:
            payload = {"model": model, "prompt": list(seq.ids), "max_tokens": 1,
                       "temperature": 0.0, "prompt_logprobs": 2}
            for attempt in range(3):
                try:
                    async with session.post(f"{base_url}/completions",
                                            json=payload) as r:
                        if r.status != 200:
                            raise RuntimeError(
                                f"HTTP {r.status}: {(await r.text())[:200]}")
                        body = await r.json()
                    return key, body["choices"][0]["prompt_logprobs"], None
                except Exception as exc:                       # noqa: BLE001
                    if attempt == 2:
                        return key, None, f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(2 ** attempt)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30,
                                    sock_read=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(one(session, k, s)) for k, s in seqs]
        for fut in asyncio.as_completed(tasks):
            key, pl, err = await fut
            out[key] = (pl, err)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--subset", default="/data/whetstone/corpora/stagea/subset_stagea.jsonl",
                    help="source of the verbose traces the structural gate "
                         "scores against — REQUIRED for verify_kept/branch_kept "
                         "to mean anything")
    ap.add_argument("--output", default="/data/whetstone/corpora/stagea_raw/scores.jsonl")
    ap.add_argument("--server", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="whetstone-scorer")
    ap.add_argument("--tokenizer", default=STUDENT_MODEL,
                    help="the STUDENT's tokenizer — the sequence being scored "
                         "is the student's unprivileged one")
    ap.add_argument("--max_len", type=int, default=8191,
                    help="scorer max_model_len minus the 1 sampled token. "
                         "Over-long sequences are recorded as skipped rather "
                         "than dropped, so the rate is visible.")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--b_target", type=int, default=B_TARGET)
    ap.add_argument("--follow", action="store_true",
                    help="keep polling for new drafts (run alongside generation)")
    ap.add_argument("--poll_s", type=int, default=30)
    ap.add_argument("--idle_exit_s", type=int, default=0,
                    help="with --follow, exit after this many seconds with no "
                         "new drafts (0 = never)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    verbose_by_uid: dict[str, str] = {}
    with open(args.subset) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("verbose_think"):
                    verbose_by_uid[r["_uid"]] = r["verbose_think"]
    print(f"[subset] {len(verbose_by_uid)} problems carry a verbose source")
    if not verbose_by_uid:
        raise SystemExit(
            f"[score] no verbose traces in {args.subset}. Every draft would be "
            "annotated no_source and the structural half of selection would "
            "silently do nothing — refusing to run.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dropped = repair_tail(args.output)
    if dropped:
        print(f"[resume] repaired torn tail: dropped {dropped} B")
    # Round-aware, for the same reason as the generator: re-generated drafts
    # carry the same (uid, candidate_idx) and would otherwise read as scored.
    seen = {(u, k, r or 1) for u, k, r in
            scan_seen(args.output, ("_uid", "candidate_idx", "gen_round"))}
    print(f"[resume] {len(seen)} drafts already scored")

    out_f = open(args.output, "a", buffering=1)
    state = Counter()
    t0 = time.time()
    idle_since = None

    while True:
        pending = read_new(args.drafts, seen)
        if not pending:
            if not args.follow:
                break
            idle_since = idle_since or time.time()
            if args.idle_exit_s and time.time() - idle_since > args.idle_exit_s:
                print(f"[idle] no new drafts for {args.idle_exit_s}s — exiting")
                break
            time.sleep(args.poll_s)
            continue
        idle_since = None
        print(f"[queue] {len(pending)} unscored drafts", flush=True)

        for i in range(0, len(pending), args.batch):
            chunk = pending[i:i + args.batch]
            seqs, skipped = [], []
            for d in chunk:
                key = (d["_uid"], d["candidate_idx"], d.get("gen_round", 1))
                try:
                    seq = build_sequence(
                        tokenizer, uid=d["_uid"], problem=d["prompt"],
                        think_body=d["compact_think"], answer=d["answer"],
                        level=d.get("level", 0), require_gate=False)
                except ValueError as exc:
                    skipped.append((key, d, f"build:{exc}"))
                    continue
                if seq.masks.g != 1:
                    # The teacher's own ids gated clean; the student's
                    # re-tokenization of the same text did not. That draft is
                    # unusable downstream (Stage B trains on this construction),
                    # so it is recorded as skipped, not silently kept.
                    skipped.append((key, d, f"student_gate:{seq.masks.reason}"))
                    continue
                if len(seq.ids) > args.max_len:
                    skipped.append((key, d, f"too_long:{len(seq.ids)}"))
                    continue
                seqs.append((key, seq))

            for key, d, reason in skipped:
                state[f"skip:{reason.split(':')[0]}"] += 1
                out_f.write(json.dumps({
                    "_uid": key[0], "candidate_idx": key[1],
                    "gen_round": key[2],
                    "score_skip_reason": reason,
                    **annotate(d, verbose_by_uid.get(key[0], ""))}) + "\n")
                seen.add(key)

            if not seqs:
                continue
            raw = asyncio.run(score_batch(seqs, args.server, args.model,
                                          args.concurrency, args.timeout_s))
            by_key = {k: s for k, s in seqs}
            for key, (pl, err) in raw.items():
                seq = by_key[key]
                d = next(x for x in chunk
                         if (x["_uid"], x["candidate_idx"],
                             x.get("gen_round", 1)) == key)
                if err is not None:
                    state["score_error"] += 1
                    continue      # NOT marked seen: a later pass retries it
                try:
                    sc = scores_from_prompt_logprobs(seq.ids, pl)
                except ValueError as exc:
                    state["contract_violation"] += 1
                    print(f"  !! {key} d_t contract: {exc}", flush=True)
                    continue
                surp, gaps = sc.at(seq.think_positions)
                t_think = seq.masks.think_len
                rec = {
                    "_uid": key[0], "candidate_idx": key[1],
                    "gen_round": key[2],
                    "score_skip_reason": None,
                    "think_dt_mean": round(sum(gaps) / len(gaps), 6) if gaps else None,
                    "think_dt_p95": round(percentile(gaps, 95), 6) if gaps else None,
                    "think_dt_max": round(max(gaps), 6) if gaps else None,
                    "frac_above_tau_leap": (
                        round(sum(g > TAU_LEAP for g in gaps) / len(gaps), 6)
                        if gaps else None),
                    "g_spike_b5": g_spike(gaps, lam=1.0, beta=5.0) if gaps else None,
                    "g_spike_b10": g_spike(gaps, lam=1.0, beta=10.0) if gaps else None,
                    "g_budget": round(g_budget(t_think, args.b_target), 6),
                    # Student-side surprisal of the teacher's think tokens.
                    # Not used by selection — this is Stage B's sizing input
                    # (activity 006 open item 2): how much of the teacher's
                    # text sits outside the student's reachable zone.
                    "think_surprisal_mean": (
                        round(sum(surp) / len(surp), 6) if surp else None),
                    "think_surprisal_p50": (
                        round(percentile(surp, 50), 6) if surp else None),
                    "think_surprisal_p90": (
                        round(percentile(surp, 90), 6) if surp else None),
                    "think_surprisal_hist": histogram(surp),
                    "surprisal_bin_edges": list(SURPRISAL_BINS),
                    "scored_think_tokens": t_think,
                    "scored_answer_tokens": seq.masks.answer_len,
                    "scored_seq_tokens": len(seq.ids),
                    "scorer_model": args.model,
                    "tau_spike": TAU_SPIKE, "tau_leap": TAU_LEAP,
                    "b_target": args.b_target,
                    **annotate(d, verbose_by_uid.get(key[0], "")),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen.add(key)
                state["scored"] += 1

            done = state["scored"]
            if done and done % (args.batch * 4) < args.batch:
                rate = done / max(1e-9, (time.time() - t0) / 60)
                print(f"[scored] {done}  {rate:.0f}/min  "
                      f"{dict(state)}", flush=True)

        out_f.flush()
        os.fsync(out_f.fileno())
        if not args.follow:
            break

    mins = (time.time() - t0) / 60
    print(f"\n[done] {state['scored']} scored in {mins:.1f} min "
          f"({state['scored']/max(1e-9,mins):.0f}/min)")
    for k, v in sorted(state.items()):
        print(f"  {k:<28} {v:>6}")
    out_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
