"""Stage-A teacher generation: K compact drafts per problem (packet P5 Part 1).

The frozen 32B teacher ghostwrites the textbook. It is **privileged** — it sees
the gold answer, the register card, and (where one exists) the student's own
verified verbose trace — and it is never trained and never shipped. What it
emits is a full rollout in the *student's* shape: a thinking block written in the
compact register, then a normal LaTeX solution ending in ``\\boxed{...}``.

Why the shape matters. Stage B trains the student to produce exactly this, from
an unprivileged prompt. So the teacher cannot be run as a text *rewriter* (the
retired ``compress_local_versionB`` flow, ``enable_thinking=False``): that
produces register text but not a rollout. It is run as a **rollout** with
``enable_thinking=True`` (ROADMAP standing rule 4), and the scaffold's job is to
make the thinking block itself be the register rather than the model's native
prose. Whether a thinking model actually complies is a real question and is why
``--max_problems`` exists — smoke 25 problems before committing a day of GPU.

**The scaffold never spells the boundary tags.** The literal strings encode as
the real ``<think>``/``</think>`` token ids even inside prose, so a scaffold that
names them injects a spurious boundary into every prompt and makes the teacher's
own output parse as malformed (P3 gotcha 1, card §1.5/§1.6). The instructions
say "thinking block"; :func:`assert_no_boundary_tokens` enforces it.

**The thinking block is prefilled, and it has to be** (measured, activity 008
smoke run). Asked in a system prompt to think in the compact register, the 32B
complies in the *answer* channel and thinks natively anyway: 50 drafts came back
at **0.11 register markers per 100 think chars against the same model's 2.10
raw baseline** (activity 006), with ``goal`` opening 1 trace out of 38. An
instruction addressed to a thinking model's scratchpad does not reach it — the
scratchpad is where it does what it always does. Prefilling ``<think>\\ngoal:``
puts the sampler *inside* the register at token one, and ``goal`` is the
register's canonical opener (it opens 925 of 960 traces in the Round-0 corpus,
and every card exemplar). The prefill's ids are prepended to the returned ids so
the stored record is a genuine full rollout — never re-tokenized, since the
prefill is what the model actually conditioned on and its ids are what it saw.

**Generation is two-phase, and the boundary is imposed rather than sampled**
(deviation from packet §5's one-request-per-draft; measured, activity 008).
Prefilled into the register, the model writes a clean compact trace and then
imitates the card exemplars all the way to their end — it stops at ``⇒ 8`` or
appends ``$$\\boxed{8}$$`` instead of closing the thinking block and writing a
solution. **48% of prefilled drafts failed on that transition alone** (20%
never emitted a close tag, 28% put the boxed result inside the block), with
perfectly good register content in every one. So phase 1 generates the think
segment with ``</think>`` as a stop string, and phase 2 generates the solution
from the *cleaned* think body with the boundary already written. The shape stops
being something the model has to remember.

Phase 1's output goes through :func:`clean_oneshot` — the same trailer cleaner
activity 005 built for exactly this artifact — and the flags it raises are
recorded per draft rather than swallowed, because a rising trailer rate is card
feedback.

**Inline CPU triage** (packet §5): segment gate → no boxed answer inside the
thinking block → deterministic verifier, all on the assembled completion. Every
draft is appended to the raw corpus **whether it passes or fails**, carrying its
rejection reason. Raw is append-only truth; selection is a separate, cheap,
re-runnable pass over it and never mutates it.

Usage::

    # on turing, with the 32B served on :8000
    python scripts/teacher_generate.py \\
        --subset /data/whetstone/corpora/stagea/subset_stagea.jsonl \\
        --output /data/whetstone/corpora/stagea_raw/drafts.jsonl \\
        --k 8 --concurrency 8

    # smoke first: 25 problems, K=2, then read the summary before scaling up
    python scripts/teacher_generate.py --max_problems 25 --k 2 \\
        --output /data/whetstone/corpora/stagea_raw/smoke.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.runio import checkpoint, repair_tail, run_completions, scan_seen
from whetstone.segments import parse_segments
from whetstone.verify import verify_response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compress_local_versionB import (  # noqa: E402
    _BOXED, _git_sha, _sha1, assert_no_boundary_tokens, clean_oneshot,
    render_card,
)

#: Generation scaffold. Byte-identical across every draft in a run; its sha1 is
#: recorded on each record so a scaffold edit is visible in the corpus itself.
#:
#: Deliberately NOT the compression scaffold: that one addresses a rewriter
#: ("rewrite the reasoning below"), this one addresses a solver whose reasoning
#: must be *written in* the register. The privileged-answer rule is stated twice
#: because it is the one instruction whose violation is invisible downstream —
#: a trace that asserts the gold rather than deriving it verifies clean, reads
#: fluently, and teaches the student nothing.
TEACHER_SCAFFOLD = """You are writing a model solution that a student will learn from. The student
sees only what you write, and must be able to follow every step.

Write your reasoning in the compact register defined by the REGISTER CARD
below. The card is the only authority on notation and style. Your thinking
block must be in that compact register from its very first line — not ordinary
prose, not a narrated monologue. After the thinking block, write a normal,
self-contained LaTeX solution that ends with the final result in \\boxed{{...}}.

Rules:
- The reference answer is given to you. Your reasoning must genuinely DERIVE
  it, step by step. Do not assert it, do not restate it as a premise, and do
  not work backwards from it.
- Preserve every load-bearing fact, variable, equation, constant, case split
  and derivation step. Never fuse two steps; never drop an intermediate value.
- Every case you try and every alternative you reject stays in. Branch
  elimination is reasoning, and it stays.
- If you check your result, keep the check.
- Do not invent content the problem does not support.
- The compact register governs the thinking block ONLY, and a boxed result must
  never appear inside it. The boxed answer belongs in the solution that follows.

REGISTER CARD:
{card}"""

USER_GOLD = """PROBLEM:
{problem}

REFERENCE ANSWER (known correct — your reasoning must derive it):
{gold}"""

USER_GOLD_TRACE = """PROBLEM:
{problem}

REFERENCE ANSWER (known correct — your reasoning must derive it):
{gold}

VERBOSE REASONING (a correct but long solution to the same problem; keep
everything load-bearing in it, drop only the narration and the repetition):
{verbose}"""


def draft_seed(uid: str, k: int, base: int) -> int:
    """Per-draft seed. **Never one seed per group** (activity 005 infra note 3):
    a shared seed collapses K samples into K copies of the same trajectory."""
    h = hashlib.sha1(f"{uid}:{k}:{base}".encode()).hexdigest()[:8]
    return int(h, 16)


#: Assistant prefill that opens the thinking block already in register. No
#: trailing space: a trailing space would be merged into the model's first word
#: piece by BPE and push it off the token boundaries the card was audited on.
PREFILL_DEFAULT = "<think>\ngoal:"


#: Text imposed between the two phases. Written as a format-time constant rather
#: than inline so there is exactly one place the boundary is spelled, and so the
#: assembled completion matches :func:`whetstone.round0.build_completion_text`
#: byte for byte — the construction every downstream scoring pass rebuilds.
THINK_CLOSE = "\n</think>\n\n"


def build_prompt(tokenizer, system_prompt: str, rec: dict, prefill: str) -> str:
    if rec["conditioned_on"] == "gold+trace":
        user = USER_GOLD_TRACE.format(problem=rec["prompt"],
                                      gold=rec["ground_truth"],
                                      verbose=rec["verbose_think"])
    else:
        user = USER_GOLD.format(problem=rec["prompt"], gold=rec["ground_truth"])
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    return text + prefill


def triage(tokenizer, think: str, answer: str, gold: str) -> dict:
    """Assemble the rollout, gate it, verify it. ``reject_reason`` is ``None``
    iff the draft survives.

    The completion is assembled from *text* and tokenized here, rather than
    carrying the sampled ids through, because the two-phase flow imposes the
    boundary itself — and because this is the construction every downstream
    consumer rebuilds (``whetstone.round0.build_sequence``, which re-tokenizes
    think+answer under the *student*). Gating the same construction the scorer
    will gate means a draft cannot pass here and fail there.
    """
    from whetstone.round0 import build_completion_text

    text = build_completion_text(think, answer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    masks = parse_segments(ids, prompt_len=0)
    out = {
        "g": masks.g,
        "gate_reason": masks.reason,
        "gate_warnings": list(masks.warnings),
        "think_tokens": masks.think_len,
        "answer_tokens": masks.answer_len,
        "compact_think": think,
        "answer": answer,
        "raw_text": text,
        "completion_token_ids": ids,
        "n_tokens": len(ids),
        "think_has_boxed": bool(_BOXED.search(think)),
        "verify_ok": False,
        "reject_reason": None,
    }
    if masks.g != 1:
        out["reject_reason"] = f"gate:{masks.reason}"
        return out
    if out["think_has_boxed"]:
        # Card §1.5 survivor check. The phase-1 cleaner already truncates at a
        # boxed flourish, so anything reaching here is a boxed result genuinely
        # embedded mid-register — reject rather than edit it.
        out["reject_reason"] = "boxed_in_think"
        return out
    out["verify_ok"] = bool(verify_response(text, gold))
    if not out["verify_ok"]:
        out["reject_reason"] = "verify_fail"
    return out


def write_reject(fh, rec: dict, k: int, meta: dict, args, reason: str) -> None:
    """Append a phase-level rejection to the raw corpus.

    Rejections are records, not silence: a cap-hit or empty-think draft is
    evidence about the teacher and about the budget, and the raw corpus is where
    the F2 denominators come from.
    """
    fh.write(json.dumps({
        "_uid": rec["_uid"], "candidate_idx": k,
        "level": rec["level"], "source": rec["source"],
        "prompt": rec["prompt"], "ground_truth": rec["ground_truth"],
        "conditioned_on": rec["conditioned_on"],
        "trace_fallback_reason": rec.get("trace_fallback_reason"),
        "g": 0, "gate_reason": reason, "gate_warnings": [],
        "think_tokens": 0, "answer_tokens": 0,
        "compact_think": "", "answer": "", "raw_text": "",
        "completion_token_ids": [], "n_tokens": 0,
        "think_has_boxed": False, "verify_ok": False,
        "reject_reason": reason,
        "draft_seed": draft_seed(rec["_uid"], k, args.seed),
        **meta,
    }, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="/data/whetstone/corpora/stagea/subset_stagea.jsonl")
    ap.add_argument("--output", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-32B-NVFP4")
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to --model; the Qwen3 family shares one "
                         "tokenizer, but the prompt is rendered with the "
                         "TEACHER's template, so it is the teacher's")
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_think_tokens", type=int, default=2048)
    ap.add_argument("--max_answer_tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=512,
                    help="drafts per phase-1/phase-2 round trip. Only affects "
                         "how often results land on disk; resume is per draft.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill", default=PREFILL_DEFAULT,
                    help="assistant prefill opening the thinking block in "
                         "register. Pass '' to disable — but read the module "
                         "docstring first: without it the register does not "
                         "land at all.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_problems", type=int, default=0,
                    help="0 = all. The subset is pre-shuffled, so the first N "
                         "is a representative slice — this is what Part 5's "
                         "500-problem calibration checkpoint uses.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    card_raw = open(args.card).read()
    card_text, dropped = render_card(card_raw)
    system_prompt = TEACHER_SCAFFOLD.format(card=card_text)
    stats = assert_no_boundary_tokens(system_prompt, tokenizer)
    print(f"[card] {args.card} sha={_git_sha(args.card)[:12]} "
          f"rendered {stats['prompt_tokens']} tok, dropped {dropped}")

    prefill_ids = (tokenizer.encode(args.prefill, add_special_tokens=False)
                   if args.prefill else [])
    if args.prefill:
        from whetstone.segments import THINK_OPEN_ID
        if not prefill_ids or prefill_ids[0] != THINK_OPEN_ID:
            raise SystemExit(
                f"[teacher] prefill {args.prefill!r} does not start with the "
                f"<think> token ({THINK_OPEN_ID}); got {prefill_ids[:3]}. The "
                "segment parser would then see a rollout with no think open.")
        print(f"[prefill] {args.prefill!r} -> {len(prefill_ids)} ids "
              f"{prefill_ids}")

    meta = {
        "teacher_model": args.model,
        "card_path": args.card,
        "card_git_sha": _git_sha(args.card),
        "rendered_prompt_sha1": _sha1(system_prompt),
        "prefill": args.prefill,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_think_tokens": args.max_think_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "seed_base": args.seed,
        "generation": "two_phase",
    }

    subset = [json.loads(l) for l in open(args.subset) if l.strip()]
    if args.max_problems:
        subset = subset[:args.max_problems]
    print(f"[subset] {len(subset)} problems x K={args.k} = "
          f"{len(subset)*args.k} drafts")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dropped_b = repair_tail(args.output)
    if dropped_b:
        print(f"[resume] repaired torn tail: dropped {dropped_b} B")
    seen = scan_seen(args.output, ("_uid", "candidate_idx"))
    print(f"[resume] {len(seen)} drafts already on disk")

    # uid-major so a problem's K drafts share one prompt prefix back to back —
    # vLLM's prefix cache then serves the card + problem for all but the first.
    work = [(rec, k) for rec in subset for k in range(args.k)
            if (rec["_uid"], k) not in seen]
    if not work:
        print("[done] nothing left to generate")
        return 0
    print(f"[work] {len(work)} drafts to go")

    out_f = open(args.output, "a", buffering=1)
    fail_f = open(f"{args.output}.failed.jsonl", "a", buffering=1)
    state = Counter()
    t0 = time.time()
    prefill_body = args.prefill.split("\n", 1)[-1] if args.prefill else ""

    def fail(rec, k, msg) -> None:
        state["request_failed"] += 1
        fail_f.write(json.dumps({"_uid": rec["_uid"], "candidate_idx": k,
                                 "error": msg}) + "\n")

    for start in range(0, len(work), args.chunk):
        batch = work[start:start + args.chunk]

        # --- phase 1: the register, stopped at the boundary ---------------
        p1_prompts = [build_prompt(tokenizer, system_prompt, rec, args.prefill)
                      for rec, _ in batch]
        p1: dict[int, object] = {}
        run_completions(
            args.server,
            [{"model": args.model, "prompt": p, "max_tokens": args.max_think_tokens,
              "temperature": args.temperature, "top_p": args.top_p, "n": 1,
              "stream": False, "seed": draft_seed(rec["_uid"], k, args.seed),
              "stop": ["</think>"]}
             for p, (rec, k) in zip(p1_prompts, batch)],
            on_result=lambda r: p1.__setitem__(r.index, r),
            concurrency=args.concurrency)

        # --- phase 2: the solution, from the CLEANED register -------------
        # Conditioning phase 2 on the cleaned think body rather than the raw
        # phase-1 text costs a little prefix-cache reuse and buys the property
        # that matters: the answer is the answer to the trace the corpus
        # actually stores, not to a trailer that was trimmed out of it.
        p2_idx, p2_bodies, thinks = [], [], {}
        for i, (rec, k) in enumerate(batch):
            res = p1.get(i)
            if res is None or not res.ok:
                fail(rec, k, f"phase1: {getattr(res, 'error', 'no result')}")
                continue
            if res.finish_reason == "length":
                state["reject:cap_think"] += 1
                write_reject(out_f, rec, k, meta, args, "cap_think")
                continue
            think, flags = clean_oneshot(prefill_body + res.text)
            for fl in flags:
                state[f"flag_{fl}"] += 1
            if not think:
                state["reject:empty_think"] += 1
                write_reject(out_f, rec, k, meta, args, "empty_think")
                continue
            thinks[i] = (think, flags)
            p2_idx.append(i)
            p2_bodies.append({
                "model": args.model,
                "prompt": p1_prompts[i] + think[len(prefill_body):] + THINK_CLOSE,
                "max_tokens": args.max_answer_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "n": 1, "stream": False,
                "seed": draft_seed(rec["_uid"], k, args.seed + 1),
            })

        p2: dict[int, object] = {}
        if p2_bodies:
            run_completions(args.server, p2_bodies,
                            on_result=lambda r: p2.__setitem__(r.index, r),
                            concurrency=args.concurrency)

        # --- assemble, gate, verify, append -------------------------------
        for j, i in enumerate(p2_idx):
            rec, k = batch[i]
            res = p2.get(j)
            if res is None or not res.ok:
                fail(rec, k, f"phase2: {getattr(res, 'error', 'no result')}")
                continue
            think, flags = thinks[i]
            if res.finish_reason == "length":
                state["reject:cap_answer"] += 1
                write_reject(out_f, rec, k, meta, args, "cap_answer")
                continue
            v = triage(tokenizer, think, res.text.strip(), rec["ground_truth"])
            state["kept" if v["reject_reason"] is None
                  else f"reject:{v['reject_reason']}"] += 1
            out_f.write(json.dumps({
                "_uid": rec["_uid"], "candidate_idx": k,
                "level": rec["level"], "source": rec["source"],
                "prompt": rec["prompt"], "ground_truth": rec["ground_truth"],
                "conditioned_on": rec["conditioned_on"],
                "trace_fallback_reason": rec.get("trace_fallback_reason"),
                "trace_candidate_idx": rec.get("trace_candidate_idx"),
                "verbose_think_tokens": rec.get("verbose_think_tokens"),
                "clean_flags": flags,
                "draft_seed": draft_seed(rec["_uid"], k, args.seed),
                **v, **meta,
            }, ensure_ascii=False) + "\n")

        state["n"] += len(batch)
        rate = state["n"] / max(1e-9, (time.time() - t0) / 60)
        checkpoint(args.output, out_f, {
            "done": state["n"], "of": len(work),
            "drafts_per_min": round(rate, 1),
            "eta_h": round((len(work) - state["n"]) / max(1e-9, rate) / 60, 2),
            **{key: v for key, v in state.items() if key != "n"},
        })
        print(f"[{state['n']}/{len(work)}] {rate:.1f}/min  "
              f"kept {state['kept']}  "
              f"eta {(len(work)-state['n'])/max(1e-9,rate)/60:.1f} h", flush=True)

    out_f.close()
    fail_f.close()

    n = max(1, state["n"])
    mins = (time.time() - t0) / 60
    print(f"\n[done] {state['n']} drafts in {mins:.1f} min "
          f"({state['n']/max(1e-9,mins):.1f}/min)")
    for key, v in sorted(state.items()):
        if key != "n":
            print(f"  {key:<28} {v:>6}  {v/n:>6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
