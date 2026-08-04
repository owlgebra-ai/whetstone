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

**Inline CPU triage, as each draft lands** (packet §5): segment gate → no boxed
answer inside the thinking block → deterministic verifier. Every draft is
appended to the raw corpus **whether it passes or fails**, carrying its rejection
reason. Raw is append-only truth; selection is a separate, cheap, re-runnable
pass over it and never mutates it.

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
    _BOXED, _git_sha, _sha1, assert_no_boundary_tokens, render_card,
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


def build_prompt(tokenizer, system_prompt: str, rec: dict) -> str:
    if rec["conditioned_on"] == "gold+trace":
        user = USER_GOLD_TRACE.format(problem=rec["prompt"],
                                      gold=rec["ground_truth"],
                                      verbose=rec["verbose_think"])
    else:
        user = USER_GOLD.format(problem=rec["prompt"], gold=rec["ground_truth"])
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def triage(text: str, token_ids: list[int], gold: str) -> dict:
    """Segment gate → boxed-in-think → verifier. Returns the record's verdict
    fields; ``reject_reason`` is ``None`` iff the draft survives.

    Order is load-bearing. A cap-hit draft has no ``</think>`` at all, so its
    "think" text is the whole completion and both later checks would report
    nonsense about it; the gate has to speak first.
    """
    masks = parse_segments(token_ids, prompt_len=0)
    out = {
        "g": masks.g,
        "gate_reason": masks.reason,
        "gate_warnings": list(masks.warnings),
        "think_tokens": masks.think_len,
        "answer_tokens": masks.answer_len,
        "compact_think": "",
        "answer": "",
        "think_has_boxed": False,
        "verify_ok": False,
        "reject_reason": None,
    }
    if masks.g != 1:
        out["reject_reason"] = f"gate:{masks.reason}"
        return out

    # g==1 guarantees exactly one of each boundary, so the string split is safe
    # here (it is not, in general — that is why the gate runs on token ids).
    think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    answer = text.split("</think>", 1)[1].lstrip("\n")
    out["compact_think"] = think
    out["answer"] = answer
    out["think_has_boxed"] = bool(_BOXED.search(think))
    if out["think_has_boxed"]:
        # Card §1.5. At K=8 rejecting beats cleaning: a sibling draft without
        # the trailer almost always exists, and cleaning would silently change
        # what the corpus says the teacher produces.
        out["reject_reason"] = "boxed_in_think"
        return out

    out["verify_ok"] = bool(verify_response(text, gold))
    if not out["verify_ok"]:
        out["reject_reason"] = "verify_fail"
    return out


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
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_problems", type=int, default=0,
                    help="0 = all. The subset is pre-shuffled, so the first N "
                         "is a representative slice — this is what Part 5's "
                         "500-problem calibration checkpoint uses.")
    ap.add_argument("--checkpoint_every", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    card_raw = open(args.card).read()
    card_text, dropped = render_card(card_raw)
    system_prompt = TEACHER_SCAFFOLD.format(card=card_text)
    stats = assert_no_boundary_tokens(system_prompt, tokenizer)
    print(f"[card] {args.card} sha={_git_sha(args.card)[:12]} "
          f"rendered {stats['prompt_tokens']} tok, dropped {dropped}")

    meta = {
        "teacher_model": args.model,
        "card_path": args.card,
        "card_git_sha": _git_sha(args.card),
        "rendered_prompt_sha1": _sha1(system_prompt),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed_base": args.seed,
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

    bodies = [{
        "model": args.model,
        "prompt": build_prompt(tokenizer, system_prompt, rec),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": 1,
        "stream": False,
        "seed": draft_seed(rec["_uid"], k, args.seed),
        "return_token_ids": True,
    } for rec, k in work]

    out_f = open(args.output, "a", buffering=1)
    fail_f = open(f"{args.output}.failed.jsonl", "a", buffering=1)
    state = Counter()
    t0 = time.time()

    def on_result(res) -> None:
        rec, k = work[res.index]
        state["n"] += 1
        if not res.ok:
            state["request_failed"] += 1
            fail_f.write(json.dumps({"_uid": rec["_uid"], "candidate_idx": k,
                                     "error": res.error}) + "\n")
            return
        if not res.token_ids:
            # Without token ids the segment gate cannot run, and re-tokenizing
            # the decoded text does not round-trip at the boundary (§12.1).
            # Treat as a request failure so a resume retries it.
            state["no_token_ids"] += 1
            fail_f.write(json.dumps({"_uid": rec["_uid"], "candidate_idx": k,
                                     "error": "server returned no token_ids — "
                                              "is return_token_ids supported?"}) + "\n")
            return

        v = triage(res.text, res.token_ids, rec["ground_truth"])
        state[f"reject:{v['reject_reason']}" if v["reject_reason"] else "kept"] += 1
        state["cap"] += int(res.finish_reason == "length")

        draft = {
            "_uid": rec["_uid"],
            "candidate_idx": k,
            "level": rec["level"],
            "source": rec["source"],
            "prompt": rec["prompt"],
            "ground_truth": rec["ground_truth"],
            "conditioned_on": rec["conditioned_on"],
            "trace_fallback_reason": rec.get("trace_fallback_reason"),
            "trace_candidate_idx": rec.get("trace_candidate_idx"),
            "verbose_think_tokens": rec.get("verbose_think_tokens"),
            "raw_text": res.text,
            "completion_token_ids": res.token_ids,
            "n_tokens": len(res.token_ids),
            "finish_reason": res.finish_reason,
            "draft_seed": draft_seed(rec["_uid"], k, args.seed),
            **v,
            **meta,
        }
        out_f.write(json.dumps(draft, ensure_ascii=False) + "\n")

        if state["n"] % args.checkpoint_every == 0:
            rate = state["n"] / max(1e-9, (time.time() - t0) / 60)
            checkpoint(args.output, out_f, {
                "done": state["n"], "of": len(work),
                "drafts_per_min": round(rate, 1),
                "eta_h": round((len(work) - state["n"]) / max(1e-9, rate) / 60, 2),
                **{k: v for k, v in state.items() if k != "n"},
            })
            print(f"[{state['n']}/{len(work)}] {rate:.1f}/min  "
                  f"kept {state['kept']}  "
                  f"eta {(len(work)-state['n'])/max(1e-9,rate)/60:.1f} h", flush=True)

    run_completions(args.server, bodies, on_result=on_result,
                    concurrency=args.concurrency)
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
