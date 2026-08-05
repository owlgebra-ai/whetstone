"""Run eval on a suite JSONL with vLLM + whetstone.verify grading.

For each suite:
  * Generate K completions per problem (greedy by default; --temperature > 0
    enables pass@K metrics).
  * Verify each via whetstone.verify.verify_response (post-</think> extraction,
    v4.6.1 patches).
  * Emit per-problem JSONL dump and a per-suite summary entry.

For miniF2F (suite tag = "minif2f"), grading requires Lean — this script
writes the generations but skips the verify step (records "verifier": "lean"
in the per-problem record).

Use the same entrypoint for baseline runs: just point --model at the base
model (e.g. google/gemma-4-E4B-it). calc_metrics.py aggregates across runs
into a comparison table.

**Baseline-card fields (added by P6 / activity 009).** The F3 gate compares a
trained student against the original checkpoint on four numbers per suite, and
this runner previously reported none of them — only a total token count, which
is precisely the combined-length number CLAUDE.md forbids. Each summary now
carries:

  * ``pass_at_1_mean`` / ``pass_at_1_std`` — accuracy of sample index k over the
    whole suite, for each k in 0..K-1, then mean ± sample std across the K
    decoding draws. This is the "Pass@1 ± seed std" of design §12.7 and the
    SCA/DeepCompress convention; ``strict_accuracy_at1`` (sample 0 alone) is
    retained for continuity with earlier runs but is one draw, not the metric.
  * ``think_tokens_median`` / ``answer_tokens_median`` — **separately**, via
    ``whetstone.segments.parse_segments`` on the generated token ids. Never a
    combined number: segment drift hides in the sum.
  * ``cap_hit_rate`` — fraction of generations that ran into ``--max_tokens``.
  * ``g_rate`` — fraction that parse as well-formed (SCA quality gate g=1).

Segment medians are taken over **g=1 generations only**, with ``*_median_all``
alongside: a truncated rollout has no ``</think>``, so its think length is
"everything generated" and its answer length is 0, and mixing those into the
median makes a cap-hit look like verbosity. ``cap_hit_rate`` is reported next to
them so the excluded mass is always visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

# v1's system prompt. RETAINED FOR REFERENCE ONLY — it is no longer the default.
# The P2 calibration probe (activity 003) measured it against no system prompt at
# all on 100 rollouts each: with it, format compliance 94% and 6 rollouts emitted
# a DUPLICATED </think>; without it, 100% compliance, zero structural failures,
# and +8 points of accuracy. Qwen3 thinks natively and does not need to be told
# about the tags — naming them in the prompt is what makes it re-emit one.
# Pass --system_prompt "<text>" to restore a system message.
SYS_PROMPT_V1 = (
    "Place all your step-by-step reasoning between <think> and </think> tags. "
    "After </think>, give the final answer."
)


def _read_suite(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _build_prompt(tokenizer, sys_prompt: str, user_text: str,
                  enable_thinking: bool = True) -> str:
    """Render the eval prompt through the tokenizer's chat template.

    P2 fix (handed forward by P1 / activity 002 note 4): ``enable_thinking``
    must be passed on **every** hybrid-Qwen3 template call — rollout, scoring
    and eval alike (ROADMAP rule 4). Without it the flag defaults to the
    template's own default; on Qwen3-1.7B that happens to be thinking-on today,
    but relying on a template default is exactly how an eval silently starts
    measuring a non-thinking model.

    An empty ``sys_prompt`` sends no system message at all — Qwen3 thinks
    natively and needs no ``<think>`` instruction.
    """
    messages = [{"role": "user", "content": user_text}]
    if sys_prompt:
        messages.insert(0, {"role": "system", "content": sys_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _model_tag(model: str) -> str:
    return os.path.basename(os.path.normpath(model))[:60]


def _verdict(rec: dict, completion: str) -> dict:
    from whetstone.verify import verify_response
    verifier = rec.get("verifier", "whetstone.verify")
    if verifier == "whetstone.verify":
        ok = verify_response(completion, rec.get("ground_truth", ""))
        return {"correct": bool(ok), "verifier": verifier}
    # lean / judge — emit None; grading happens out-of-band.
    return {"correct": None, "verifier": verifier}


def _segment_stats(token_ids, blank_ids) -> dict:
    """`(think_len, answer_len, g, reason)` for one generation's token ids.

    ``token_ids`` is the completion alone (vLLM does not echo the prompt), and
    under ``enable_thinking=True`` it *starts with* ``<think>`` because the Qwen3
    template does not pre-fill it (activity 003) — so ``prompt_len=0`` is right.
    """
    from whetstone.segments import parse_segments

    try:
        m = parse_segments(list(token_ids), prompt_len=0, blank_token_ids=blank_ids)
    except ValueError as e:  # multi-turn assert — a template bug, not a bad rollout
        return {"think_tokens": None, "answer_tokens": None, "g": 0,
                "gate_reason": f"parse_error:{e.__class__.__name__}"}
    return {"think_tokens": m.think_len, "answer_tokens": m.answer_len,
            "g": m.g, "gate_reason": m.reason}


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _pass_at_1(dump_rows: list[dict], K: int) -> tuple[float | None, float | None, list]:
    """Mean ± sample std of per-draw accuracy across the K decoding draws.

    Draw k is sample index k of every problem — a full independent pass over the
    suite. Accuracy is computed per draw, then averaged; the std is the spread
    *between draws*, which is what "± seed std" means and what makes a 1-point
    F3 threshold interpretable. With K=1 there is no spread and std is None.
    """
    if not dump_rows or K < 1:
        return None, None, []
    per_draw = []
    for k in range(K):
        graded = [r["candidates"][k]["correct"] for r in dump_rows
                  if len(r["candidates"]) > k and r["candidates"][k]["correct"] is not None]
        if not graded:
            return None, None, []
        per_draw.append(sum(graded) / len(graded))
    mean = statistics.mean(per_draw)
    std = statistics.stdev(per_draw) if len(per_draw) > 1 else None
    return mean, std, per_draw


def _load_model(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if args.assistant_model:
        llm_kwargs["speculative_config"] = {
            "model": args.assistant_model,
            "method": "draft",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        print(f"[eval] speculative decoding with {args.assistant_model}", flush=True)
    llm = LLM(**llm_kwargs)
    return tokenizer, llm


def _eval_suite(suite_path: str, tokenizer, llm, args, blank_ids=frozenset()) -> dict:
    from vllm import SamplingParams
    rows = _read_suite(suite_path)
    suite_name = rows[0].get("suite", Path(suite_path).stem) if rows else Path(suite_path).stem
    n_full = len(rows)
    if args.limit and args.limit < n_full:
        rows = rows[: args.limit]
    print(f"[eval] {suite_name}: {len(rows)} problems"
          + (f" (LIMITED from {n_full} — smoke run, not a reportable number)"
             if len(rows) < n_full else ""), flush=True)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        n=args.K,
        seed=args.seed,
    )

    sys_prompt = "" if args.no_system_prompt else args.system_prompt
    prompts = [_build_prompt(tokenizer, sys_prompt, r["prompt"], args.enable_thinking)
               for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    dump_rows = []
    n_correct_at1 = 0
    n_pass_at_k = 0
    n_total = len(rows)
    n_tokens_total = 0
    is_lean = any(r.get("verifier") != "whetstone.verify" for r in rows)

    n_cap_hit = 0
    n_g_ok = 0
    n_gen = 0
    think_ok, answer_ok, think_all, answer_all = [], [], [], []

    for rec, out in zip(rows, outs):
        verdicts = []
        for cand in out.outputs:
            text = cand.text
            v = _verdict(rec, text)
            seg = _segment_stats(cand.token_ids, blank_ids)
            cap_hit = cand.finish_reason == "length"
            verdicts.append({
                "text": text,
                "correct": v["correct"],
                "verifier": v["verifier"],
                "n_tokens": len(cand.token_ids),
                "finish_reason": cand.finish_reason,
                "cap_hit": cap_hit,
                **seg,
            })
            n_tokens_total += len(cand.token_ids)
            n_gen += 1
            n_cap_hit += int(cap_hit)
            n_g_ok += int(seg["g"] == 1)
            think_all.append(seg["think_tokens"])
            answer_all.append(seg["answer_tokens"])
            if seg["g"] == 1:
                think_ok.append(seg["think_tokens"])
                answer_ok.append(seg["answer_tokens"])
        any_correct = any(v["correct"] for v in verdicts)
        at1_correct = bool(verdicts[0]["correct"]) if verdicts else False
        if not is_lean:
            if at1_correct:
                n_correct_at1 += 1
            if any_correct:
                n_pass_at_k += 1
        dump_rows.append({
            "_uid": rec.get("_uid"),
            "prompt": rec.get("prompt"),
            "ground_truth": rec.get("ground_truth"),
            "suite": suite_name,
            "candidates": verdicts,
            "at1_correct": at1_correct,
            "pass_at_k": any_correct,
        })

    p1_mean, p1_std, p1_per_draw = (
        _pass_at_1(dump_rows, args.K) if not is_lean else (None, None, [])
    )

    summary = {
        "suite": suite_name,
        "n_problems": n_total,
        "K": args.K,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "system_prompt": sys_prompt,
        "seed": args.seed,
        # Loud, because a limited run must never be mistaken for a suite number.
        "limited": bool(args.limit and args.limit < n_full),
        "n_problems_full_suite": n_full,
        "wall_seconds": round(dur, 1),
        "n_tokens_total": n_tokens_total,
        # --- the four baseline-card numbers (P6 / activity 009) --------------
        "pass_at_1_mean": p1_mean,
        "pass_at_1_std": p1_std,
        "pass_at_1_per_draw": p1_per_draw,
        "think_tokens_median": _median(think_ok),
        "answer_tokens_median": _median(answer_ok),
        "think_tokens_median_all": _median(think_all),
        "answer_tokens_median_all": _median(answer_all),
        "cap_hit_rate": (n_cap_hit / n_gen) if n_gen else None,
        "g_rate": (n_g_ok / n_gen) if n_gen else None,
        "n_generations": n_gen,
        "segment_median_population": "g==1 generations only; *_median_all spans every generation",
        # --- retained for continuity with pre-009 runs ------------------------
        "strict_accuracy_at1": (n_correct_at1 / n_total) if n_total and not is_lean else None,
        "pass_at_k": (n_pass_at_k / n_total) if n_total and not is_lean else None,
        "verifier": "mixed" if is_lean else "whetstone.verify",
        "model": _model_tag(args.model),
    }
    summary["tokens_per_correct"] = (
        n_tokens_total / n_correct_at1 if n_correct_at1 > 0 else None
    )
    return {"summary": summary, "rows": dump_rows}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE eval runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--assistant_model", default=None,
                    help="Draft model for vLLM speculative decoding")
    ap.add_argument("--num_speculative_tokens", type=int, default=3)
    ap.add_argument("--suites", default=None,
                    help="Comma list of suite JSONL paths or names under --suite_dir")
    ap.add_argument("--suite_dir", default=None,
                    help="Directory containing <suite>.jsonl files")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--system_prompt", default="",
                    help="Empty by default — see SYS_PROMPT_V1 above; the P2 probe "
                         "showed v1's system prompt costs compliance and accuracy "
                         "on Qwen3.")
    # Defaults are the SCA-matched protocol (design §12.7): N=8, T=0.7,
    # top_p=0.95, 32k max new tokens. The old v1 settings (K=1, T=0.0,
    # max_tokens=12288) stay reachable as flags — the calibration probe and the
    # per-checkpoint continuity dashboard both want a cheap run.
    ap.add_argument("--K", type=int, default=8, help="Samples per problem (pass@K)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_model_len", type=int, default=36864)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--no_enable_thinking", dest="enable_thinking",
                    action="store_false",
                    help="Hybrid-Qwen3 templates only; leave ON (ROADMAP rule 4).")
    ap.set_defaults(enable_thinking=True)
    ap.add_argument("--no_system_prompt", action="store_true",
                    help="Send no system message (Qwen3 thinks natively).")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Smoke tests only: evaluate the first N problems of each "
                         "suite. Sets `limited: true` in the summary — a limited "
                         "run is never a reportable suite number.")
    ap.add_argument("--data_root", default=None,
                    help="If set, prepended to sys.path so whetstone.verify resolves")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        sys.path.insert(0, args.data_root)

    # Resolve suite list.
    suite_paths: list[str] = []
    if args.suites:
        for s in args.suites.split(","):
            s = s.strip()
            if not s:
                continue
            if os.path.exists(s):
                suite_paths.append(s)
            elif args.suite_dir and os.path.exists(os.path.join(args.suite_dir, f"{s}.jsonl")):
                suite_paths.append(os.path.join(args.suite_dir, f"{s}.jsonl"))
            else:
                print(f"[eval] WARN: suite not found: {s}", file=sys.stderr)
    elif args.suite_dir:
        suite_paths = sorted(glob.glob(os.path.join(args.suite_dir, "*.jsonl")))
    if not suite_paths:
        raise SystemExit("no suite files resolved; pass --suites or --suite_dir")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, llm = _load_model(args)
    model_tag = _model_tag(args.model)

    # One vocab scan (~seconds), reused for every suite: lets the empty-answer
    # rule catch a rollout that emitted "</think>\n\n" and then hit the cap.
    from whetstone.segments import blank_token_ids_for
    blank_ids = blank_token_ids_for(tokenizer)
    print(f"[eval] {len(blank_ids)} whitespace-only token ids", flush=True)

    summaries = []
    for suite_path in suite_paths:
        result = _eval_suite(suite_path, tokenizer, llm, args, blank_ids)
        suite_name = result["summary"]["suite"]
        rows_path = os.path.join(args.output_dir, f"{suite_name}__{model_tag}.jsonl")
        with open(rows_path, "w") as f:
            for r in result["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries.append(result["summary"])
        s = result["summary"]
        _pct = lambda x: "n/a" if x is None else f"{100 * x:.2f}"
        std = "" if s["pass_at_1_std"] is None else f" ± {100 * s['pass_at_1_std']:.2f}"
        print(f"[eval] {suite_name}: "
              f"Pass@1={_pct(s['pass_at_1_mean'])}{std} "
              f"pass@{s['K']}={_pct(s['pass_at_k'])} | "
              f"think_med={s['think_tokens_median']} "
              f"answer_med={s['answer_tokens_median']} "
              f"cap_hit={_pct(s['cap_hit_rate'])}% "
              f"g={_pct(s['g_rate'])}% "
              f"({s['wall_seconds']}s)",
              flush=True)

    summary_path = os.path.join(args.output_dir, f"summary__{model_tag}.json")
    with open(summary_path, "w") as f:
        json.dump({"model": model_tag, "suites": summaries}, f, indent=2)
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
