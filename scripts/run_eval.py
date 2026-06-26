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
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

SYS_PROMPT = (
    "Place all your step-by-step reasoning between <think> and </think> tags. "
    "After </think>, give the final answer."
)


def _read_suite(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _build_prompt(tokenizer, sys_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
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


def _eval_suite(suite_path: str, tokenizer, llm, args) -> dict:
    from vllm import SamplingParams
    rows = _read_suite(suite_path)
    suite_name = rows[0].get("suite", Path(suite_path).stem) if rows else Path(suite_path).stem
    print(f"[eval] {suite_name}: {len(rows)} problems", flush=True)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        n=args.K,
        seed=args.seed,
    )

    prompts = [_build_prompt(tokenizer, args.system_prompt, r["prompt"]) for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    dump_rows = []
    n_correct_at1 = 0
    n_pass_at_k = 0
    n_total = len(rows)
    n_tokens_total = 0
    is_lean = any(r.get("verifier") != "whetstone.verify" for r in rows)

    for rec, out in zip(rows, outs):
        verdicts = []
        for cand in out.outputs:
            text = cand.text
            v = _verdict(rec, text)
            verdicts.append({
                "text": text,
                "correct": v["correct"],
                "verifier": v["verifier"],
                "n_tokens": len(cand.token_ids),
            })
            n_tokens_total += len(cand.token_ids)
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

    summary = {
        "suite": suite_name,
        "n_problems": n_total,
        "K": args.K,
        "temperature": args.temperature,
        "wall_seconds": round(dur, 1),
        "n_tokens_total": n_tokens_total,
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
    ap.add_argument("--system_prompt", default=SYS_PROMPT)
    ap.add_argument("--K", type=int, default=1, help="Samples per problem (pass@K)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=12288)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
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

    summaries = []
    for suite_path in suite_paths:
        result = _eval_suite(suite_path, tokenizer, llm, args)
        suite_name = result["summary"]["suite"]
        rows_path = os.path.join(args.output_dir, f"{suite_name}__{model_tag}.jsonl")
        with open(rows_path, "w") as f:
            for r in result["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries.append(result["summary"])
        s = result["summary"]
        print(f"[eval] {suite_name}: "
              f"strict@1={s['strict_accuracy_at1']} "
              f"pass@{s['K']}={s['pass_at_k']} "
              f"tokens/correct={s['tokens_per_correct']} "
              f"({s['wall_seconds']}s)",
              flush=True)

    summary_path = os.path.join(args.output_dir, f"summary__{model_tag}.json")
    with open(summary_path, "w") as f:
        json.dump({"model": model_tag, "suites": summaries}, f, indent=2)
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
