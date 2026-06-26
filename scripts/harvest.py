"""Stage 1 — Blind Harvest.

Samples K rollouts per problem from a base model with only the problem
statement visible, then writes one JSONL line per (uid, candidate_idx).

Blindness is non-negotiable (§2): no gold conditioning, no few-shot, no
teacher of higher capability. The high-entropy decision-token distribution
that downstream stages depend on collapses if the model is shown the answer.

Chat-template driven: prompts are produced via
`tokenizer.apply_chat_template(messages, add_generation_prompt=True)` so the
script is model-agnostic (Qwen <|im_start|>, Gemma <start_of_turn>, etc.).
The <think> prefill, if any, is controlled by --prefill_think: many base
models (Gemma-4 base included) emit thinking tags inside their template.

For faster inference on Gemma-4, pass --assistant_model to enable vLLM
speculative decoding with `google/gemma-4-E4B-it-assistant` as the draft.

Resume-safe: append-only output, scans existing (uid, k) pairs on startup.
Multi-worker safe: workers slice the pool by _uid hash mod n_workers, each
writing to its own output file. Never have N workers append to a shared file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _uid_hash_mod(uid: str, n: int) -> int:
    return int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % n


def _load_system_prompt(path: str | None) -> str:
    default = (
        "Place all your step-by-step reasoning between <think> and </think> tags. "
        "After </think>, give the final answer."
    )
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return f.read().strip() or default


def _scan_seen(output: str) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    if not os.path.exists(output):
        return seen
    with open(output) as f:
        for line in f:
            try:
                r = json.loads(line)
                seen.add((r["_uid"], r.get("candidate_idx", -1)))
            except json.JSONDecodeError:
                # Last-line in-flight write at crash time: skip, do not reject file.
                continue
    return seen


def _build_prompt(tokenizer, sys_prompt: str, user_text: str,
                  prefill_think: bool) -> str:
    """Build the prompt via the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if prefill_think and "<think>" not in prompt:
        prompt = prompt + "<think>\n"
    return prompt


def _spec_config(args):
    """Build vLLM speculative_config dict if --assistant_model was passed."""
    if not args.assistant_model:
        return None
    return {
        "model": args.assistant_model,
        "method": "draft",
        "num_speculative_tokens": args.num_speculative_tokens,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE Stage 1 blind harvest")
    ap.add_argument("--input", required=True, help="Pool JSONL (_uid, prompt, ground_truth)")
    ap.add_argument("--output", required=True, help="Append-only output JSONL")
    ap.add_argument("--model", required=True, help="HF model id or path")
    ap.add_argument("--assistant_model", default=None,
                    help="Draft model for vLLM speculative decoding "
                         "(e.g. google/gemma-4-E4B-it-assistant)")
    ap.add_argument("--num_speculative_tokens", type=int, default=3)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1, help=">0 to enable")
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--max_model_len", type=int, default=33024)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--system_prompt_file", default=None)
    ap.add_argument("--prefill_think", action="store_true",
                    help="Append '<think>\\n' to the chat-template prompt "
                         "if the template does not already include it.")
    ap.add_argument("--no_prefill_think", dest="prefill_think", action="store_false")
    ap.set_defaults(prefill_think=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--data_root",
        default=None,
        help="If set, prepended to sys.path so whetstone.verify resolves",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        sys.path.insert(0, args.data_root)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    seen = _scan_seen(args.output)
    print(f"[resume] {len(seen)} (uid, k) pairs already done", flush=True)

    sys_prompt = _load_system_prompt(args.system_prompt_file)

    problems: list[dict] = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            uid = r["_uid"]
            if args.n_workers > 1 and _uid_hash_mod(uid, args.n_workers) != args.worker_id:
                continue
            prompt = r.get("prompt") or r.get("problem") or ""
            gold = r.get("ground_truth") or r.get("gold") or ""
            for k in range(args.K):
                if (uid, k) in seen:
                    continue
                problems.append({"uid": uid, "k": k, "prompt": prompt, "gold": gold})

    if not problems:
        print("[harvest] nothing to do", flush=True)
        return
    print(f"[load] {len(problems)} rollouts to generate", flush=True)

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
    spec = _spec_config(args)
    if spec is not None:
        llm_kwargs["speculative_config"] = spec
        print(f"[harvest] speculative decoding with {spec['model']}", flush=True)
    llm = LLM(**llm_kwargs)

    sp_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if args.top_k > 0:
        sp_kwargs["top_k"] = args.top_k
    sp = SamplingParams(**sp_kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)
    batch = max(1, args.batch)
    model_name = os.path.basename(os.path.normpath(args.model))

    for i in range(0, len(problems), batch):
        chunk = problems[i : i + batch]
        prompts = [_build_prompt(tokenizer, sys_prompt, p["prompt"], args.prefill_think)
                   for p in chunk]
        outs = llm.generate(prompts, sp)
        for p, out in zip(chunk, outs):
            text = out.outputs[0].text
            # If we prefilled "<think>\n", the completion does not start with it;
            # prepend so extract_answer() in whetstone.verify sees the same shape
            # it sees at eval time.
            if args.prefill_think and not text.lstrip().startswith("<think>"):
                text = "<think>\n" + text
            rec = {
                "_uid": p["uid"],
                "candidate_idx": p["k"],
                "prompt": p["prompt"],
                "ground_truth": p["gold"],
                "completion": text,
                "n_tokens": len(out.outputs[0].token_ids),
                "model": model_name,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "worker_id": args.worker_id,
            }
            out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        print(f"[gen] {i + len(chunk)}/{len(problems)}", flush=True)

    out_f.close()
    print("[harvest] done", flush=True)


if __name__ == "__main__":
    main()
