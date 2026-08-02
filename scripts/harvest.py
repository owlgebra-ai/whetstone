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

Output records carry, besides the completion text:
  * `level` / `source`, passed through from the input pool so yield tables are
    a groupby on this file (packet P3 Part 1) with no join back;
  * `completion_token_ids` — vLLM's own ids. Every consumer of a harvest file
    routes by <think> boundaries, and `whetstone.segments` is deliberately
    token-level: re-tokenizing the decoded text does not round-trip at the
    boundary (design §12.1). Re-deriving ids later is not equivalent, so they
    are stored, not recomputed;
  * `finish_reason` — distinguishes a cap-hit (`length`) from a clean stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _uid_hash_mod(uid: str, n: int) -> int:
    return int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % n


# v1's system prompt. RETAINED FOR REFERENCE ONLY — no longer the default; see
# the note in _load_system_prompt and activity 003 Part 3.
SYS_PROMPT_V1 = (
    "Place all your step-by-step reasoning between <think> and </think> tags. "
    "After </think>, give the final answer."
)


def _load_system_prompt(path: str | None) -> str:
    """Read the system prompt from ``path``; **no system prompt** when unset.

    v1 defaulted to :data:`SYS_PROMPT_V1`. The P2 calibration probe (activity
    003) measured both on 100 rollouts each: with that prompt, format compliance
    94% and 6 rollouts emitted a **duplicated** ``</think>``; without any system
    message, 100% compliance and +8 points of accuracy. Qwen3 thinks natively —
    naming the tags in the prompt is what makes it re-emit one. Pass
    ``--system_prompt_file`` to supply one deliberately.
    """
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


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
                  prefill_think: bool, enable_thinking: bool = True) -> str:
    """Build the prompt via the tokenizer's chat template.

    ``enable_thinking`` is forwarded to the template (P2 fix; ROADMAP rule 4:
    every hybrid-Qwen3 template call passes it — rollout, scoring and eval
    alike). Templates that do not define the variable ignore it, so the script
    stays model-agnostic.

    ``prefill_think`` is a **Gemma-era** switch and now defaults to False. On
    Qwen3 with ``enable_thinking=True`` the rendered prompt ends at the
    assistant header and the model emits ``<think>`` itself; appending
    ``<think>\\n`` here would move the opener into the prompt, so every
    completion would parse as ``missing_think_open`` in
    :mod:`whetstone.segments` unless callers also set
    ``think_opened_by_prompt=True``. Leave it off for v2.
    """
    messages = [{"role": "user", "content": user_text}]
    if sys_prompt:
        messages.insert(0, {"role": "system", "content": sys_prompt})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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
                    help="Append '<think>\\n' to the chat-template prompt if the "
                         "template does not already include it. Gemma-era switch; "
                         "OFF by default for v2 — on Qwen3 it moves the <think> "
                         "opener into the prompt and every completion then parses "
                         "as missing_think_open.")
    ap.add_argument("--no_prefill_think", dest="prefill_think", action="store_false")
    ap.set_defaults(prefill_think=False)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--no_enable_thinking", dest="enable_thinking",
                    action="store_false",
                    help="Hybrid-Qwen3 templates only; leave ON (ROADMAP rule 4).")
    ap.set_defaults(enable_thinking=True)
    ap.add_argument("--no_system_prompt", action="store_true",
                    help="Send no system message at all (Qwen3 needs no <think> "
                         "instruction — it thinks natively).")
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

    sys_prompt = "" if args.no_system_prompt else _load_system_prompt(args.system_prompt_file)

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
                # level/source ride along so downstream yield tables are a
                # groupby on the harvest file itself, with no join back to the
                # pool (packet P3: "Log yield per level band").
                problems.append({"uid": uid, "k": k, "prompt": prompt, "gold": gold,
                                 "level": r.get("level"), "source": r.get("source")})

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
        prompts = [_build_prompt(tokenizer, sys_prompt, p["prompt"],
                                 args.prefill_think, args.enable_thinking)
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
                "level": p["level"],
                "source": p["source"],
                "completion": text,
                "completion_token_ids": list(out.outputs[0].token_ids),
                "n_tokens": len(out.outputs[0].token_ids),
                "finish_reason": out.outputs[0].finish_reason,
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
