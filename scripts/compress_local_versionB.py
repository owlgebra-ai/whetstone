"""Stage 2 — Chunkwise Self-Compression (Version B).

Rewrites each verified-correct trace into the compact register using a
single-turn prefill trick: at depth k, the model is shown ORIGINAL CHUNKS
1..k and the COMPACT versions of chunks 1..k-1, and completes only COMPACT
CHUNK k. Depth-batching across problems gives vLLM throughput.

The compressor is the SAME base model that produced the harvest. No external
teacher (central-model principle, §3).

Invariants (§3.4):
  * Paragraph splitter with per-chunk token cap, merge enforcing max_chunks.
  * Prefill assembled via apply_chat_template(enable_thinking=False).
  * clean_compact_lines strips trailing chunk markers.
  * Periodic checkpoint every N depths.
  * Inviolable: only compress traces already verified correct (§3.7).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

MAX_CHUNK_TOKENS_DEFAULT = 800

SYSTEM_PROMPT = (
    "You are a compression engine. Rewrite verbose chain-of-thought reasoning "
    "into a compact symbolic register.\n\n"
    "Rules:\n"
    "- Preserve every load-bearing fact, variable, equation, constant, case-split, "
    "and derivation step.\n"
    "- Use terse symbolic notation: =, ⇒, →, ✓, ⚠, ?.\n"
    "- One to a few lines per chunk. Use ';' to separate claims on the same line.\n"
    "- No verbose connective prose. No caveman style ('compute. add. done.').\n"
    "- Do NOT invent content not in the original.\n"
    "- If the original contains self-correction ('wait', 'actually', 'no'), "
    "preserve it with ⚠ or ? markers.\n"
)

STOP_STRINGS = ["\nCOMPACT CHUNK", "\nORIGINAL CHUNK", "<|im_end|>"]


def _approx_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def chunk_thinking(thinking: str, max_chunk_tokens: int, tokenizer=None) -> list[str]:
    """Split thinking into paragraphs subject to a per-chunk token cap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", thinking) if p.strip()]
    chunks: list[str] = []
    cur = ""
    cur_toks = 0
    for p in paras:
        pt = _approx_tokens(p, tokenizer)
        if pt > max_chunk_tokens:
            for sentence in re.split(r"(?<=[.!?])\s+", p):
                st = _approx_tokens(sentence, tokenizer)
                if cur and cur_toks + st > max_chunk_tokens:
                    chunks.append(cur)
                    cur, cur_toks = sentence, st
                else:
                    cur = f"{cur} {sentence}".strip() if cur else sentence
                    cur_toks = _approx_tokens(cur, tokenizer)
        elif cur and cur_toks + pt > max_chunk_tokens:
            chunks.append(cur)
            cur, cur_toks = p, pt
        else:
            cur = f"{cur}\n\n{p}".strip() if cur else p
            cur_toks = _approx_tokens(cur, tokenizer)
    if cur:
        chunks.append(cur)
    return chunks


def merge_and_cap(chunks: list[str], max_chunks: int) -> list[str]:
    """Merge adjacent chunks to fit max_chunks when there are too many."""
    if len(chunks) <= max_chunks:
        return chunks
    buckets: list[list[str]] = [[] for _ in range(max_chunks)]
    for i, c in enumerate(chunks):
        buckets[i % max_chunks].append(c)
    return ["\n\n".join(b) for b in buckets if b]


def extract_thinking(completion: str) -> str:
    if "<think>" in completion and "</think>" in completion:
        return completion.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "<think>" in completion:
        return completion.split("<think>", 1)[1].strip()
    return completion.strip()


def clean_compact_lines(text: str) -> str:
    """Strip trailing '\nCOMPACT CHUNK N+1' / '\nORIGINAL CHUNK N+1' markers,
    then join multi-line compacts with ' | '."""
    text = re.split(r"\n(?:COMPACT CHUNK|ORIGINAL CHUNK)\s*\d+", text)[0]
    text = text.rstrip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines) if lines else ""


def build_prefill_prompt(tokenizer, problem: str, originals: list[str],
                         prev_compacts: list[str]) -> str:
    """Assemble the full prompt with assistant prefill through chunk k-1.
    Model completes only COMPACT CHUNK k."""
    user_lines = [f"PROBLEM: {problem}"]
    for i, o in enumerate(originals):
        user_lines.append(f"ORIGINAL CHUNK {i + 1}: {o}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefill_lines = [f"COMPACT CHUNK {i + 1}: {c}" for i, c in enumerate(prev_compacts)]
    k = len(prev_compacts) + 1
    prefill_lines.append(f"COMPACT CHUNK {k}: ")
    return prompt_text + "\n".join(prefill_lines)


def _scan_seen(output: str) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    if not os.path.exists(output):
        return seen
    with open(output) as f:
        for line in f:
            try:
                r = json.loads(line)
                seen.add((r["_uid"], r.get("src_candidate_idx", 0)))
            except json.JSONDecodeError:
                continue
    return seen


def _write_checkpoint(output: str, problems: list[dict]) -> int:
    done = 0
    tmp = output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(tmp, "w") as f:
        for p in problems:
            compacts = p["compacts_per_chunk"]
            if not compacts or any(c is None for c in compacts):
                continue
            compact = "; ".join(compacts)
            rec = {
                "_uid": p["uid"],
                "src_candidate_idx": p["src_candidate_idx"],
                "compress_idx": 0,
                "prompt": p["prompt"],
                "ground_truth": p["ground_truth"],
                "thinking_original": p["thinking_original"],
                "thinking_original_tokens": _approx_tokens(p["thinking_original"]),
                "n_chunks": len(p["chunks"]),
                "chunks": p["chunks"],
                "compacts_per_chunk": compacts,
                "compact": compact,
                "compact_tokens": _approx_tokens(compact),
                "compression_ratio": (
                    _approx_tokens(compact) / max(1, _approx_tokens(p["thinking_original"]))
                ),
            }
            f.write(json.dumps(rec) + "\n")
            done += 1
    os.replace(tmp, output)
    return done


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE Stage 2 chunkwise compression")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--max-tokens-per-compact", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--max-chunk-tokens", type=int, default=MAX_CHUNK_TOKENS_DEFAULT)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--first-candidate-only", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=3)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.temperature != 0.0:
        # §3.3: do not sample. Chunkwise sampling is high-variance; the Δlogp
        # gate downstream would reject the noise anyway.
        print("[compress] WARNING: temperature != 0; chunkwise sampling is unstable",
              file=sys.stderr)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seen = _scan_seen(args.output)
    print(f"[resume] {len(seen)} problems already compressed", flush=True)

    problems: list[dict] = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            cidx = r.get("candidate_idx", 0)
            if args.first_candidate_only and cidx > 0:
                continue
            if (r["_uid"], cidx) in seen:
                continue
            thinking = extract_thinking(r.get("completion", ""))
            if not thinking:
                continue
            problems.append({
                "uid": r["_uid"],
                "src_candidate_idx": cidx,
                "prompt": r.get("prompt", ""),
                "ground_truth": r.get("ground_truth", ""),
                "thinking_original": thinking,
            })

    if not problems:
        print("[compress] nothing to do", flush=True)
        return
    print(f"[load] {len(problems)} problems to compress", flush=True)

    for p in problems:
        chunks = chunk_thinking(p["thinking_original"], args.max_chunk_tokens, tokenizer)
        p["chunks"] = merge_and_cap(chunks, args.max_chunks)
        p["n_chunks"] = len(p["chunks"])
        p["compacts_per_chunk"] = [None] * p["n_chunks"]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_compact,
        stop=STOP_STRINGS,
    )

    max_depth = max(p["n_chunks"] for p in problems)
    last_save = 0
    for depth in range(1, max_depth + 1):
        active = [p for p in problems
                  if p["n_chunks"] >= depth and p["compacts_per_chunk"][depth - 1] is None]
        if not active:
            continue
        prompts = [
            build_prefill_prompt(tokenizer, p["prompt"], p["chunks"][:depth],
                                 p["compacts_per_chunk"][: depth - 1])
            for p in active
        ]
        outs = llm.generate(prompts, sp)
        for p, out in zip(active, outs):
            cleaned = clean_compact_lines(out.outputs[0].text)
            p["compacts_per_chunk"][depth - 1] = cleaned or "[empty]"
        print(f"[depth {depth}/{max_depth}] compressed {len(active)} chunks", flush=True)

        if depth - last_save >= args.checkpoint_every or depth == max_depth:
            n = _write_checkpoint(args.output, problems)
            last_save = depth
            print(f"[checkpoint] depth={depth} wrote {n} completed records", flush=True)

    n = _write_checkpoint(args.output, problems)
    print(f"[compress] done, {n} records", flush=True)


if __name__ == "__main__":
    main()
