"""Prompted chunkwise compression into the compact register (v1 §3, v2 P3 Part 2).

Rewrites each verified-correct trace's **think segment** into the compact
register using a single-turn prefill trick: at depth k, the model is shown
ORIGINAL CHUNKS 1..k and the COMPACT versions of chunks 1..k-1, and completes
only COMPACT CHUNK k. Depth-batching across problems gives vLLM throughput.

The compressor is the SAME base model that produced the harvest. No external
teacher (central-model principle, v1 §3).

v2 changes (packet P3a, "Pinned compression prompt")
----------------------------------------------------
v1's hardcoded ``SYSTEM_PROMPT`` prescribed the symbolic notation by name
(``=, ⇒, →, ✓, ⚠, ?``) and explicitly banned caveman style. That is
disqualifying for a register comparison — it biases arm A and sabotages arm B —
and it named ``⚠``, which activity 003 rejected on token cost. It is replaced
by a **card-parametric** scaffold: a neutral instruction block that is
byte-identical across arms, plus ``--card`` pasted verbatim. The card is the
only authority on notation and style.

``enable_thinking=False`` on the compressor call is **deliberate and correct**
(v1 §3.1): this is a prefill-trick text rewrite, so we want the compact chunk
emitted directly rather than a thinking preamble. It is the one standing
exception to the ROADMAP's ``enable_thinking=True`` rule, which covers
rollouts, scoring and eval — compression is none of those.

Invariants (v1 §3.4, §3.7)
-------------------------
  * Paragraph splitter with per-chunk token cap, merge enforcing max_chunks.
  * Prefill assembled via apply_chat_template(enable_thinking=False).
  * Periodic checkpoint every N depths.
  * Only compress traces already verified correct.
  * **The answer segment is copied through untouched** and ``verify_response``
    is re-asserted on every emitted record — a failure means the compressor
    leaked past ``</think>``, which is a hard bug (P3 gotcha 3).
  * **Line structure survives.** Both register cards make the newline the step
    boundary, so chunk compacts are joined with ``\\n`` and multi-line compacts
    are kept multi-line. (v1 flattened both with ``" | "`` / ``"; "``, which
    would destroy the register being measured.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.verify import verify_response

MAX_CHUNK_TOKENS_DEFAULT = 800

# --- the pinned neutral scaffold (packet P3a) ------------------------------
# Byte-identical across arms. Anything that differs between two compression
# runs other than the card block contaminates the comparison.
SCAFFOLD = """You are a compression engine. Rewrite verbose chain-of-thought reasoning into
the compact register defined by the REGISTER CARD below. The card is the only
authority on notation and style.

Rules:
- Preserve every load-bearing fact, variable, equation, constant, case-split,
  and derivation step. Never fuse steps; never drop an intermediate value.
- Do NOT invent content not in the original.
- If the original contains self-correction ("wait", "actually", "no"),
  preserve the correction and its resolution in register form.
- One to a few lines per chunk; follow the card's step-marker convention.
{mode_rule}
REGISTER CARD:
{card}"""

# Chunkwise only. Without it the cumulative-context loop is a repetition
# attractor: the model locks onto one whole-problem summary and re-emits it at
# every subsequent depth (activity 004). Meaningless in one-shot mode, where
# there are no chunks — hence mode-selected rather than always present.
CHUNK_RULE = """- Rewrite ONLY the LAST ORIGINAL CHUNK shown. The earlier original and compact
  chunks are context: do not restate, re-derive or re-summarize what an earlier
  COMPACT CHUNK already covers, and continue its step numbering rather than
  restarting at 1.
"""

# Card sections that are *not* register spec and must not reach the compressor.
# Dropping them is what keeps the two prompts differing only in notation:
#   - the HTML provenance header (A's is ~26 lines and names rejected symbols;
#     B's is ~14 and describes the bake-off) — pure asymmetric noise;
#   - "Structural whitelist for R" — a P4 token-set artifact, not notation
#     (A's carries four lines of raw Qwen3 token ids);
#   - "Self-check before flipping" — a human ratification checklist, A only;
#   - "Exemplars 3–8" — an A-only ⟨PENDING⟩ stub.
# Every drop is recorded in the sidecar together with the rendered prompt, so
# what the model actually saw is auditable.
DROP_HEADINGS_DEFAULT = (
    "Structural whitelist",
    "Self-check before flipping",
    "Exemplars 3–8",
)
_META_LINE = re.compile(r"^\*\*(Status|Author|Date|Target model):\*\*.*\n?", re.M)

STOP_STRINGS = ["\nCOMPACT CHUNK", "\nORIGINAL CHUNK", "<|im_end|>"]


# ---------------------------------------------------------------------------
# card rendering + provenance
# ---------------------------------------------------------------------------

def render_card(text: str, drop_headings=DROP_HEADINGS_DEFAULT) -> str:
    """Strip non-notation scaffolding from a register card."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Split on top-level (## / ###) headings, keep sections whose heading does
    # not match a drop pattern.
    parts = re.split(r"(?m)^(#{2,3} .*)$", text)
    kept = [parts[0]]
    for heading, body in zip(parts[1::2], parts[2::2]):
        if any(d in heading for d in drop_headings):
            continue
        kept.append(heading + body)
    out = "".join(kept)
    out = _META_LINE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _git_sha(path: str) -> str:
    """git blob sha of the card file, so a card edit is detectable from output."""
    try:
        return subprocess.run(["git", "hash-object", path], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def build_system_prompt(card_text: str, mode: str = "oneshot") -> str:
    return SCAFFOLD.format(card=card_text,
                           mode_rule=CHUNK_RULE if mode == "chunkwise" else "")


# ---------------------------------------------------------------------------
# chunking (v1 §3.4, unchanged)
# ---------------------------------------------------------------------------

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
    """Merge adjacent chunks to fit max_chunks when there are too many.

    NOTE the round-robin bucketing is v1's and is *order-preserving only within
    a bucket* — kept unchanged so v1 comparisons stay valid, but it is why
    chunk alignment is hand-checked on 2 traces before every bulk run.
    """
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


def extract_answer(completion: str) -> str:
    if "</think>" in completion:
        return completion.split("</think>", 1)[1].lstrip("\n")
    return ""


def clean_compact_lines(text: str) -> str:
    """Strip trailing chunk markers; keep the line structure intact.

    v1 collapsed multi-line output to ``" | "``-joined text. Both v2 register
    cards define the newline as the step boundary (card §1.2), so flattening
    here would measure a register neither card specifies.
    """
    text = re.split(r"\n(?:COMPACT CHUNK|ORIGINAL CHUNK)\s*\d+", text)[0]
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def clean_oneshot(text: str) -> str:
    """Clean a whole-trace compact rewrite.

    Both cards present their exemplars as 4-space-indented markdown code
    blocks, and the model faithfully copies that indentation into its output.
    It is a card-formatting artifact rather than register notation — identical
    in both arms — so it is dedented here rather than being paid for in tokens
    on every line.
    """
    text = re.sub(r"^\s*```[a-zA-Z]*\n|```\s*$", "", text.strip())
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return ""
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    return "\n".join(ln[indent:] for ln in lines)


def build_oneshot_prompt(tokenizer, system_prompt: str, problem: str,
                         thinking: str) -> str:
    """Whole think segment in, whole compact register out.

    The chunkwise loop below is v1 §3.4's machinery and it **fails on Qwen3-1.7B
    with a notation-neutral prompt** (activity 004): cumulative ORIGINAL CHUNKS
    plus previously-emitted COMPACT CHUNKS is a repetition attractor — the model
    locks onto one whole-problem summary and copies it at every subsequent
    depth — and register-marker density comes out ~10x below the cards' own
    exemplars. One-shot reproduces the card's exemplar style directly.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"PROBLEM: {problem}\n\nVERBOSE REASONING:\n{thinking}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_prefill_prompt(tokenizer, system_prompt: str, problem: str,
                         originals: list[str], prev_compacts: list[str]) -> str:
    """Assemble the full prompt with assistant prefill through chunk k-1.
    Model completes only COMPACT CHUNK k."""
    user_lines = [f"PROBLEM: {problem}"]
    for i, o in enumerate(originals):
        user_lines.append(f"ORIGINAL CHUNK {i + 1}: {o}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefill_lines = [f"COMPACT CHUNK {i + 1}: {c}" for i, c in enumerate(prev_compacts)]
    k = len(prev_compacts) + 1
    prefill_lines.append(f"COMPACT CHUNK {k}: ")
    return prompt_text + "\n".join(prefill_lines)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

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


def _build_record(p: dict, tokenizer, meta: dict) -> dict:
    compacts = p["compacts_per_chunk"]
    compact_think = "\n".join(c for c in compacts if c)
    answer = p["answer"]
    completion = f"<think>\n{compact_think}\n</think>\n\n{answer}"
    return {
        "_uid": p["uid"],
        "src_candidate_idx": p["src_candidate_idx"],
        "level": p.get("level"),
        "prompt": p["prompt"],
        "ground_truth": p["ground_truth"],
        "verbose_think": p["thinking_original"],
        "verbose_think_tokens": _approx_tokens(p["thinking_original"], tokenizer),
        "n_chunks": len(p["chunks"]),
        "chunks": p["chunks"],
        "compacts_per_chunk": compacts,
        "compact_think": compact_think,
        "compact_think_tokens": _approx_tokens(compact_think, tokenizer),
        "answer": answer,
        # `completion` is the field entropy_audit.py --traces and the style-tax
        # scorer consume: a full, re-parseable rollout in compact register.
        "completion": completion,
        "compression_ratio": (
            _approx_tokens(compact_think, tokenizer)
            / max(1, _approx_tokens(p["thinking_original"], tokenizer))
        ),
        "verify_ok": bool(verify_response(completion, p["ground_truth"])),
        "arm": meta["arm"],
        "card_path": meta["card_path"],
        "card_git_sha": meta["card_git_sha"],
        "rendered_prompt_sha1": meta["rendered_prompt_sha1"],
    }


def _write_checkpoint(output: str, problems: list[dict], tokenizer, meta: dict) -> int:
    done = 0
    tmp = output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(tmp, "w") as f:
        for p in problems:
            compacts = p["compacts_per_chunk"]
            if not compacts or any(c is None for c in compacts):
                continue
            f.write(json.dumps(_build_record(p, tokenizer, meta)) + "\n")
            done += 1
    os.replace(tmp, output)
    return done


def _finalize(args, meta: dict, n: int) -> None:
    """Answer-segment integrity (P3 gotcha 3) + provenance sidecar.

    The answer text was copied through untouched, so every record must still
    verify exactly as its source trace did. A failure means the compressor
    leaked past ``</think>`` — a hard bug, not a quality issue.
    """
    n_bad = 0
    with open(args.output) as f:
        for line in f:
            if not json.loads(line)["verify_ok"]:
                n_bad += 1
    meta["n_records"] = n
    meta["n_verify_fail"] = n_bad
    with open(args.output + ".meta.json", "w") as f:
        json.dump({**meta, "config": vars(args)}, f, indent=1)

    print(f"[compress] done, {n} records -> {args.output}", flush=True)
    if n_bad:
        print(f"[compress] *** HARD BUG: {n_bad}/{n} records fail verify_response — "
              "the compressor leaked past </think>. Do not use this corpus.",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"[compress] verify_response: {n}/{n} pass", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE prompted register compression")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--card", required=True,
                    help="register card markdown; pasted into the pinned neutral "
                         "scaffold. The ONLY thing that may differ between arms.")
    ap.add_argument("--arm", default="",
                    help="label recorded on every record (e.g. A / B)")
    ap.add_argument("--mode", choices=["oneshot", "chunkwise"], default="oneshot",
                    help="oneshot: whole think segment -> whole compact rewrite "
                         "(default; chunkwise degenerates on Qwen3-1.7B, see "
                         "activity 004). chunkwise: v1 §3.4 prefill loop.")
    ap.add_argument("--max-tokens-oneshot", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens-per-compact", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--max-chunk-tokens", type=int, default=MAX_CHUNK_TOKENS_DEFAULT)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (debug aid)")
    ap.add_argument("--first-candidate-only", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=3)
    ap.add_argument("--dump-prompt", default="",
                    help="render the system prompt to this path and exit "
                         "(no GPU needed) — used to diff the two arms")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    card_raw = open(args.card).read()
    card_text = render_card(card_raw)
    system_prompt = build_system_prompt(card_text, args.mode)
    meta = {
        "arm": args.arm,
        "card_path": args.card,
        "card_git_sha": _git_sha(args.card),
        "card_raw_sha1": _sha1(card_raw),
        "card_rendered_sha1": _sha1(card_text),
        "scaffold_sha1": _sha1(SCAFFOLD),
        "mode": args.mode,
        "rendered_prompt_sha1": _sha1(system_prompt),
        "rendered_prompt_chars": len(system_prompt),
        "dropped_headings": list(DROP_HEADINGS_DEFAULT),
    }

    if args.dump_prompt:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_prompt)), exist_ok=True)
        with open(args.dump_prompt, "w") as f:
            f.write(system_prompt)
        print(json.dumps(meta, indent=1))
        return

    if args.temperature > 0.6:
        print("[compress] WARNING: T > 0.6; chunkwise sampling gets unstable "
              "(v1 §3.3). P3/P3a specify T=0.4.", file=sys.stderr)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seen = _scan_seen(args.output)
    print(f"[resume] {len(seen)} problems already compressed", flush=True)
    print(f"[card] {args.card} blob={meta['card_git_sha'][:12]} "
          f"rendered_prompt_sha1={meta['rendered_prompt_sha1'][:12]} "
          f"({meta['rendered_prompt_chars']} chars)", flush=True)

    problems: list[dict] = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            cidx = r.get("candidate_idx", r.get("src_candidate_idx", 0))
            if args.first_candidate_only and cidx > 0:
                continue
            if (r["_uid"], cidx) in seen:
                continue
            thinking = r.get("verbose_think") or extract_thinking(r.get("completion", ""))
            answer = r.get("answer")
            if answer is None:
                answer = extract_answer(r.get("completion", ""))
            if not thinking or not answer:
                continue
            problems.append({
                "uid": r["_uid"],
                "src_candidate_idx": cidx,
                "level": r.get("level"),
                "prompt": r.get("prompt", ""),
                "ground_truth": r.get("ground_truth", ""),
                "thinking_original": thinking,
                "answer": answer,
            })
    if args.limit:
        problems = problems[: args.limit]

    if not problems:
        print("[compress] nothing to do", flush=True)
        return
    print(f"[load] {len(problems)} problems to compress", flush=True)

    if args.mode == "chunkwise":
        for p in problems:
            chunks = chunk_thinking(p["thinking_original"], args.max_chunk_tokens,
                                    tokenizer)
            p["chunks"] = merge_and_cap(chunks, args.max_chunks)
            p["n_chunks"] = len(p["chunks"])
            p["compacts_per_chunk"] = [None] * p["n_chunks"]
    else:
        # One "chunk" = the whole think segment, so every downstream consumer
        # (metrics, inspection, checkpointing) sees the same record shape.
        for p in problems:
            p["chunks"] = [p["thinking_original"]]
            p["n_chunks"] = 1
            p["compacts_per_chunk"] = [None]

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
        seed=args.seed,
    )

    if args.mode == "oneshot":
        sp_one = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_tokens_oneshot, seed=args.seed)
        prompts = [build_oneshot_prompt(tokenizer, system_prompt, p["prompt"],
                                        p["thinking_original"]) for p in problems]
        outs = llm.generate(prompts, sp_one)
        n_cap = 0
        for p, out in zip(problems, outs):
            o = out.outputs[0]
            p["compacts_per_chunk"] = [clean_oneshot(o.text) or "[empty]"]
            n_cap += int(o.finish_reason == "length")
        n = _write_checkpoint(args.output, problems, tokenizer, meta)
        meta["cap_hit_rate"] = n_cap / max(1, len(problems))
        print(f"[oneshot] {n} records, cap-hit {meta['cap_hit_rate']:.1%} "
              f"at max_tokens={args.max_tokens_oneshot}", flush=True)
        _finalize(args, meta, n)
        return

    max_depth = max(p["n_chunks"] for p in problems)
    last_save = 0
    for depth in range(1, max_depth + 1):
        active = [p for p in problems
                  if p["n_chunks"] >= depth and p["compacts_per_chunk"][depth - 1] is None]
        if not active:
            continue
        prompts = [
            build_prefill_prompt(tokenizer, system_prompt, p["prompt"],
                                 p["chunks"][:depth], p["compacts_per_chunk"][: depth - 1])
            for p in active
        ]
        outs = llm.generate(prompts, sp)
        for p, out in zip(active, outs):
            cleaned = clean_compact_lines(out.outputs[0].text)
            p["compacts_per_chunk"][depth - 1] = cleaned or "[empty]"
        print(f"[depth {depth}/{max_depth}] compressed {len(active)} chunks", flush=True)

        if depth - last_save >= args.checkpoint_every or depth == max_depth:
            n = _write_checkpoint(args.output, problems, tokenizer, meta)
            last_save = depth
            print(f"[checkpoint] depth={depth} wrote {n} completed records", flush=True)

    n = _write_checkpoint(args.output, problems, tokenizer, meta)
    _finalize(args, meta, n)


if __name__ == "__main__":
    main()
