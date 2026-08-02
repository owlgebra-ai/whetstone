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

Two execution paths, same records out:

* **server/client (default when ``--server`` is set; preferred)** — one HTTP
  request per rollout against a resident ``vllm serve`` instance, with a bounded
  in-flight window. The offline path below issues ``llm.generate(batch)``, and
  each such call is a **barrier**: the batch does not return until its slowest
  member finishes, so one 32k-token rollout idles the other 31 slots. Streaming
  every rollout as an independent request lets vLLM's continuous batching refill
  a slot the moment it frees, which is where the utilisation comes from. The
  server also outlives the client, so a resume costs no model load::

      source .venv/bin/activate && \\
      vllm serve Qwen/Qwen3-1.7B --port 8000 --max-model-len 34816 \\
          --max-num-seqs 32 --gpu-memory-utilization 0.90

  ``/v1/completions`` is used, never ``/v1/chat/completions``: the prompt is
  rendered here by ``apply_chat_template`` so the blindness contract and the
  ``enable_thinking`` flag stay under this script's control and byte-identical
  to the offline path.

* **offline (no ``--server``)** — in-process ``vllm.LLM``. Kept for boxes with
  no server up and for reproducing pre-P3 runs.

Sampling seeds are derived per rollout as ``sha1(uid:k:seed)`` when ``--seed``
is given (both paths). A single shared seed would make the K candidates for one
problem byte-identical; deriving per rollout keeps them independent *and* makes
a resumed run regenerate exactly what the interrupted one would have.

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


def _rollout_seed(uid: str, k: int, base_seed: int | None) -> int | None:
    """Per-rollout seed, or None to let the engine sample freely.

    One shared seed across a K-sample group makes every candidate for a problem
    identical — the group collapses to K copies of one trace and the harvest
    yields nothing the verifier can stratify. Deriving from (uid, k) keeps the
    candidates independent while making the run reproducible, including across
    a resume: rollout (uid, k) gets the same seed whenever it is regenerated.
    """
    if base_seed is None:
        return None
    h = hashlib.sha1(f"{uid}:{k}:{base_seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)          # 32-bit, inside every engine's seed range


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


def _repair_tail(output: str) -> int:
    """Truncate a trailing partial line, returning the bytes dropped.

    A kill mid-``write()`` leaves the file ending in half a JSON object. Merely
    *skipping* that line on read is not enough: the next append lands on the
    same line, fusing garbage with a good record, and the good record is then
    lost silently on every future pass. Since the harvest is expected to be
    interrupted (packet P3 Part 1), the file is repaired before it is reopened
    for append — the only place that can be done safely.
    """
    if not os.path.exists(output) or os.path.getsize(output) == 0:
        return 0
    with open(output, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return 0
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0
        # Walk back to the last newline; everything after it is a torn write.
        chunk = 65536
        pos = size
        while pos > 0:
            start = max(0, pos - chunk)
            f.seek(start)
            buf = f.read(pos - start)
            nl = buf.rfind(b"\n")
            if nl != -1:
                keep = start + nl + 1
                f.truncate(keep)
                return size - keep
            pos = start
        f.truncate(0)                      # no newline at all: whole file torn
        return size


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
                # Torn write that _repair_tail could not reach (e.g. a file
                # copied mid-flight): skip, do not reject the whole file.
                continue
    return seen


def _checkpoint(output: str, out_f, payload: dict) -> None:
    """Flush + fsync the corpus, then rewrite the progress sidecar.

    Order matters: the sidecar must never advertise records that are not yet on
    disk, so the data is fsynced first. Line buffering already survives a
    process kill; the fsync is what survives the box going down.
    """
    out_f.flush()
    os.fsync(out_f.fileno())
    tmp = f"{output}.progress.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, f"{output}.progress.json")   # atomic


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
    ap.add_argument("--server", default=None,
                    help="Base URL of a running `vllm serve` instance, e.g. "
                         "http://127.0.0.1:8000/v1 . When set, rollouts are "
                         "issued as independent /v1/completions requests "
                         "instead of batched in-process (no per-batch barrier).")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="Max in-flight requests in --server mode. Oversubscribe "
                         "the server's --max-num-seqs slightly so a finished "
                         "sequence is replaced immediately.")
    ap.add_argument("--request_timeout", type=int, default=5400,
                    help="Per-request socket read timeout, seconds. A 32k-token "
                         "rollout under load takes minutes, not seconds.")
    ap.add_argument("--flush_every", type=int, default=20,
                    help="fsync-visible progress granularity in --server mode.")
    ap.add_argument(
        "--data_root",
        default=None,
        help="If set, prepended to sys.path so whetstone.verify resolves",
    )
    return ap.parse_args(argv)


def _record(p: dict, text: str, token_ids, finish_reason, args, model_name: str) -> dict:
    return {
        "_uid": p["uid"],
        "candidate_idx": p["k"],
        "prompt": p["prompt"],
        "ground_truth": p["gold"],
        "level": p["level"],
        "source": p["source"],
        "completion": text,
        "completion_token_ids": list(token_ids) if token_ids else [],
        "n_tokens": len(token_ids) if token_ids else 0,
        "finish_reason": finish_reason,
        "model": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": p["seed"],
        "worker_id": args.worker_id,
    }


def _run_server(args, problems, tokenizer, sys_prompt, out_f, model_name) -> None:
    """Issue one /v1/completions request per rollout, ``--concurrency`` in flight.

    The window is what makes this faster than the offline path: a slot freed by
    a short rollout is refilled at once instead of waiting for the batch's
    longest member. Results are written in completion order (the file is a bag
    keyed by (_uid, candidate_idx), never positional).
    """
    import asyncio
    import time

    import aiohttp

    url = args.server.rstrip("/") + "/completions"
    written = [0]
    failed: list[tuple[str, int, str]] = []
    total = len(problems)
    t0 = time.time()
    fail_path = f"{args.output}.failed.jsonl"
    fail_f = open(fail_path, "a", buffering=1)

    def _progress(done: bool = False) -> dict:
        el = time.time() - t0
        return {
            "output": args.output,
            "written_this_run": written[0],
            "failed_this_run": len(failed),
            "queued_this_run": total,
            "remaining_this_run": total - written[0] - len(failed),
            "elapsed_s": round(el, 1),
            "rollouts_per_min": round(60 * written[0] / el, 2) if el > 0 else 0.0,
            "done": done,
        }

    async def one(session, sem, p):
        body = {
            "model": args.model,
            "prompt": _build_prompt(tokenizer, sys_prompt, p["prompt"],
                                    args.prefill_think, args.enable_thinking),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": 1,
            "stream": False,
            "return_token_ids": True,
        }
        if args.top_k > 0:
            body["top_k"] = args.top_k
        if p["seed"] is not None:
            body["seed"] = p["seed"]

        async with sem:
            last = ""
            for attempt in range(4):
                try:
                    async with session.post(url, json=body) as resp:
                        if resp.status != 200:
                            last = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                            # 4xx is a request bug — retrying cannot fix it.
                            if 400 <= resp.status < 500:
                                break
                            raise RuntimeError(last)
                        data = await resp.json()
                    ch = data["choices"][0]
                    text = ch["text"]
                    if args.prefill_think and not text.lstrip().startswith("<think>"):
                        text = "<think>\n" + text
                    return _record(p, text, ch.get("token_ids") or [],
                                   ch.get("finish_reason"), args, model_name)
                except Exception as exc:                       # noqa: BLE001
                    last = last or f"{type(exc).__name__}: {exc}"
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)
            # Recorded, not silently dropped: the ledger is the audit trail for
            # "every rollout this run attempted". It is deliberately NOT the
            # corpus file — a failure written there would make resume treat the
            # rollout as done and never retry it.
            failed.append((p["uid"], p["k"], last))
            fail_f.write(json.dumps({
                "_uid": p["uid"], "candidate_idx": p["k"], "level": p["level"],
                "error": last, "seed": p["seed"],
            }) + "\n")
            return None

    async def run():
        sem = asyncio.Semaphore(args.concurrency)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30,
                                        sock_read=args.request_timeout)
        conn = aiohttp.TCPConnector(limit=args.concurrency + 8)
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            tasks = [asyncio.create_task(one(session, sem, p)) for p in problems]
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                if rec is None:
                    continue
                out_f.write(json.dumps(rec) + "\n")
                written[0] += 1
                if written[0] % args.flush_every == 0:
                    _checkpoint(args.output, out_f, _progress())
                    pr = _progress()
                    print(f"[gen] {written[0]}/{total} written "
                          f"({len(failed)} failed, {pr['rollouts_per_min']}/min)",
                          flush=True)

    try:
        asyncio.run(run())
    finally:
        # Checkpoint on the way out of *any* exit path — Ctrl-C, server death,
        # unhandled error — so the resume point is never worse than the last
        # completed rollout.
        _checkpoint(args.output, out_f, _progress(done=True))
        fail_f.close()

    print(f"[gen] {written[0]}/{total} written, {len(failed)} failed", flush=True)
    for uid, k, err in failed[:10]:
        print(f"[fail] {uid} k={k}: {err}", flush=True)
    if failed:
        print(f"[harvest] {len(failed)} rollouts failed, logged to {fail_path} — "
              "re-run to retry them (resume skips everything already written)",
              flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.data_root:
        sys.path.insert(0, args.data_root)

    from transformers import AutoTokenizer

    dropped = _repair_tail(args.output)
    if dropped:
        print(f"[resume] repaired torn tail: dropped {dropped} B of a partial "
              "record from the previous run", flush=True)
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
                                 "level": r.get("level"), "source": r.get("source"),
                                 "seed": _rollout_seed(uid, k, args.seed)})

    if not problems:
        print("[harvest] nothing to do", flush=True)
        return
    print(f"[load] {len(problems)} rollouts to generate", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)
    model_name = os.path.basename(os.path.normpath(args.model))

    if args.server:
        print(f"[harvest] server mode: {args.server}, "
              f"{args.concurrency} in flight", flush=True)
        try:
            _run_server(args, problems, tokenizer, sys_prompt, out_f, model_name)
        finally:
            out_f.close()
        print("[harvest] done", flush=True)
        return

    from vllm import LLM, SamplingParams

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

    def _sp(p):
        sp_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=p["seed"],
        )
        if args.top_k > 0:
            sp_kwargs["top_k"] = args.top_k
        return SamplingParams(**sp_kwargs)

    batch = max(1, args.batch)

    for i in range(0, len(problems), batch):
        chunk = problems[i : i + batch]
        prompts = [_build_prompt(tokenizer, sys_prompt, p["prompt"],
                                 args.prefill_think, args.enable_thinking)
                   for p in chunk]
        outs = llm.generate(prompts, [_sp(p) for p in chunk])
        for p, out in zip(chunk, outs):
            text = out.outputs[0].text
            # If we prefilled "<think>\n", the completion does not start with it;
            # prepend so extract_answer() in whetstone.verify sees the same shape
            # it sees at eval time.
            if args.prefill_think and not text.lstrip().startswith("<think>"):
                text = "<think>\n" + text
            out_f.write(json.dumps(_record(
                p, text, out.outputs[0].token_ids,
                out.outputs[0].finish_reason, args, model_name)) + "\n")
        _checkpoint(args.output, out_f, {
            "output": args.output,
            "written_this_run": i + len(chunk),
            "queued_this_run": len(problems),
            "remaining_this_run": len(problems) - (i + len(chunk)),
            "mode": "offline",
            "done": (i + len(chunk)) >= len(problems),
        })
        print(f"[gen] {i + len(chunk)}/{len(problems)}", flush=True)

    out_f.close()
    print("[harvest] done", flush=True)


if __name__ == "__main__":
    main()
