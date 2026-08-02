"""Durable append-only JSONL + a bounded-concurrency vLLM completions client.

Every v2 stage that streams model output to disk needs the same two things, and
both are easy to get subtly wrong:

**Durability.** Runs here are hours long on a box that is also the trainer, so
they *will* be interrupted (packet P3 Part 1 mandates a deliberate kill test).
Append-only JSONL keyed by ``(_uid, candidate_idx)`` makes resume a set
difference — but only if a torn trailing line is repaired *before* the file is
reopened for append. Merely skipping an unparseable line on read is not enough:
the next append lands on that same line, fusing garbage with a good record and
silently losing the good one on every future pass. :func:`repair_tail` is the
fix and must run before the append handle is opened.

**Throughput.** The offline ``llm.generate(batch)`` API is a barrier: the call
does not return until the batch's slowest member finishes, so one 32k-token
rollout idles every other slot in the batch. Issuing one request per unit of
work against a resident ``vllm serve`` lets continuous batching refill a slot
the moment it frees. :func:`run_completions` is that client — a bounded
in-flight window over ``/v1/completions``, delivering results in completion
order so the caller can write and checkpoint incrementally.

``/v1/completions`` is used rather than ``/v1/chat/completions`` throughout:
callers render prompts with ``apply_chat_template`` themselves, which is what
keeps the blindness contract (v1 §2) and the ``enable_thinking`` flag
(ROADMAP rule 4) under the calling script's control and auditable.

No torch/vLLM import here — this module must stay importable on a laptop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "repair_tail",
    "scan_seen",
    "checkpoint",
    "CompletionResult",
    "run_completions",
]


# ---------------------------------------------------------------------------
# durable append-only JSONL
# ---------------------------------------------------------------------------

def repair_tail(path: str) -> int:
    """Truncate a trailing partial line; return bytes dropped.

    Call this *before* opening ``path`` for append. See the module docstring for
    why skipping the bad line on read is not a substitute.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0
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
        f.truncate(0)                     # no newline anywhere: whole file torn
        return size


def scan_seen(path: str, key_fields: Sequence[str] = ("_uid", "candidate_idx")) -> set:
    """Set of composite keys already present in ``path`` (missing file -> empty)."""
    seen: set = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn write repair_tail could not reach
            seen.add(tuple(r.get(k) for k in key_fields))
    return seen


def checkpoint(path: str, fh, payload: dict) -> None:
    """fsync the corpus, then atomically replace ``<path>.progress.json``.

    Order is load-bearing: the sidecar must never advertise records that are not
    yet on disk, so the data is fsynced first and the sidecar is swapped in with
    :func:`os.replace`. Line buffering alone survives a process kill; the fsync
    is what survives the box going down.
    """
    fh.flush()
    os.fsync(fh.fileno())
    tmp = f"{path}.progress.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, f"{path}.progress.json")


# ---------------------------------------------------------------------------
# bounded-concurrency completions client
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    """One finished request. ``error`` is set iff the request never succeeded."""

    index: int                       # position in the caller's `bodies` list
    text: str = ""
    token_ids: list = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_completions(
    server: str,
    bodies: Sequence[dict],
    *,
    on_result: Callable[[CompletionResult], Any],
    concurrency: int = 64,
    timeout_s: int = 5400,
    retries: int = 4,
) -> None:
    """POST each body to ``{server}/completions``, ``concurrency`` in flight.

    ``on_result`` is invoked from the event loop as each request settles, in
    **completion order** — callers must key their output by id, never by
    position. It is called for failures too (``result.ok is False``) so nothing
    is silently dropped; a caller that wants failures retried on the next run
    should keep them out of its resume file.

    Retries are bounded and exponential, and 4xx responses are not retried —
    those are request bugs, and hammering the server cannot fix them.
    """
    import asyncio

    import aiohttp

    url = server.rstrip("/") + "/completions"

    async def one(session, sem, i, body) -> CompletionResult:
        async with sem:
            last = ""
            for attempt in range(retries):
                try:
                    async with session.post(url, json=body) as resp:
                        if resp.status != 200:
                            last = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                            if 400 <= resp.status < 500:
                                break
                            raise RuntimeError(last)
                        data = await resp.json()
                    ch = data["choices"][0]
                    return CompletionResult(
                        index=i,
                        text=ch["text"],
                        token_ids=list(ch.get("token_ids") or []),
                        finish_reason=ch.get("finish_reason"),
                    )
                except Exception as exc:                       # noqa: BLE001
                    last = last or f"{type(exc).__name__}: {exc}"
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
            return CompletionResult(index=i, error=last or "unknown error")

    async def run() -> None:
        sem = asyncio.Semaphore(concurrency)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30,
                                        sock_read=timeout_s)
        conn = aiohttp.TCPConnector(limit=concurrency + 8)
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            tasks = [asyncio.create_task(one(session, sem, i, b))
                     for i, b in enumerate(bodies)]
            for fut in asyncio.as_completed(tasks):
                on_result(await fut)

    asyncio.run(run())
