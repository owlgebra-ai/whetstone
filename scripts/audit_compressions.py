"""Stage 2.5 — Compression Process Audit.

Cross-family LLM auditor that catches what the §3.6 Δlogp gate misses:
hallucinated steps, dropped self-correction, verbose-prose creep, caveman-style
collapse, and trivially-restated-the-problem no-ops.

The auditor MUST be from a different model family than the compressor (the
compressor is the same base model as the student, so they share blind spots).
Default: Claude Sonnet 4.6 via cloud API.

Resume-safe: append-only output, skips already-audited (uid, src_candidate_idx)
on restart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

AUDIT_PROMPT = """You are auditing a compressed reasoning trace.

PROBLEM:
{problem}

VERBOSE ORIGINAL:
{verbose}

COMPACT (under audit):
{compact}

GROUND TRUTH ANSWER:
{gold}

Return STRICT JSON only (no prose, no markdown fences) with these criteria:

1. "faithful" (bool): every claim, equation, and intermediate result in COMPACT
   also appears (in some form) in VERBOSE ORIGINAL. No hallucinated content.
2. "preserves_load_bearing" (bool): a reader given only PROBLEM + COMPACT could
   retrace to GOLD without inventing steps. All required variables, constants,
   decompositions, and the derivation path are present.
3. "compact_register" (bool): COMPACT uses terse symbolic notation (=, ⇒, →, ✓,
   ⚠, ?). NOT verbose prose. NOT caveman ("compute. add. done."). NOT a copy
   of VERBOSE.
4. "preserves_self_correction" (bool or null): if VERBOSE contained "wait" /
   "actually" / "no" / "this gives...", COMPACT preserves them with ⚠ / ? or
   explicit retraction. null if VERBOSE had no self-correction.
5. "compression_quality" (int 1..5): 1=broken, 2=poor, 3=acceptable, 4=good,
   5=excellent.
6. "reason" (str, ≤1 sentence): the dominant reason for the verdict.

Output a single JSON object.
"""


def _verdict_from(audit: dict) -> str:
    if audit.get("preserves_self_correction") is False:
        return "fail"
    if not (audit.get("faithful") and audit.get("preserves_load_bearing")
            and audit.get("compact_register")
            and audit.get("compression_quality", 0) >= 3):
        return "fail"
    return "pass"


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in audit response: {text[:200]!r}")
    obj = json.loads(m.group(0))
    obj["verdict"] = _verdict_from(obj)
    return obj


async def _audit_one(client, model: str, row: dict, sem: asyncio.Semaphore) -> dict:
    prompt = AUDIT_PROMPT.format(
        problem=row.get("prompt", ""),
        verbose=row.get("thinking_original", ""),
        compact=row.get("compact", ""),
        gold=row.get("ground_truth", ""),
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                audit = _extract_json(text)
                return audit
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"verdict": "error", "reason": f"{type(e).__name__}: {e}"}
                await asyncio.sleep(2 ** attempt)
    return {"verdict": "error", "reason": "unreachable"}


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


async def _run(args):
    import anthropic

    api_key = os.environ.get("CLOUD_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: CLOUD_API_KEY or ANTHROPIC_API_KEY must be set", file=sys.stderr)
        sys.exit(1)
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    seen = _scan_seen(args.output)
    pending: list[dict] = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            key = (r["_uid"], r.get("src_candidate_idx", 0))
            if key in seen:
                continue
            pending.append(r)
    print(f"[audit] {len(pending)} compressions to audit", flush=True)
    if not pending:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)
    batch = max(1, args.concurrency * 2)
    n_pass = n_total = 0
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        results = await asyncio.gather(*[_audit_one(client, args.model, r, sem) for r in chunk])
        for r, audit in zip(chunk, results):
            rec = dict(r)
            rec["audit"] = audit
            out_f.write(json.dumps(rec) + "\n")
            n_total += 1
            n_pass += int(audit.get("verdict") == "pass")
        out_f.flush()
        rate = n_pass / n_total
        print(f"[audit] {i + len(chunk)}/{len(pending)} done, pass rate so far = {rate:.2%}",
              flush=True)

    out_f.close()
    rate = n_pass / max(1, n_total)
    print(f"[audit] DONE. {n_pass}/{n_total} = {rate:.2%} pass", flush=True)
    if rate < 0.6:
        print("[audit] WARN: pass rate < 60%. Compression prompt or chunk size may be off.",
              file=sys.stderr)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WHETSTONE Stage 2.5 compression audit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
