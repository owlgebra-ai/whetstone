"""LLM-judge faithfulness audit of a compressed register corpus (P3 Part 2 DoD).

**Why this exists.** Every automated check the compression pipeline already runs
is orthogonal to faithfulness:

  * ``verify_response`` grades the **answer segment**, which is copied through
    untouched — it would pass on an empty ``compact_think``;
  * register-marker density, compression ratio and entropy p80 measure style,
    size and predictability;
  * the Δlogp gate is the only content signal, and it is a low bar — ``delta > 0``
    means "better than no trace at all", which a trace retaining a fraction of
    the reasoning clears.

So the pipeline can certify a corpus as clean, well-formed and on-register while
saying nothing about whether the compression preserved the reasoning. The packet
asks for 5 hand-inspected examples; that cannot characterise the tail. This
script does at sample scale what that eyeball does at n=5.

**Deviation from the central-model principle (v1 §3), deliberate and logged.**
The compressor must be the same base model that produced the harvest — no
external teacher. That rule governs *corpus generation*. This judge only reads
and scores; nothing it emits enters the corpus, and its verdicts gate nothing
automatically. Keep it that way: if a judge verdict ever becomes a filter on
training data, an external model is shaping the corpus and the invariant is
broken.

Endpoint: any Anthropic-compatible Messages API. Configured for GLM-5.2 via
z.ai. Model-specific Anthropic parameters (``thinking``, ``output_config``,
``betas``) are deliberately **not** sent — they are Claude-model features and a
third-party endpoint will reject or silently ignore them.

Usage::

    export FAITHFULNESS_BASE_URL=https://api.z.ai/api/anthropic
    export FAITHFULNESS_AUTH_TOKEN=...
    python scripts/faithfulness_audit.py \\
        --corpus  /data/whetstone/corpora/seed_register/seed_register.jsonl \\
        --output  /data/whetstone/runs/faithfulness/verdicts.jsonl \\
        --n 200 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, stratified_sample, write_meta
from whetstone.runio import checkpoint, repair_tail, scan_seen

SYSTEM = (
    "You are a strict grader auditing whether a COMPACT rewrite of a "
    "chain-of-thought preserves the reasoning of the VERBOSE original. "
    "You are not grading style, brevity, notation, or whether the answer is "
    "correct — only whether the reasoning survived the rewrite. "
    "Reply with ONLY a JSON object. No prose, no code fence."
)

RUBRIC = """PROBLEM:
{problem}

VERBOSE REASONING (the original):
{verbose}

COMPACT REWRITE (what must be judged):
{compact}

The compact rewrite is allowed to drop: restatements of the problem, self-talk
("hmm", "let me think"), repeated re-derivations of the same result, and
narration. It is NOT allowed to drop: any step's final value, a case split, a
rejected branch, or a correction the verbose trace made to itself.

Return JSON with exactly these keys:
  "dropped_values"   : bool  — a load-bearing intermediate value present in the
                              verbose trace is absent from the compact one
  "fused_steps"      : bool  — two or more derivation steps collapsed such that
                              an intermediate result no longer appears
  "dropped_branch"   : bool  — a case split, rejected alternative, or
                              self-correction in the verbose trace is gone
  "invented_content" : bool  — the compact trace asserts something that is not
                              in, and does not follow from, the verbose trace
  "off_topic"        : bool  — the compact trace is about a different problem
  "verdict"          : "faithful" | "lossy" | "wrong"
  "note"             : string, <= 30 words, citing the specific step if not faithful

"lossy" = reasoning was thinned but the chain still follows; "wrong" = the
compact trace is unusable as a record of this reasoning."""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
BOOL_KEYS = ("dropped_values", "fused_steps", "dropped_branch",
             "invented_content", "off_topic")


def _parse(text: str) -> dict | None:
    """Tolerant JSON extraction — judges fence their output about half the time."""
    t = _FENCE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="glm-5.2",
                    help="Judge model id. NB the Claude Code alias form "
                         "'glm-5.2[1m]' is rejected by the raw API — use 'glm-5.2'.")
    ap.add_argument("--base_url", default=os.environ.get("FAITHFULNESS_BASE_URL"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_verbose_chars", type=int, default=40000,
                    help="Truncate very long verbose traces. A truncated trace "
                         "cannot evidence a drop in its tail, so the count of "
                         "truncated records is reported and excluded from the "
                         "dropped_* rates.")
    args = ap.parse_args()

    token = os.environ.get("FAITHFULNESS_AUTH_TOKEN")
    if not token or not args.base_url:
        raise SystemExit("[audit] set FAITHFULNESS_BASE_URL and "
                         "FAITHFULNESS_AUTH_TOKEN")

    import anthropic
    client = anthropic.Anthropic(base_url=args.base_url, auth_token=token,
                                 max_retries=4)

    rows = read_jsonl(args.corpus)
    # A record with no verbose source has nothing for the rubric to compare
    # against: the judge sees an empty ORIGINAL and grades the rewrite against
    # nothing. Measured on the Stage-A corpus (activity 008), those judgments
    # came back 94.7% "faithful" against 78.0% for source-bearing records — so
    # they do not merely add noise, they inflate the headline. Stage-A corpora
    # legitimately contain such records (the ~34% of problems with no verified
    # trace), so this is filtered here rather than assumed away.
    n_all = len(rows)
    rows = [r for r in rows if (r.get("verbose_think") or "").strip()]
    if len(rows) != n_all:
        print(f"[in] {n_all - len(rows)}/{n_all} records dropped: no verbose "
              f"source to judge against", flush=True)
    if not rows:
        raise SystemExit("[audit] no source-bearing records in the corpus")
    dropped = repair_tail(args.output)
    if dropped:
        print(f"[resume] repaired torn tail: dropped {dropped} B", flush=True)
    done = {k[0] for k in scan_seen(args.output, ("_uid",))}
    sample = [r for r in stratified_sample(rows, lambda r: str(r.get("level", "_")),
                                           args.n, random.Random(args.seed))
              if r["_uid"] not in done]
    print(f"[in] corpus {len(rows)}, sampled {args.n}, "
          f"{len(done)} already judged, {len(sample)} to do", flush=True)
    if not sample:
        print("[audit] nothing to do", flush=True)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)
    lock = threading.Lock()
    state = {"n": 0, "fail": 0, "unparsed": 0}
    t0 = time.time()

    def judge(r: dict) -> dict:
        verbose = r["verbose_think"]
        truncated = len(verbose) > args.max_verbose_chars
        if truncated:
            verbose = verbose[:args.max_verbose_chars] + "\n…[TRUNCATED]"
        prompt = RUBRIC.format(problem=r.get("prompt", ""), verbose=verbose,
                               compact=r["compact_think"])
        # No thinking/effort/betas: those are Claude-model parameters and this
        # is a third-party Anthropic-compatible endpoint.
        msg = client.messages.create(
            model=args.model, max_tokens=args.max_tokens,
            system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        parsed = _parse(text)
        rec = {
            "_uid": r["_uid"], "level": r.get("level"),
            "verdict_raw": None if parsed else text[:500],
            "parsed": bool(parsed),
            "verbose_truncated": truncated,
            "judge_model": args.model,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        if parsed:
            rec.update({k: bool(parsed.get(k)) for k in BOOL_KEYS})
            rec["verdict"] = parsed.get("verdict")
            rec["note"] = parsed.get("note")
        return rec

    def run_one(r: dict):
        try:
            rec = judge(r)
        except anthropic.NotFoundError as e:
            return ("fatal", f"{r['_uid']}: model not found — {e}")
        except anthropic.AuthenticationError as e:
            return ("fatal", f"{r['_uid']}: auth failed — {e}")
        except anthropic.RateLimitError as e:
            return ("fail", f"{r['_uid']}: rate limited — {e}")
        except anthropic.APIStatusError as e:
            return ("fail", f"{r['_uid']}: HTTP {e.status_code} — {e}")
        except anthropic.APIConnectionError as e:
            return ("fail", f"{r['_uid']}: connection — {e}")
        with lock:
            out_f.write(json.dumps(rec) + "\n")
            state["n"] += 1
            state["unparsed"] += int(not rec["parsed"])
            if state["n"] % 20 == 0:
                el = max(1e-9, time.time() - t0)
                checkpoint(args.output, out_f, {
                    "judged": state["n"], "queued": len(sample),
                    "failed": state["fail"], "unparsed": state["unparsed"],
                    "per_min": round(60 * state["n"] / el, 2), "done": False,
                })
                print(f"[audit] {state['n']}/{len(sample)} judged "
                      f"({state['fail']} failed, {state['unparsed']} unparsed)",
                      flush=True)
        return ("ok", None)

    fatal = None
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, r): r for r in sample}
        for f in as_completed(futs):
            kind, msg = f.result()
            if kind == "fatal":
                fatal = fatal or msg
                for g in futs:
                    g.cancel()
                break
            if kind == "fail":
                with lock:
                    state["fail"] += 1
                print(f"[fail] {msg}", flush=True)

    checkpoint(args.output, out_f, {"judged": state["n"], "queued": len(sample),
                                    "failed": state["fail"], "done": True})
    out_f.close()
    if fatal:
        raise SystemExit(f"[audit] aborted: {fatal}")

    # ---- report -----------------------------------------------------------
    verdicts = [r for r in read_jsonl(args.output) if r.get("parsed")]
    if not verdicts:
        print("[audit] no parsable verdicts", file=sys.stderr)
        return 1
    untrunc = [r for r in verdicts if not r.get("verbose_truncated")]
    vc = Counter(r.get("verdict") for r in verdicts)
    n = len(verdicts)
    print(f"\n=== faithfulness audit: {n} judged "
          f"({len(verdicts) - len(untrunc)} truncated) ===")
    print("verdict: " + ", ".join(f"{k}={v} ({v/n:.0%})" for k, v in vc.most_common()))
    base = untrunc or verdicts
    for k in BOOL_KEYS:
        hits = sum(1 for r in base if r.get(k))
        print(f"  {k:<18} {hits}/{len(base)} = {hits/len(base):.1%}")

    by_lvl: dict[str, Counter] = defaultdict(Counter)
    for r in verdicts:
        by_lvl[str(r.get("level"))][r.get("verdict")] += 1
    print("\n| level | n | faithful | lossy | wrong |")
    print("|---|---|---|---|---|")
    for lv in sorted(by_lvl, key=lambda x: (x == "None", x)):
        c = by_lvl[lv]
        t = sum(c.values())
        print(f"| {lv} | {t} | {c['faithful']/t:.0%} | {c['lossy']/t:.0%} | "
              f"{c['wrong']/t:.0%} |")

    worst = [r for r in verdicts if r.get("verdict") == "wrong"][:10]
    if worst:
        print("\nworst (verdict=wrong):")
        for r in worst:
            print(f"  {r['_uid']:<24} {r.get('note','')}")

    write_meta(args.output, {
        "builder": "scripts/faithfulness_audit.py",
        "packet": "P3 Part 2 (faithfulness audit)",
        "corpus": args.corpus, "n_judged": n, "seed": args.seed,
        "judge_model": args.model, "base_url": args.base_url,
        "verdicts": dict(vc),
        "rates": {k: round(sum(1 for r in base if r.get(k)) / len(base), 4)
                  for k in BOOL_KEYS},
        "n_truncated": len(verdicts) - len(untrunc),
        "external_judge_note": (
            "External LLM judge — a deliberate, logged exception to the "
            "central-model principle (v1 §3). Evaluation only: no judge output "
            "enters the corpus and no verdict gates training data."),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
