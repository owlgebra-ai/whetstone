"""Bootstrap a register corpus with an EXTERNAL compressor (attested deviation).

⚠️  **This deviates from the central-model principle** (v1 §3, CLAUDE.md
invariant: "The compressor is the SAME base model that produced the harvest. No
external teacher"). The deviation is deliberate, user-authorised, and attested
on every record it writes.

**Why.** The pilot faithfulness audit (activity 005 finding 9) measured
Qwen3-1.7B's own compressions at 49% faithful / 22% wrong, collapsing to 0%
faithful at level 7. A seed corpus built from that demonstrates the register
badly. A stronger compressor produces a corpus that shows the target behaviour.

**Where its output may and may not be consumed.** The risk is not uniform, and
every record is stamped so no stage can consume it unknowingly:

  * ``Stage-A teacher conditioning`` — **safe, arguably better.** These are
    in-context demonstrations, and Stage-A RL pulls the teacher toward whatever
    earns reward under the frozen student scorer regardless of what the
    exemplars looked like.
  * ``Round-0 scorer inoculation`` — **unsafe.** Round 0 calibrates the scorer
    on *the student's own* register statistics so register tokens read as "hum".
    Inoculating on an external model's distribution calibrates against text the
    student never produces, which is the silently-inverted-meter failure
    CLAUDE.md names as the project's largest risk.
  * ``H_pivot`` — **unsafe.** p80 of this corpus under Qwen3 measures "how
    surprising is external text to Qwen3", not "how surprising is Qwen3's own
    register to Qwen3". Different quantity; mis-sets the SED gate.
  * ``Stage-B assimilation`` — **wasteful.** The ZPD band-pass gates off tokens
    outside the student's reachable zone, so much of an external corpus is
    masked rather than learned.

Every record carries ``compressor_model`` and ``central_model_deviation: true``.
**Do not strip those fields.** They are how a later stage knows what it is
holding.

Record shape matches ``compress_local_versionB.py`` so the Δlogp gate, entropy
audit and ``finalize_seed_register.py`` consume it unchanged.

Usage::

    export FAITHFULNESS_BASE_URL=... FAITHFULNESS_AUTH_TOKEN=...
    python scripts/glm_compress.py \\
        --input  /data/whetstone/runs/card_ab/compression_inputs.jsonl \\
        --output /data/whetstone/corpora/seed_register_glm/compact.jsonl \\
        --card configs/register_card.md --n 1000 --concurrency 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_meta
from whetstone.runio import checkpoint, repair_tail, scan_seen
from whetstone.verify import verify_response

# Reuse the pinned scaffold + card rendering + cleaner so the only difference
# from the in-house path is which model completes the prompt.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compress_local_versionB import (  # noqa: E402
    _BOXED, build_system_prompt, clean_oneshot, render_card, _git_sha, _sha1,
)


def diverse_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin across (level, topic-family) cells rather than proportionally.

    Proportional stratification reproduces the pool's shape, which is peaked at
    levels 5–8 — so rare-but-real strata (levels 2, 3, 9, 10 and thin topics)
    would contribute almost nothing to a corpus whose whole job is to
    *demonstrate the register across the space*. Round-robin over cells gives
    thin strata representation without letting them dominate.
    """
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        topic = (r.get("topic") or "").split("->")
        fam = topic[1].strip() if len(topic) > 1 else (r.get("source") or "_")
        cells[(str(r.get("level")), fam)].append(r)
    for c in cells.values():
        rng.shuffle(c)
    keys = sorted(cells)
    out: list[dict] = []
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k]:
                out.append(cells[k].pop())
                if len(out) >= n:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pool", default="/data/whetstone/data/pool/train_30k.jsonl",
                    help="Joined by _uid to recover DeepMath's `topic`, which "
                         "the harvest does not carry. Topic is the second axis "
                         "of the diversity sample; without it the sampler falls "
                         "back to source alone.")
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--base_url", default=os.environ.get("FAITHFULNESS_BASE_URL"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()

    token = os.environ.get("FAITHFULNESS_AUTH_TOKEN")
    if not token or not args.base_url:
        raise SystemExit("[glm] set FAITHFULNESS_BASE_URL and FAITHFULNESS_AUTH_TOKEN")

    import anthropic
    client = anthropic.Anthropic(base_url=args.base_url, auth_token=token,
                                 max_retries=5)

    card_raw = open(args.card).read()
    card_text, dropped_headings = render_card(card_raw)
    system_prompt = build_system_prompt(card_text, "oneshot")
    meta = {
        "compressor_model": args.model,
        "central_model_deviation": True,
        "card_path": args.card,
        "card_git_sha": _git_sha(args.card),
        "rendered_prompt_sha1": _sha1(system_prompt),
        "dropped_headings": dropped_headings,
    }
    print(f"[card] {args.card} blob={meta['card_git_sha'][:12]} "
          f"prompt_sha1={meta['rendered_prompt_sha1'][:12]}", flush=True)

    rows = read_jsonl(args.input)
    if args.pool and os.path.exists(args.pool):
        topics = {r["_uid"]: r.get("topic") for r in read_jsonl(args.pool)}
        hit = 0
        for r in rows:
            t = topics.get(r["_uid"])
            if t:
                r["topic"] = t
                hit += 1
        print(f"[pool] topic joined for {hit}/{len(rows)} inputs", flush=True)

    dropped = repair_tail(args.output)
    if dropped:
        print(f"[resume] repaired torn tail: dropped {dropped} B", flush=True)
    done = {k[0] for k in scan_seen(args.output, ("_uid",))}
    sample = [r for r in diverse_sample(rows, args.n, random.Random(args.seed))
              if r["_uid"] not in done]
    lv = Counter(str(r.get("level")) for r in sample)
    print(f"[in] {len(rows)} inputs -> {len(sample)} to compress "
          f"({len(done)} already done)", flush=True)
    print(f"     by level: {dict(sorted(lv.items()))}", flush=True)
    if not sample:
        print("[glm] nothing to do", flush=True)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", buffering=1)
    lock = threading.Lock()
    st = {"n": 0, "fail": 0, "bad_verify": 0, "boxed": 0, "flags": Counter()}
    t0 = time.time()

    def one(r: dict):
        try:
            msg = client.messages.create(
                model=args.model, max_tokens=args.max_tokens, system=system_prompt,
                messages=[{"role": "user", "content":
                           f"PROBLEM: {r['prompt']}\n\n"
                           f"VERBOSE REASONING:\n{r['verbose_think']}"}],
            )
        except anthropic.AuthenticationError as e:
            return ("fatal", f"auth failed — {e}")
        except anthropic.NotFoundError as e:
            return ("fatal", f"model not found — {e}")
        except Exception as e:                                   # noqa: BLE001
            return ("fail", f"{r['_uid']}: {type(e).__name__}: {e}")

        text = "".join(b.text for b in msg.content if b.type == "text")
        compact, flags = clean_oneshot(text)
        compact = compact or "[empty]"
        answer = r["answer"]
        completion = f"<think>\n{compact}\n</think>\n\n{answer}"
        rec = {
            "_uid": r["_uid"],
            "src_candidate_idx": r.get("src_candidate_idx", 0),
            "level": r.get("level"), "source": r.get("source"),
            "prompt": r["prompt"], "ground_truth": r.get("ground_truth", ""),
            "verbose_think": r["verbose_think"],
            "verbose_think_tokens": r.get("verbose_think_tokens"),
            "compact_think": compact,
            # Char-based: this path has no local tokenizer, and the field is
            # relabelled so it is never mistaken for a real token count.
            "compact_think_chars": len(compact),
            "compression_ratio_chars": len(compact) / max(1, len(r["verbose_think"])),
            "answer": answer,
            "completion": completion,
            "verify_ok": bool(verify_response(completion, r.get("ground_truth", ""))),
            "think_has_boxed": bool(_BOXED.search(compact)),
            "clean_flags": flags,
            "output_tokens": msg.usage.output_tokens,
            **meta,
        }
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["n"] += 1
            st["bad_verify"] += int(not rec["verify_ok"])
            st["boxed"] += int(rec["think_has_boxed"])
            st["flags"].update(flags)
            if st["n"] % 25 == 0:
                el = max(1e-9, time.time() - t0)
                checkpoint(args.output, out_f, {
                    "written": st["n"], "queued": len(sample),
                    "failed": st["fail"], "verify_fail": st["bad_verify"],
                    "boxed_in_think": st["boxed"],
                    "per_min": round(60 * st["n"] / el, 2), "done": False,
                })
                print(f"[glm] {st['n']}/{len(sample)} "
                      f"({st['fail']} failed, {st['bad_verify']} verify-fail, "
                      f"{60*st['n']/el:.1f}/min)", flush=True)
        return ("ok", None)

    fatal = None
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, r) for r in sample]
        for f in as_completed(futs):
            kind, msg = f.result()
            if kind == "fatal":
                fatal = fatal or msg
                for g in futs:
                    g.cancel()
                break
            if kind == "fail":
                with lock:
                    st["fail"] += 1
                if st["fail"] <= 10:
                    print(f"[fail] {msg}", flush=True)

    checkpoint(args.output, out_f, {"written": st["n"], "queued": len(sample),
                                    "failed": st["fail"], "done": True})
    out_f.close()
    if fatal:
        raise SystemExit(f"[glm] aborted: {fatal}")

    n_all = sum(1 for _ in open(args.output))
    print(f"\n[out] {st['n']} written this run, {n_all} total -> {args.output}")
    print(f"      failed {st['fail']}, verify-fail {st['bad_verify']}, "
          f"boxed-in-think {st['boxed']}, cleaner {dict(st['flags'])}")
    write_meta(args.output, {
        "builder": "scripts/glm_compress.py", **meta,
        "input": args.input, "n": n_all, "seed": args.seed,
        "sampling": "round-robin over (level, topic-family) cells",
        "WARNING": (
            "EXTERNAL COMPRESSOR — attested deviation from the central-model "
            "principle (v1 §3). Safe for Stage-A teacher conditioning. NOT for "
            "Round-0 scorer inoculation or H_pivot: both measure the STUDENT's "
            "own distribution, and this corpus is not from it."),
    })
    if st["bad_verify"] or st["boxed"]:
        print(f"[glm] *** {st['bad_verify']} verify failures and {st['boxed']} "
              "boxed-in-think — inspect before use", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
