"""Draft long-trace register exemplars with an external model (activity 005 finding 9).

**Why an external model may author exemplars.** CLAUDE.md's central-model
principle (v1 §3) requires that the *compressor* be the same base model that
produced the harvest — no external teacher generating corpus content. It governs
who compresses, not who writes the card. The card is "the one human design input"
(design §1, precondition 2), and its existing exemplars were drafted by Claude
and ratified by the user. An exemplar drafted by GLM-5.2 sits in exactly that
category: an external model proposes card text, the user ratifies it, and
Qwen3-1.7B still performs every compression that produces corpus data.

**What this fixes.** All five ratified exemplars compress *short* problems
(127–533 chars of output). A model imitating them compresses an 8,800-token
level-7 trace by the same visual proportion, which is where the pilot audit
measured 0% faithfulness. This asks a stronger model to demonstrate the register
on a genuinely long, branch-heavy trace — showing that a faithful rewrite of a
40-step derivation is itself long.

**The reachability caveat is the whole risk.** An exemplar Qwen3-1.7B cannot
imitate is useless or harmful — that is precisely how arm B failed the bake-off
(0.24 markers/100 tokens: a register the model never installed). So output here
is a *candidate* for an A/B arm, never a card edit, and it must be hand-read
before use.

Usage::

    export FAITHFULNESS_BASE_URL=... FAITHFULNESS_AUTH_TOKEN=...
    python scripts/make_glm_exemplars.py \\
        --inputs /data/whetstone/runs/card_ab/compression_inputs.jsonl \\
        --out    /data/whetstone/runs/card_ab/glm_exemplars.jsonl \\
        --n 3 --min-think-tokens 6000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl

SYSTEM = (
    "You rewrite verbose mathematical chain-of-thought into a compact symbolic "
    "register. You follow the register specification exactly. You never drop a "
    "derivation step, a case branch, a rejected alternative, or a "
    "self-correction. Output ONLY the rewritten trace — no preamble, no code "
    "fence, no commentary."
)

PROMPT = """Here is the register specification you must write in:

{spec}

Rewrite the following verbose reasoning into that register.

This trace is LONG and the rewrite must be long enough to carry all of its
reasoning. Do not compress by omitting steps. Every intermediate value, every
case considered, every rejected approach, and every self-correction in the
original must appear in your rewrite with its verdict. A short rewrite of a long
derivation is a failure.

PROBLEM: {problem}

VERBOSE REASONING:
{verbose}"""


def _spec(card_path: str) -> str:
    """The notation spec only (§1.1–§1.4) — not the existing exemplars.

    Feeding the short exemplars back in would reproduce the very length anchor
    this script exists to break.
    """
    text = open(card_path).read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    start = text.index("## 1. Notation spec")
    end = text.index("### 1.5")
    return text[start:end].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="compression_inputs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default="configs/register_card.md")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--base_url", default=os.environ.get("FAITHFULNESS_BASE_URL"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-think-tokens", type=int, default=6000)
    ap.add_argument("--min-level", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()

    token = os.environ.get("FAITHFULNESS_AUTH_TOKEN")
    if not token or not args.base_url:
        raise SystemExit("[exemplars] set FAITHFULNESS_BASE_URL and "
                         "FAITHFULNESS_AUTH_TOKEN")

    import anthropic
    client = anthropic.Anthropic(base_url=args.base_url, auth_token=token,
                                 max_retries=4)
    spec = _spec(args.card)

    rows = [r for r in read_jsonl(args.inputs)
            if r.get("verbose_think_tokens", 0) >= args.min_think_tokens
            and int(r.get("level") or 0) >= args.min_level]
    rows.sort(key=lambda r: -r["verbose_think_tokens"])
    # Spread across the long tail rather than taking the N longest, so the
    # exemplars are not all pathological outliers.
    picks = rows[:: max(1, len(rows) // max(1, args.n))][: args.n]
    print(f"[in] {len(rows)} candidates (level>={args.min_level}, "
          f"think>={args.min_think_tokens}) -> {len(picks)} picked")

    out = []
    for r in picks:
        msg = client.messages.create(
            model=args.model, max_tokens=args.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT.format(
                spec=spec, problem=r["prompt"], verbose=r["verbose_think"])}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        rec = {
            "_uid": r["_uid"], "level": r.get("level"),
            "prompt": r["prompt"], "ground_truth": r.get("ground_truth"),
            "verbose_think": r["verbose_think"],
            "verbose_think_tokens": r["verbose_think_tokens"],
            "exemplar": text,
            "exemplar_chars": len(text),
            "exemplar_lines": len(text.splitlines()),
            "author_model": args.model,
            # Hard post-conditions — card §1.5 forbids a boxed answer in think.
            "has_boxed": bool(re.search(r"\\boxed\{", text)),
            "has_fence": "```" in text,
        }
        out.append(rec)
        print(f"  {r['_uid']:<24} L{r['level']} {r['verbose_think_tokens']:6d} tok "
              f"-> {rec['exemplar_lines']:3d} lines / {rec['exemplar_chars']:5d} chars"
              f"{'  ** BOXED **' if rec['has_boxed'] else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[out] {len(out)} candidate exemplars -> {args.out}")
    print("[!] HAND-READ THESE before building a card variant: an exemplar the "
          "1.7B compressor cannot imitate is worse than none (bake-off arm B).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
