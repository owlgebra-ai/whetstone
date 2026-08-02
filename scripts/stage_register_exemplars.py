"""Stage real (problem, verbose think trace) pairs for the register card.

Packet P2 Part 4 hands the register card to the user as a fill-in exercise. The
card asks for 5-10 **verbose -> compact** exemplar pairs drawn from real pool
problems, spanning difficulty bands and covering at least one algebra, one
combinatorics and one geometry problem.

Writing the compact side is the human's job (design §1: the register is
*specified, not discovered*). Producing the verbose side is not — it is just
the model's own output. This script picks well-formed, verifier-correct
rollouts spanning the level bands and topic buckets and writes them into a
markdown file with an empty slot under each, so the user only writes register
notation.

Topic bucketing is a transparent keyword heuristic over the problem text — it
exists to make "at least one algebra / combinatorics / geometry" checkable, not
to be a classifier. Every staged item prints its bucket so a wrong call is
obvious and easy to swap.

Usage::

    python scripts/stage_register_exemplars.py \\
        --rollouts /data/whetstone/runs/entropy_audit/rollouts.jsonl \\
        --out configs/register_card_exemplars_staged.md --n 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl
from whetstone.reward.extract import split_think_close
from whetstone.verify import verify_response

TOPIC_PATTERNS = [
    ("geometry", r"\b(triangle|circle|angle|polygon|radius|perimeter|area of|"
                 r"rectangle|square|sphere|cylinder|parallel|tangent|vertex|"
                 r"vertices|hypotenuse|coordinates?)\b"),
    ("combinatorics", r"\b(how many ways|permutation|combination|choose|distinct "
                      r"arrangements|probability|binomial|factorial|subsets?|"
                      r"counting)\b"),
    ("number_theory", r"\b(divisor|prime|modulo|mod |remainder|gcd|lcm|integer "
                      r"solutions|digits?)\b"),
    ("calculus", r"\b(integral|derivative|limit|\\int|\\lim|converges?|diverges?|"
                 r"series|\\sum)\b"),
    ("algebra", r"\b(solve for|equation|polynomial|roots?|factor|simplify|"
                r"expression|inequality|function|x\^|matrix)\b"),
]


def topic_of(prompt: str) -> str:
    low = prompt.lower()
    for name, pat in TOPIC_PATTERNS:
        if re.search(pat, low):
            return name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--pool", default=None,
                    help="pool JSONL to join golds from, by _uid, when the "
                         "rollouts file predates entropy_audit.py carrying "
                         "ground_truth through. Without a gold the trace cannot "
                         "be verifier-filtered and is skipped.")
    ap.add_argument("--out", default="configs/register_card_exemplars_staged.md")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_think_chars", type=int, default=12000,
                    help="skip traces longer than this — unreadable as exemplars. "
                         "Note the median native think trace on this pool is "
                         "~6,100 tokens (~20k chars), so a tight cap here leaves "
                         "almost nothing: 4000 yielded 1 candidate out of 200.")
    args = ap.parse_args()

    rows = read_jsonl(args.rollouts)
    golds: dict = {}
    if args.pool:
        golds = {r["_uid"]: r.get("ground_truth") for r in read_jsonl(args.pool)}
        print(f"[join] {len(golds)} golds from {args.pool}")

    cands = []
    n_no_gold = 0
    for r in rows:
        sp = split_think_close(r.get("completion", ""))
        if not sp.has_closed_think or not sp.post_think.strip():
            continue
        gold = r.get("ground_truth") or golds.get(r["_uid"]) or ""
        # Only correct traces — an exemplar built on a wrong trace teaches the
        # register on reasoning that never reaches the answer. A trace with no
        # gold is unverifiable, so it is skipped rather than trusted.
        if not gold:
            n_no_gold += 1
            continue
        if not verify_response(r["completion"], gold):
            continue
        if len(sp.think) > args.max_think_chars:
            continue
        # sp.think keeps the opening "<think>" tag; strip it so the staged text
        # is the reasoning body the user actually rewrites.
        think_body = sp.think.strip()
        if think_body.startswith("<think>"):
            think_body = think_body[len("<think>"):].strip()

        cands.append({
            "_uid": r["_uid"],
            "level": r.get("level"),
            "topic": topic_of(r["prompt"]),
            "prompt": r["prompt"],
            "think": think_body,
            "answer": sp.post_think.strip(),
            "gold": gold,
            "think_chars": len(think_body),
        })

    if n_no_gold:
        print(f"[warn] {n_no_gold} traces skipped: no gold available "
              f"(pass --pool to join them)")
    if not cands:
        print("No usable candidates. Are golds present in the rollouts file?")
        return 1

    # Spread over topic first (the card requires 3 named topics), then level.
    picked: list[dict] = []
    for want in ("algebra", "combinatorics", "geometry"):
        pool = [c for c in cands if c["topic"] == want and c not in picked]
        if pool:
            # Prefer the shorter end — these get hand-rewritten by a person.
            picked.append(sorted(pool, key=lambda c: c["think_chars"])[0])
    by_level: dict = {}
    for c in sorted(cands, key=lambda c: c["think_chars"]):
        by_level.setdefault(c["level"], []).append(c)
    for lvl in sorted(by_level):
        if len(picked) >= args.n:
            break
        for c in by_level[lvl]:
            if c not in picked:
                picked.append(c)
                break
    picked = picked[: args.n]
    picked.sort(key=lambda c: (c["level"] or 0))

    lines = [
        "<!-- GENERATED by scripts/stage_register_exemplars.py — raw material for",
        "     configs/register_card.md §3. Each block below is a REAL pool problem",
        "     with the model's REAL verbose think trace. Write the compact-register",
        "     rewrite in the empty slot, then copy the finished pairs into the card.",
        "",
        "     Topic labels come from a keyword heuristic — if one looks wrong, swap",
        "     the problem for another; this file is disposable. -->",
        "",
        "# Staged exemplar candidates (verbose side only)",
        "",
        f"Source: `{args.rollouts}`  ·  {len(picked)} of {len(cands)} eligible traces",
        "(well-formed think block, verifier-correct, think trace under "
        f"{args.max_think_chars} chars).",
        "",
        "Topic coverage: " + ", ".join(sorted({c["topic"] for c in picked})) + ".",
        "",
        "**Reminder — the one constraint:** shorter lines, not bigger jumps. Every",
        "step in the verbose trace must survive into the compact one; only the prose",
        "around the values is dropped. Never elide a step's final value.",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(picked, 1):
        lines += [
            f"## Candidate {i} — level {c['level']}, {c['topic']}",
            "",
            f"- **_uid:** `{c['_uid']}`",
            f"- **gold:** `{c['gold']}`",
            "",
            "**Problem:**",
            "",
            "```",
            c["prompt"].strip(),
            "```",
            "",
            f"**Verbose think trace ({c['think_chars']} chars):**",
            "",
            "```",
            c["think"],
            "```",
            "",
            "**Answer segment (copied through untouched — compression never "
            "touches post-`</think>` text):**",
            "",
            "```",
            c["answer"],
            "```",
            "",
            "**Compact-register rewrite:** ⟨TODO — you write this⟩",
            "",
            "```",
            "",
            "```",
            "",
            "---",
            "",
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))

    print(f"[staged] {len(picked)} candidates -> {args.out}")
    for c in picked:
        print(f"   level {str(c['level']):>2}  {c['topic']:<15} "
              f"{c['think_chars']:>5} chars  {c['_uid']}")
    missing = {"algebra", "combinatorics", "geometry"} - {c["topic"] for c in picked}
    if missing:
        print(f"[warn] card requires these topics but none were staged: "
              f"{sorted(missing)} — widen --n or pick manually from the pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
