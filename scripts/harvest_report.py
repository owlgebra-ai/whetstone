"""Per-level yield table for a seed harvest (packet P3 Part 1, "Log yield per level band").

Three gates matter for a harvest and they fail for different reasons, so they
are reported separately rather than as one "yield" number:

  * **verifier** (:func:`whetstone.verify.verify_response`) — did the rollout
    reach the gold answer;
  * **segment parser** (:func:`whetstone.segments.parse_segments`, ``g``) — is
    the rollout structurally well-formed. A cap-hit trace is verifier-wrong
    *and* has no answer segment to copy through, so Part 2 needs this gate too
    and a corpus filtered on the verifier alone would still carry junk;
  * **cap-hit** (``finish_reason == "length"``) — ran out of budget. Tracked on
    its own because a rise here is a sampling problem, not a difficulty problem.

Reference numbers to compare against (activity 003 probe, same sampling config,
no system prompt): **73% at K=2 over 50 problems**, U-shaped in level — 86% at
level 1, ~56% at level 5, 50% at level 9 — and the bulk harvest is expected to
land ~3 points under the probe from known extraction-shape losses (unit
suffixes, ``$$`` blocks; activity 003 finding 9). Substantially below that is a
bug, not difficulty: compare the per-level column against the probe before
burning more GPU time.

Think and answer lengths are reported as **separate** medians — one combined
length number is how segment drift hides (CLAUDE.md invariant).

Usage (any box with the tokenizer; CPU only)::

    python scripts/harvest_report.py \\
        --harvest /data/whetstone/corpora/seed/seed_harvest.jsonl \\
        --out_dir /data/whetstone/runs/seed_harvest_report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.segments import blank_token_ids_for, parse_segments
from whetstone.verify import verify_response


def _median(xs):
    if not xs:
        return 0
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--progress_every", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    blank = blank_token_ids_for(tok)

    per_level: dict[str, Counter] = defaultdict(Counter)
    think_len: dict[str, list] = defaultdict(list)
    ans_len: dict[str, list] = defaultdict(list)
    gate_reasons: Counter = Counter()
    # problem-level: did ANY candidate for this uid verify (the K=2 solve rate)
    solved: dict[str, bool] = {}
    lvl_of: dict[str, str] = {}
    n = 0

    with open(args.harvest) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            lvl = str(r.get("level"))
            uid = r["_uid"]
            lvl_of[uid] = lvl
            c = per_level[lvl]
            c["rollouts"] += 1

            ok = verify_response(r.get("completion", ""), r.get("ground_truth", ""))
            c["verified"] += int(ok)
            solved[uid] = solved.get(uid, False) or ok

            if r.get("finish_reason") == "length":
                c["cap_hit"] += 1

            ids = r.get("completion_token_ids") or []
            if ids:
                m = parse_segments(ids, blank_token_ids=blank)
                c["gate_ok"] += int(m.g == 1)
                if m.g != 1:
                    gate_reasons[m.reason] += 1
                else:
                    think_len[lvl].append(m.think_len)
                    ans_len[lvl].append(m.answer_len)
                    # The gate Part 2 actually selects on.
                    c["verified_and_gated"] += int(ok)
            if args.progress_every and n % args.progress_every == 0:
                print(f"[report] {n} rollouts scanned", flush=True)

    for uid, ok in solved.items():
        per_level[lvl_of[uid]]["problems"] += 1
        per_level[lvl_of[uid]]["problems_solved"] += int(ok)

    levels = sorted(per_level, key=lambda x: (x == "None", x if x == "None" else int(x)))
    tot = Counter()
    for lv in levels:
        tot.update(per_level[lv])

    rows = []
    for lv in levels + ["ALL"]:
        c = tot if lv == "ALL" else per_level[lv]
        rr = c["rollouts"] or 1
        pp = c["problems"] or 1
        tl = ([t for lvv in levels for t in think_len[lvv]] if lv == "ALL"
              else think_len[lv])
        al = ([a for lvv in levels for a in ans_len[lvv]] if lv == "ALL"
              else ans_len[lv])
        rows.append({
            "level": lv,
            "problems": c["problems"],
            "rollouts": c["rollouts"],
            "verify_rate": round(c["verified"] / rr, 4),
            "solve_rate_at_K": round(c["problems_solved"] / pp, 4),
            "gate_rate": round(c["gate_ok"] / rr, 4),
            "verified_and_gated": c["verified_and_gated"],
            "usable_rate": round(c["verified_and_gated"] / rr, 4),
            "cap_hit_rate": round(c["cap_hit"] / rr, 4),
            "think_tokens_median": _median(tl),
            "answer_tokens_median": _median(al),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "harvest": args.harvest,
        "n_rollouts": n,
        "n_problems": len(solved),
        "rows": rows,
        "gate_fail_reasons": dict(gate_reasons),
        "note": ("verify_rate is per rollout; solve_rate_at_K is per problem "
                 "(any candidate correct). usable_rate = verified AND parser "
                 "gate g==1 — the pool Part 2 selects compression inputs from."),
    }
    with open(os.path.join(args.out_dir, "yield.json"), "w") as f:
        json.dump(summary, f, indent=1)

    hdr = ("| level | problems | rollouts | verify | solve@K | gate | usable | "
           "cap-hit | think med | answer med |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['problems']} | {r['rollouts']} | "
            f"{r['verify_rate']:.1%} | {r['solve_rate_at_K']:.1%} | "
            f"{r['gate_rate']:.1%} | {r['usable_rate']:.1%} | "
            f"{r['cap_hit_rate']:.1%} | {r['think_tokens_median']} | "
            f"{r['answer_tokens_median']} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "yield.md"), "w") as f:
        f.write(table + "\n\ngate failures: "
                + json.dumps(dict(gate_reasons)) + "\n")

    print(table)
    print("\ngate failures:", dict(gate_reasons))
    print(f"\n[out] {args.out_dir}/yield.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
