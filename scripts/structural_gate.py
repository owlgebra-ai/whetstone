"""Structural faithfulness gate — replaces the Δlogp gate (activity 005 findings 9, 11).

**Why Δlogp was retired.** Every metric of the form ``P(gold | q, compact)``
scores highest when the answer is literally present in the context, so it cannot
distinguish "derived 30 and wrote 30" from "just wrote 30". Measured three ways:
17% of gate-passing traces were judged unfaithful in exactly that shape; the
more faithful GLM corpus *passed less often* than the terser Qwen3 one (58% vs
70%); and masking the trailing conclusion did not fix it (56% vs 63%) because
the gold string survives masking in 54–61% of traces — a correct derivation ends
by producing the answer.

**What this measures instead.** Card §1.4's "never elided" column, directly, by
comparing compact against its own verbose source. Every check is a deterministic
string operation: no model, no external judge, no distributional bias, free.

  * ``branch_kept`` — the verbose trace explored cases / rejected an approach /
    corrected itself, and the compact trace shows that (card §1.4: "branch
    elimination is reasoning, and it stays");
  * ``verify_kept`` — the verbose trace verified its answer and the compact
    carries a ``chk:``/``✓``;
  * ``value_coverage`` — of the numbers the verbose trace keeps *returning to*
    (its load-bearing quantities), how many survive;
  * ``no_invention`` — the compact introduces no numeral absent from the source;
  * ``length_floor`` — compact length scales with source complexity, so a
    40-step derivation cannot pass as six lines.

Each check fires **only when the source warrants it** — a trace whose verbose
form has no case split is not penalised for lacking one. That is what keeps this
a faithfulness measure rather than a style measure.

**Records are annotated, never dropped.** Whether to filter is a per-corpus
decision, and the two corpora want opposite things: the Stage-A conditioning
corpus wants quality, while the Round-0 / H_pivot corpus wants *representativeness*
of what the student actually produces — filtering that one biases the very
distribution it exists to measure.

Usage::

    # calibrate: show feature distributions, apply no verdict
    python scripts/structural_gate.py --input compact.jsonl --calibrate

    # annotate with pass/fail under chosen thresholds
    python scripts/structural_gate.py --input compact.jsonl --output gated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, write_meta

# --- source-side detectors: did the ORIGINAL contain this thing at all? -----
# Deliberately generous. A false positive here only means we demand a feature
# the compact may not need; a false negative silently excuses a real drop.
VERBOSE_BRANCH = re.compile(
    r"\b(?:but wait|actually|let me reconsider|that'?s wrong|doesn'?t work|"
    r"contradiction|alternatively|another approach|case \d|either way|"
    r"on the other hand|scratch that|no,? that)", re.I)
VERBOSE_VERIFY = re.compile(
    r"\b(?:let me (?:verify|check|double.?check)|plug(?:ging)? (?:it |this )?back|"
    r"verify(?:ing)? (?:that|the)|sanity check|check:|confirms?\b)", re.I)

# --- compact-side detectors: does the REWRITE carry it? --------------------
COMPACT_BRANCH = re.compile(r"(?:^|\s)(?:case\b|✗)", re.M)
COMPACT_VERIFY = re.compile(r"(?:^|\s)(?:chk:|✓)", re.M)

#: Thousands-grouped first, then plain. Matching in this order matters: a
#: blanket ``replace(",", "")`` fuses step back-references — the register's
#: own ``from 4,5:`` becomes the numeral 45 — which then reads as invented
#: content. That artifact scales with how much derivation structure a rewrite
#: shows, so it penalised exactly the traces it should reward.
NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
STEP = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s|=")
#: Line-leading step markers ("3.", "12)"). The register numbers its steps, so
#: a longer — i.e. more faithful — rewrite mechanically introduces more of these
#: integers. Counting them as content made the *better* corpus look like it
#: invented more numbers (0.25 vs 0.083 before this was stripped).
STEP_NUM = re.compile(r"(?m)^\s*\d+[.)]\s")


def _nums(t: str, strip_step_numbers: bool = False) -> list[str]:
    if strip_step_numbers:
        t = STEP_NUM.sub("", t)
    return [m.replace(",", "") for m in NUM.findall(t)]


def features(r: dict) -> dict:
    verbose, compact = r.get("verbose_think", ""), r.get("compact_think", "")
    vn = _nums(verbose)
    cn = set(_nums(compact, strip_step_numbers=True))

    # Load-bearing quantities: values the verbose trace keeps returning to.
    # A number mentioned once is usually incidental (a coordinate, an index);
    # one mentioned repeatedly is a result the reasoning is built on.
    freq = Counter(vn)
    anchors = [v for v, c in freq.most_common(12) if c >= 2]
    covered = sum(1 for v in anchors if v in cn)

    src_branch = bool(VERBOSE_BRANCH.search(verbose))
    src_verify = bool(VERBOSE_VERIFY.search(verbose))
    c_lines = len([l for l in compact.splitlines() if l.strip()])
    # Source complexity proxy: numbered items and equations in the original.
    v_steps = len(STEP.findall(verbose))

    return {
        "src_has_branch": src_branch,
        "src_has_verify": src_verify,
        "branch_kept": (not src_branch) or bool(COMPACT_BRANCH.search(compact)),
        "verify_kept": (not src_verify) or bool(COMPACT_VERIFY.search(compact)),
        "n_anchors": len(anchors),
        "value_coverage": round(covered / len(anchors), 4) if anchors else 1.0,
        "invented": len(cn - set(vn)),
        "invented_frac": round(len(cn - set(vn)) / max(1, len(cn)), 4),
        "compact_lines": c_lines,
        "verbose_steps": v_steps,
        "lines_per_step": round(c_lines / max(1, v_steps), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None,
                    help="annotate every record with `structural_*` fields; "
                         "records are never dropped")
    ap.add_argument("--calibrate", action="store_true",
                    help="print feature distributions and exit — use this to "
                         "choose thresholds before applying any")
    ap.add_argument("--min_value_coverage", type=float, default=0.6)
    ap.add_argument("--max_invented_frac", type=float, default=1.0,
                    help="DIAGNOSTIC ONLY by default (1.0 = disabled). Audited "
                         "on the GLM corpus: the numerals it flags are notation "
                         "(`S^{-1}`, `{1,f,f²}`), step cross-references "
                         "(`from 4,5:`), or concrete counterexamples the rewrite "
                         "constructs to check itself — the last of which is work "
                         "we want. It also scores abstract problems worst simply "
                         "because they contain few numbers. Set < 1.0 only with "
                         "evidence from your own corpus.")
    ap.add_argument("--min_lines_per_step", type=float, default=0.0,
                    help="0 disables; set from --calibrate output")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    feats = [features(r) for r in rows]
    n = len(rows)
    print(f"[in] {n} records from {args.input}")

    def dist(key: str) -> str:
        vals = sorted(f[key] for f in feats)
        q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
        return (f"p10 {q(.10)}  p25 {q(.25)}  med {q(.50)}  "
                f"p75 {q(.75)}  p90 {q(.90)}")

    print(f"\n  source has branch      {sum(f['src_has_branch'] for f in feats)/n:.1%}"
          f"   -> kept {sum(f['branch_kept'] for f in feats)/n:.1%}")
    print(f"  source has verify      {sum(f['src_has_verify'] for f in feats)/n:.1%}"
          f"   -> kept {sum(f['verify_kept'] for f in feats)/n:.1%}")
    for k in ("value_coverage", "invented_frac", "lines_per_step", "compact_lines"):
        print(f"  {k:<18} {dist(k)}")

    if args.calibrate:
        print("\n[calibrate] no verdict applied; choose thresholds from the above")
        return 0

    n_pass = 0
    fails: Counter = Counter()
    out = []
    for r, f in zip(rows, feats):
        checks = {
            "branch_kept": f["branch_kept"],
            "verify_kept": f["verify_kept"],
            "value_coverage": f["value_coverage"] >= args.min_value_coverage,
            "no_invention": f["invented_frac"] <= args.max_invented_frac,
            "length_floor": f["lines_per_step"] >= args.min_lines_per_step,
        }
        ok = all(checks.values())
        n_pass += ok
        for k, v in checks.items():
            if not v:
                fails[k] += 1
        out.append({**r, **{f"structural_{k}": v for k, v in f.items()},
                    "structural_checks": checks, "structural_pass": ok})

    print(f"\n[gate] {n_pass}/{n} = {n_pass/n:.1%} pass")
    for k, v in fails.most_common():
        print(f"   fail {k:<16} {v}/{n} = {v/n:.1%}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_meta(args.output, {
            "builder": "scripts/structural_gate.py",
            "input": args.input, "n": n, "n_pass": n_pass,
            "pass_rate": round(n_pass / n, 4),
            "thresholds": {"min_value_coverage": args.min_value_coverage,
                           "max_invented_frac": args.max_invented_frac,
                           "min_lines_per_step": args.min_lines_per_step},
            "fail_counts": dict(fails),
            "note": ("Records are annotated, not dropped. Filter the Stage-A "
                     "conditioning corpus on structural_pass; do NOT filter the "
                     "Round-0/H_pivot corpus, which needs to stay representative "
                     "of what the student actually produces."),
        })
        print(f"[out] annotated -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
