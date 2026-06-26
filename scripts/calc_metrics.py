"""Aggregate WHETSTONE eval summaries into a comparison table.

Reads one or more `summary__<model>.json` files (written by run_eval.py) and
emits a Markdown table comparing strict@1 accuracy, pass@K, and tokens-per-correct
across models and suites. Identifies the Pareto frontier over accuracy and
tokens-per-correct (§7.9 endpoint selection).

Usage:
    python scripts/calc_metrics.py \\
        --summaries out/eval/baseline/summary__gemma-4-E4B-it.json \\
                    out/eval/v4b_spike/summary__audit_passed_v1.json \\
                    out/eval/dapo_p2/summary__dapo_phase2_merged_30.json \\
        --out_md reports/endpoint_table.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt(x, prec: int = 1, pct: bool = False) -> str:
    if x is None:
        return "—"
    if pct:
        return f"{x * 100:.{prec}f}%"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def _pareto_frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Pareto-maximize (acc, -tokens_per_correct). Returns list of
    (acc, tokens_per_correct, model) on the frontier."""
    frontier: list[tuple[float, float, str]] = []
    for acc, tpc, model in sorted(points, key=lambda p: (-p[0], p[1])):
        if frontier and tpc >= frontier[-1][1]:
            continue
        frontier.append((acc, tpc, model))
    return frontier


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate WHETSTONE eval summaries")
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="One or more summary__<model>.json files")
    ap.add_argument("--out_md", default=None, help="Write a Markdown table here")
    ap.add_argument("--suite", default=None,
                    help="Optional: restrict to one suite (e.g. aime24)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = []
    for p in args.summaries:
        if not os.path.exists(p):
            print(f"skip missing: {p}", file=sys.stderr)
            continue
        runs.append(_load(p))

    if not runs:
        raise SystemExit("no summaries loaded")

    # Build suite -> {model: summary}
    suite_model: dict[str, dict[str, dict]] = {}
    for r in runs:
        model = r.get("model", Path(r.get("suites", [{}])[0].get("model", "?")).stem)
        for s in r.get("suites", []):
            suite_model.setdefault(s["suite"], {})[model] = s

    suites = sorted(suite_model.keys())
    if args.suite:
        suites = [s for s in suites if s == args.suite]
    models = sorted({m for sm in suite_model.values() for m in sm.keys()})

    lines: list[str] = []
    title = f"# WHETSTONE eval — suite: {args.suite}" if args.suite else "# WHETSTONE eval"
    lines.append(title)
    lines.append("")

    # Accuracy table
    lines.append("## Strict accuracy @1")
    header = "| Suite | " + " | ".join(models) + " |"
    sep = "|---|" + "|".join(["---"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)
    for suite in suites:
        row = [suite]
        for m in models:
            s = suite_model.get(suite, {}).get(m)
            row.append(_fmt(s["strict_accuracy_at1"] if s else None, 1, pct=True))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Tokens-per-correct table
    lines.append("## Tokens per correct (lower = more efficient)")
    lines.append(header)
    lines.append(sep)
    for suite in suites:
        row = [suite]
        for m in models:
            s = suite_model.get(suite, {}).get(m)
            row.append(_fmt(s["tokens_per_correct"] if s else None, 0))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Pareto frontier over the requested suite (or all suites averaged)
    lines.append("## Pareto frontier (acc ↑, tokens/correct ↓)")
    parelo_points: list[tuple[float, float, str]] = []
    for m in models:
        accs = []
        tpcs = []
        for suite in suites:
            s = suite_model.get(suite, {}).get(m)
            if s and s.get("strict_accuracy_at1") is not None:
                accs.append(s["strict_accuracy_at1"])
                if s.get("tokens_per_correct") is not None:
                    tpcs.append(s["tokens_per_correct"])
        if accs and tpcs:
            parelo_points.append((sum(accs) / len(accs), sum(tpcs) / len(tpcs), m))
    frontier = _pareto_frontier(parelo_points)
    frontier_models = {p[2] for p in frontier}
    for acc, tpc, m in sorted(parelo_points, key=lambda p: -p[0]):
        mark = " ★" if m in frontier_models else ""
        lines.append(f"- **{m}**{mark}: acc={acc * 100:.1f}%, tokens/correct={tpc:.0f}")
    lines.append("")

    md = "\n".join(lines)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(md)
        print(f"[metrics] wrote {args.out_md}", flush=True)
    else:
        print(md)


if __name__ == "__main__":
    main()
