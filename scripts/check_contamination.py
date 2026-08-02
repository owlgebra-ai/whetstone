"""Train-pool / eval-suite contamination check (packet P1 part 5).

Compares every train-pool prompt against every eval-suite prompt:

  1. **Exact** match of the normalized prompt (whitespace collapsed, case folded).
  2. **Near-duplicate**: word 8-gram Jaccard > 0.8 over the first 400 characters
     of the normalized prompt. An inverted 8-gram index over the (small) eval
     side keeps this linear in practice — only train rows that share at least one
     8-gram with some eval row are scored.

Hits are removed from the **train pool only** — eval suites are never modified.
With `--apply` the train JSONL is rewritten in place (the pre-removal file is
kept as `<name>.precontam.jsonl`) and `pool_stats.json` is refreshed.

Known risk pairs called out by the packet: GSM8K-train vs GSM8K-test overlap and
DeepMath vs MATH-500 overlap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from whetstone.poolutil import jaccard, match_key, ngrams, read_jsonl, write_jsonl


def _load_eval_suites(eval_dir: str, exclude: set[str]) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.jsonl"))):
        suite = os.path.splitext(os.path.basename(path))[0]
        if suite in exclude:
            print(f"[contam] skipping {suite}", flush=True)
            continue
        for r in read_jsonl(path):
            rows.append({
                "suite": r.get("suite", suite),
                "_uid": r.get("_uid", ""),
                "prompt": r.get("prompt", ""),
            })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/eval contamination check")
    ap.add_argument("--train", required=True, help="train pool JSONL")
    ap.add_argument("--eval_dir", required=True, help="dir of eval suite JSONLs")
    ap.add_argument("--out", default="", help="report JSON (default: alongside train)")
    ap.add_argument("--exclude_suites", default="",
                    help="comma list of suite file stems to skip")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--first_chars", type=int, default=400)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the train pool with hits removed")
    args = ap.parse_args(argv)

    train = read_jsonl(args.train)
    ev = _load_eval_suites(args.eval_dir, {s for s in args.exclude_suites.split(",") if s})
    print(f"[contam] train={len(train)} eval={len(ev)}", flush=True)

    # Eval-side indices.
    exact: dict[str, list[int]] = defaultdict(list)
    ev_grams: list[set[str]] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for i, e in enumerate(ev):
        exact[match_key(e["prompt"])].append(i)
        g = ngrams(e["prompt"], args.ngram, args.first_chars)
        ev_grams.append(g)
        for gram in g:
            inverted[gram].add(i)

    hits: list[dict] = []
    hit_uids: set[str] = set()
    for t in train:
        key = match_key(t["prompt"])
        matched = None
        if key in exact:
            j = exact[key][0]
            matched = {"kind": "exact", "score": 1.0, "eval": ev[j]}
        else:
            g = ngrams(t["prompt"], args.ngram, args.first_chars)
            cands: set[int] = set()
            for gram in g:
                cands |= inverted.get(gram, set())
            best, best_j = 0.0, -1
            for j in cands:
                s = jaccard(g, ev_grams[j])
                if s > best:
                    best, best_j = s, j
            if best > args.threshold:
                matched = {"kind": "near", "score": round(best, 4), "eval": ev[best_j]}
        if matched:
            hit_uids.add(t["_uid"])
            hits.append({
                "train_uid": t["_uid"],
                "train_source": t.get("source", ""),
                "train_prompt": t["prompt"][:200],
                "eval_uid": matched["eval"]["_uid"],
                "eval_suite": matched["eval"]["suite"],
                "eval_prompt": matched["eval"]["prompt"][:200],
                "kind": matched["kind"],
                "score": matched["score"],
            })

    by_suite = Counter(h["eval_suite"] for h in hits)
    by_kind = Counter(h["kind"] for h in hits)
    print(f"[contam] {len(hits)} contaminated train rows: "
          f"{dict(by_kind)} across suites {dict(by_suite)}", flush=True)

    report = {
        "train_file": args.train,
        "eval_dir": args.eval_dir,
        "params": {"threshold": args.threshold, "ngram": args.ngram,
                   "first_chars": args.first_chars},
        "n_train_before": len(train),
        "n_eval": len(ev),
        "n_hits": len(hits),
        "by_suite": dict(by_suite),
        "by_kind": dict(by_kind),
        "applied": bool(args.apply),
        "hits": hits,
    }

    if args.apply and hits:
        kept = [t for t in train if t["_uid"] not in hit_uids]
        # Pre-removal copy lives in a `_backup/` subdir, not next to the pool —
        # downstream code globs `*.jsonl` and must not pick up both versions.
        backup_dir = os.path.join(os.path.dirname(os.path.abspath(args.train)), "_backup")
        os.makedirs(backup_dir, exist_ok=True)
        pre = os.path.join(
            backup_dir,
            os.path.basename(args.train).replace(".jsonl", ".precontam.jsonl"))
        if not os.path.exists(pre):
            os.rename(args.train, pre)
        n = write_jsonl(args.train, kept)
        report["n_train_after"] = n
        report["pre_removal_file"] = pre
        print(f"[contam] rewrote {args.train}: {len(train)} → {n} "
              f"(pre-removal kept at {pre})", flush=True)

        stats_path = os.path.join(os.path.dirname(os.path.abspath(args.train)),
                                  "pool_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            stats.setdefault("train", {})
            stats["train"]["n"] = n
            stats["train"]["by_source"] = dict(Counter(r["source"] for r in kept))
            stats["train"]["by_level"] = {
                str(k): v for k, v in sorted(Counter(r["level"] for r in kept).items())
            }
            stats["contamination"] = {k: v for k, v in report.items() if k != "hits"}
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"[contam] refreshed {stats_path}", flush=True)
    else:
        report["n_train_after"] = len(train)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.train)), "contamination_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[contam] report → {out}", flush=True)


if __name__ == "__main__":
    main()
