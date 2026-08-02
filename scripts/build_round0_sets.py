"""Round-0 measurement sets + Δlogp distribution (packet P3 Part 3).

Writes the four files P4 consumes, and writes them **once**: P4 is forbidden
from re-splitting, because a split recomputed under a different seed would put
probe traces into training and quietly invalidate the corrupted-trace probe —
the one unit test that can invalidate the scorer on its own (design §2).

Splits of the seed register corpus (fixed seed, proportional level strata):

  * ``train`` (~80%) — Round-0 inoculation training set;
  * ``heldout_register`` (~10%) — Round-0 stop criterion S1 and meter unit
    test (a). Never trained on;
  * ``probe_pool`` (~10%) — reserved for the corrupted-trace probe. These must
    never appear in training in any round.

Plus the **verbose control set** (~200 native verbose traces, Round 0's KL
drift gauge and unit test (b)). It is drawn from Part 1's verified harvest with
every ``_uid`` in the register corpus excluded: the control set exists to
measure whether inoculation moved the model on *ordinary* text, so overlap with
the traces the scorer was inoculated on would mask exactly the drift S2 is
watching for.

Also emits the Δlogp histogram (packet deliverable) over whatever the scorer
annotated, with the pass/fail threshold marked.

Usage::

    python scripts/build_round0_sets.py \\
        --register /data/whetstone/corpora/seed_register/seed_register.jsonl \\
        --verified /data/whetstone/corpora/seed/seed_verified.jsonl \\
        --out_dir  /data/whetstone/corpora/seed_register \\
        --scored   /data/whetstone/corpora/seed_register/seed_register_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.poolutil import read_jsonl, stratified_sample, write_jsonl, write_meta

SPLITS = (("train", 0.80), ("heldout_register", 0.10), ("probe_pool", 0.10))


def _level(r: dict) -> str:
    return str(r.get("level", "_"))


def _plot_delta(scored: list[dict], threshold: float, path: str) -> dict:
    deltas = [r["delta_logp"] for r in scored
              if isinstance(r.get("delta_logp"), (int, float))
              and r["delta_logp"] != float("-inf")]
    n_inf = sum(1 for r in scored if r.get("delta_logp") == float("-inf"))
    if not deltas:
        return {"n": 0}
    s = sorted(deltas)
    stats = {
        "n": len(s), "n_neg_inf": n_inf,
        "min": s[0], "p05": s[int(0.05 * len(s))], "p25": s[len(s) // 4],
        "median": s[len(s) // 2], "p75": s[3 * len(s) // 4],
        "p95": s[int(0.95 * len(s))], "max": s[-1],
        "mean": sum(s) / len(s),
        "pass_rate": sum(1 for d in s if d > threshold) / len(s),
        "threshold": threshold,
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(s, bins=60, color="#4878a8", edgecolor="none")
        ax.axvline(threshold, color="#c04040", lw=1.5,
                   label=f"threshold {threshold:g} (pass {stats['pass_rate']:.1%})")
        ax.set_xlabel(r"$\Delta\log p = \log P(a^*\mid q,\ \mathrm{compact}) - "
                      r"\log P(a^*\mid q)$")
        ax.set_ylabel("traces")
        ax.set_title(f"Seed register corpus — sufficiency gate (n={len(s)})")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        stats["plot"] = path
    except Exception as exc:                                   # noqa: BLE001
        stats["plot_error"] = f"{type(exc).__name__}: {exc}"
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", required=True,
                    help="accepted seed register corpus (post-Δlogp)")
    ap.add_argument("--verified", required=True,
                    help="Part 1 seed_verified.jsonl, source of the control set")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scored", default=None,
                    help="Δlogp-annotated corpus (all rows) for the histogram")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Δlogp pass threshold (v1 §3.6: delta > 0)")
    ap.add_argument("--n_control", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing splits (refuses by default — P4 "
                         "must see a stable split)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {name: os.path.join(args.out_dir, f"{name}.jsonl") for name, _ in SPLITS}
    paths["verbose_control"] = os.path.join(args.out_dir, "verbose_control.jsonl")
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing and not args.force:
        raise SystemExit(
            "[round0] refusing to overwrite existing splits:\n  "
            + "\n  ".join(existing)
            + "\nThese are P4's fixed inputs; re-splitting silently moves probe "
              "traces into training. Pass --force only if you mean it.")

    reg = read_jsonl(args.register)
    rng = random.Random(args.seed)
    print(f"[in] {len(reg)} accepted register traces")

    # Disjoint, level-stratified splits: draw the two small ones from the pool,
    # remainder is train. stratified_sample keeps the level histogram in each.
    remaining = list(reg)
    out: dict[str, list] = {}
    for name, frac in SPLITS[1:]:
        k = max(1, int(round(frac * len(reg)))) if reg else 0
        pick = stratified_sample(remaining, _level, k, rng)
        picked = {id(r) for r in pick}
        remaining = [r for r in remaining if id(r) not in picked]
        out[name] = pick
    out["train"] = remaining

    reg_uids = {r["_uid"] for r in reg}
    verified = read_jsonl(args.verified)
    # One row per problem, and nothing that fed the register corpus.
    seen_uid: set[str] = set()
    control_pool = []
    for r in verified:
        uid = r["_uid"]
        if uid in reg_uids or uid in seen_uid:
            continue
        seen_uid.add(uid)
        control_pool.append(r)
    control = stratified_sample(control_pool, _level, args.n_control, rng)
    out["verbose_control"] = control
    print(f"[control] {len(control_pool)} candidate problems disjoint from the "
          f"register corpus -> {len(control)} selected")

    meta_splits = {}
    for name, rows in out.items():
        rows.sort(key=lambda r: (r["_uid"], r.get("src_candidate_idx", 0)))
        write_jsonl(paths[name], rows)
        lv = Counter(_level(r) for r in rows)
        meta_splits[name] = {"n": len(rows), "path": paths[name],
                             "by_level": {k: lv[k] for k in sorted(lv)}}
        print(f"[out] {name}: {len(rows)} -> {paths[name]}")

    delta_stats = {}
    if args.scored and os.path.exists(args.scored):
        delta_stats = _plot_delta(read_jsonl(args.scored), args.threshold,
                                  os.path.join(args.out_dir, "delta_logp_hist.png"))
        print(f"[Δlogp] {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in delta_stats.items()})}")

    write_meta(os.path.join(args.out_dir, "round0_sets.jsonl"), {
        "builder": "scripts/build_round0_sets.py",
        "packet": "P3 Part 3",
        "register_corpus": args.register,
        "verified_harvest": args.verified,
        "seed": args.seed,
        "splits": meta_splits,
        "delta_logp": delta_stats,
        "invariant": ("P4 must not re-split. probe_pool is reserved for the "
                      "corrupted-trace probe and must never be trained on."),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
