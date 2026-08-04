"""Stage-A dashboards — the F2d panel (packet P5 Part 6).

Design §7 makes dashboards a first-class deliverable of each stage, built
alongside the training code rather than after it. For a *frozen* teacher there
are no training dynamics to plot, so the panel answers the questions that do
apply to a generate-and-select stage:

1. **Symbol density**, selected vs raw — did selection concentrate the register,
   and does the corpus sit near the 32B's own 2.10 marks/100-char baseline
   (activity 006) rather than drifting into caveman or back into prose?
2. **Segment lengths, think and answer as separate panels.** Never one combined
   number — that is how drift hides (CLAUDE.md invariant, and activity 007
   finding 4 is a live example of a targeted degradation invisible in a mean).
3. **Per-level coverage and the unserved rate** — where the corpus is thin, and
   whether thinness tracks difficulty (it does; that is Stage-C rescue's
   clientele, not a silent loss).
4. **Selection-reason histogram** — what the lexicographic rule actually fired
   on, so "prefer verify_kept" can be read as a frequency rather than assumed.
5. **Structural retention, raw vs selected, per level** — the F2b panel: does
   best-of-K recover the verification and branch retention that a single draft
   does not have?

Usage::

    python scripts/stagea_dashboards.py \\
        --selected /data/whetstone/corpora/stagea_selected/selected.jsonl \\
        --drafts   /data/whetstone/corpora/stagea_raw/drafts.jsonl \\
        --scores   /data/whetstone/corpora/stagea_raw/scores.jsonl \\
        --unserved /data/whetstone/corpora/stagea_selected/unserved_uids.json \\
        --outdir   activity/assets/008
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whetstone.round0 import MARKER_CLASSES, percentile             # noqa: E402

ALL_MARKERS = tuple(m for cls in MARKER_CLASSES.values() for m in cls)
#: activity 006, single-draft rates for this same teacher — every "did selection
#: help?" panel is read against these, not against zero.
RAW_32B_MARK_100CH = 2.10
RAW_32B_VERIFY = 70.6
RAW_32B_BRANCH = 13.9


def density(think: str) -> float:
    return 100.0 * sum(think.count(m) for m in ALL_MARKERS) / max(1, len(think))


def load(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected",
                    default="/data/whetstone/corpora/stagea_selected/selected.jsonl")
    ap.add_argument("--drafts", default="/data/whetstone/corpora/stagea_raw/drafts.jsonl")
    ap.add_argument("--scores", default="/data/whetstone/corpora/stagea_raw/scores.jsonl")
    ap.add_argument("--unserved",
                    default="/data/whetstone/corpora/stagea_selected/unserved_uids.json")
    ap.add_argument("--subset", default="/data/whetstone/corpora/stagea/subset_stagea.jsonl")
    ap.add_argument("--outdir", default="activity/assets/008")
    ap.add_argument("--tag", default="stagea")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sel = load(args.selected)
    drafts = load(args.drafts)
    scores = {(s["_uid"], s["candidate_idx"]): s for s in load(args.scores)}
    unserved = json.load(open(args.unserved)) if os.path.exists(args.unserved) else {}
    subset = load(args.subset)
    level_of = {r["_uid"]: r["level"] for r in subset}

    surv = [d for d in drafts if d.get("reject_reason") is None]
    levels = sorted({r.get("level") for r in sel if r.get("level") is not None})

    plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5})

    # ---- panel 1: symbol density, raw vs selected ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    d_raw = [density(d.get("compact_think", "")) for d in surv]
    d_sel = [density(r.get("compact_think", "")) for r in sel]
    ax = axes[0]
    bins = [i * 0.25 for i in range(0, 33)]
    ax.hist(d_raw, bins=bins, alpha=.55, label=f"raw drafts (n={len(d_raw)})",
            color="#8892b0")
    ax.hist(d_sel, bins=bins, alpha=.75, label=f"selected (n={len(d_sel)})",
            color="#2a6f97")
    ax.axvline(RAW_32B_MARK_100CH, color="crimson", ls="--", lw=1.2,
               label=f"32B raw baseline {RAW_32B_MARK_100CH}")
    ax.set_xlabel("register markers / 100 think chars")
    ax.set_ylabel("drafts")
    ax.set_title("symbol density")
    ax.legend(fontsize=6.5)

    # ---- panel 2/3: segment lengths, ALWAYS separate ------------------
    for ax, key, name in ((axes[1], "scored_think_tokens", "think"),
                          (axes[2], "scored_answer_tokens", "answer")):
        vals = [r.get(key) or 0 for r in sel]
        vals = [v for v in vals if v > 0]
        ax.hist(vals, bins=50, color="#2a6f97", alpha=.8)
        med, p25, p75 = (percentile(vals, 50), percentile(vals, 25),
                         percentile(vals, 75))
        ax.axvline(med, color="crimson", lw=1.2, label=f"median {med:.0f}")
        ax.axvspan(p25, p75, color="crimson", alpha=.10,
                   label=f"IQR [{p25:.0f}, {p75:.0f}]")
        if name == "think":
            ax.axvline(600, color="k", ls=":", lw=1.1, label="B_target 600")
        ax.set_xlabel(f"{name} tokens")
        ax.set_title(f"{name} length (selected)")
        ax.legend(fontsize=6.5)
    fig.suptitle("Stage A — register density and segment lengths "
                 "(think and answer never combined)", fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, f"{args.tag}_density_lengths.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ---- panel 4: coverage + unserved per level -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    served = Counter(r["_uid"] for r in sel)
    prob_by_level = Counter(level_of.get(u) for u in served)
    unserved_by_level = Counter(level_of.get(u) for u in unserved)
    attempted = Counter(level_of.get(d["_uid"]) for d in
                        {d["_uid"]: d for d in drafts}.values())
    ax = axes[0]
    xs = list(range(len(levels)))
    ax.bar(xs, [prob_by_level.get(l, 0) for l in levels], color="#2a6f97",
           label="served")
    ax.bar(xs, [unserved_by_level.get(l, 0) for l in levels],
           bottom=[prob_by_level.get(l, 0) for l in levels], color="#c1121f",
           label="unserved")
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("level"); ax.set_ylabel("problems")
    ax.set_title("per-level coverage"); ax.legend(fontsize=6.5)

    ax = axes[1]
    rate = [100 * unserved_by_level.get(l, 0) / max(1, attempted.get(l, 0))
            for l in levels]
    ax.bar(xs, rate, color="#c1121f")
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("level"); ax.set_ylabel("% unserved")
    ax.set_title("unserved rate per level")

    ax = axes[2]
    reasons = Counter(r.get("selection_reason", "?") for r in sel)
    lab = [k for k, _ in reasons.most_common(8)][::-1]
    ax.barh(range(len(lab)), [reasons[k] for k in lab], color="#4a7c59")
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels([k.replace("gspike_gbudget", "G")[:38] for k in lab],
                       fontsize=6)
    ax.set_xlabel("traces"); ax.set_title("selection reasons")
    fig.suptitle("Stage A — coverage, unserved, and what the selection rule fired on",
                 fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(args.outdir, f"{args.tag}_coverage_selection.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # ---- panel 5: structural retention raw vs selected, per level -----
    def rates(records, key_elig, key_kept):
        num, den = defaultdict(int), defaultdict(int)
        for r, s in records:
            if s.get(key_elig):
                den[r.get("level")] += 1
                num[r.get("level")] += bool(s.get(key_kept))
        return {l: 100 * num[l] / den[l] for l in den if den[l]}, dict(den)

    raw_pairs = [(d, scores[(d["_uid"], d["candidate_idx"])]) for d in surv
                 if (d["_uid"], d["candidate_idx"]) in scores]
    sel_pairs = [(r, r) for r in sel]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, elig, kept, base, name in (
            (axes[0], "src_has_verify", "verify_kept", RAW_32B_VERIFY, "verify"),
            (axes[1], "src_has_branch", "branch_kept", RAW_32B_BRANCH, "branch")):
        r_rate, r_n = rates(raw_pairs, elig, kept)
        s_rate, _ = rates(sel_pairs, elig, kept)
        ls = [l for l in levels if l in r_rate or l in s_rate]
        xs = list(range(len(ls)))
        ax.bar([x - .2 for x in xs], [r_rate.get(l, 0) for l in ls], width=.4,
               color="#8892b0", label="raw drafts")
        ax.bar([x + .2 for x in xs], [s_rate.get(l, 0) for l in ls], width=.4,
               color="#2a6f97", label="selected")
        ax.axhline(base, color="crimson", ls="--", lw=1.1,
                   label=f"32B single-draft {base}%")
        ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in ls])
        ax.set_xlabel("level"); ax.set_ylabel(f"% {kept}")
        ax.set_title(f"{name} retention (eligible drafts only)")
        ax.legend(fontsize=6.5)
    fig.suptitle("Stage A — F2b: does best-of-K recover structure a single draft lacks?",
                 fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(args.outdir, f"{args.tag}_structural.png")
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    for p in (p1, p2, p3):
        print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
