"""Part 5 — the binding G_spike x branch-retention check (packet P4 §9).

Activity 006 decoupled the teacher from the student and made Stage A
generate-and-select from a frozen Qwen3-32B, because branch preservation is
scale-emergent (3.1 / 5.9 / 13.9% at 1.7B / 14B / 32B) and no prompting channel
transfers it. That decision buys branch-preserving compressions — **and the
Stage-A reward might select them straight back out**. `G_spike` rewards traces
the 1.7B finds *followable*; branch-preserving traces are longer and
structurally harder. Best-of-K under `R_acc * G_spike * G_budget` could
systematically prefer the 32B compressions that dropped their branches, undoing
the move.

This is the **binding** run of that check, and it has to happen here rather than
before F1. Branch-preserving traces carry more `case` / X / `chk` markers,
markers carry the style tax, and the tax inflates d_t — so measuring under pi_0
would report an anti-correlation that is really the accent Round 0 exists to
remove (activity 006 open item 1, sequencing correction).

Reports, for lambda=1 and beta in {5, 10}:
  * point-biserial correlation of G_spike against `structural_branch_kept`
  * the G_spike distributions split by branch_kept
  * the residual per-marker-class tax on the branch vocabulary under scorer_v1

No scipy on this box, so the correlation is computed directly and its
significance from the normal approximation to the t statistic — at n ~ 1,200 the
two agree to well past the decision point.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0 import (  # noqa: E402
    build_sequence, g_spike, load_jsonl, marker_class_ids, percentile,
)
from whetstone.round0_eval import score_positions  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"


def point_biserial(x: np.ndarray, y: np.ndarray):
    """(r, t, p_two_sided) for continuous ``x`` against binary ``y``."""
    y = y.astype(bool)
    n1, n0, n = int(y.sum()), int((~y).sum()), len(y)
    if n1 < 2 or n0 < 2:
        return math.nan, math.nan, math.nan
    s = x.std()
    if s == 0:
        return math.nan, math.nan, math.nan
    r = (x[y].mean() - x[~y].mean()) / s * math.sqrt(n1 * n0 / (n * n))
    if abs(r) >= 1:
        return r, math.inf, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = math.erfc(abs(t) / math.sqrt(2))     # normal approx, two-sided
    return r, t, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, help="scorer_v1 checkpoint")
    ap.add_argument("--corpus", default="/data/whetstone/corpora/seed_register_qwen32b")
    ap.add_argument("--text-file", default="compact_qwen32b.jsonl")
    ap.add_argument("--struct-file", default="gated_32b.jsonl")
    ap.add_argument("--tokenizer", default=MODEL)
    ap.add_argument("--betas", type=float, nargs="+", default=[5.0, 10.0])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/data/whetstone/runs/round0/gspike_branch_check.json")
    ap.add_argument("--assets", default="activity/assets/007")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    class_ids = marker_class_ids(tok)

    text = load_jsonl(Path(args.corpus) / args.text_file)
    struct = {r["_uid"]: r for r in load_jsonl(Path(args.corpus) / args.struct_file)}
    if args.limit:
        text = text[: args.limit]
    print(f"[load] {len(text)} 32B traces, {len(struct)} structural rows")

    model = AutoModelForCausalLM.from_pretrained(
        args.scorer, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    print(f"[scorer] {args.scorer}")

    rows: List[dict] = []
    branch_gaps: Dict[str, List[float]] = {k: [] for k in class_ids}
    for i, r in enumerate(text, 1):
        st = struct.get(r["_uid"])
        if st is None:
            continue
        seq = build_sequence(
            tok, uid=r["_uid"], problem=r["prompt"], think_body=r["compact_think"],
            answer=r["answer"], level=r.get("level", 0), require_gate=False,
        )
        if seq.masks.g != 1:
            continue
        pos = np.array([p for p in seq.think_positions if p > 0], dtype=np.int64)
        if pos.size < 5:
            continue
        s = score_positions(model, seq, pos)
        gaps = s["gap"].tolist()
        tok_ids = np.asarray(seq.ids)[pos]
        for name, ids in class_ids.items():
            sel = np.isin(tok_ids, list(ids))
            if sel.any():
                branch_gaps[name].extend(s["gap"][sel].tolist())

        row = {
            "_uid": r["_uid"], "level": r.get("level", 0),
            "branch_kept": bool(st["structural_branch_kept"]),
            "src_has_branch": bool(st["structural_src_has_branch"]),
            "verify_kept": bool(st["structural_verify_kept"]),
            "structural_pass": bool(st.get("structural_pass", False)),
            "think_tokens": int(pos.size),
            "mean_d": float(np.mean(gaps)), "p95_d": percentile(gaps, 95),
        }
        for b in args.betas:
            row[f"g_spike_b{b:g}"] = g_spike(gaps, lam=args.lam, beta=b)
        rows.append(row)
        if i % 200 == 0:
            print(f"  scored {i}/{len(text)}", flush=True)

    print(f"[scored] {len(rows)} traces")
    kept = np.array([r["branch_kept"] for r in rows])
    elig = np.array([r["src_has_branch"] for r in rows])
    print(f"[branch] kept {kept.sum()}/{len(kept)} = {kept.mean():.1%}; "
          f"source-eligible {elig.sum()}/{len(elig)} = {elig.mean():.1%}")

    report: Dict[str, object] = {
        "scorer": args.scorer, "n": len(rows), "lam": args.lam,
        "branch_kept_rate": float(kept.mean()),
        "src_has_branch_rate": float(elig.mean()),
    }

    for subset, mask, label in (
        ("all", np.ones(len(rows), bool), "all traces"),
        ("eligible", elig, "source-branching traces only"),
    ):
        sub = {}
        for b in args.betas:
            x = np.array([r[f"g_spike_b{b:g}"] for r in rows])[mask]
            y = kept[mask]
            r_pb, t, p = point_biserial(x, y)
            sub[f"beta{b:g}"] = {
                "r_pb": r_pb, "t": t, "p": p,
                "mean_kept": float(x[y].mean()) if y.any() else math.nan,
                "mean_dropped": float(x[~y].mean()) if (~y).any() else math.nan,
                "median_kept": float(np.median(x[y])) if y.any() else math.nan,
                "median_dropped": float(np.median(x[~y])) if (~y).any() else math.nan,
                "n_kept": int(y.sum()), "n_dropped": int((~y).sum()),
            }
            e = sub[f"beta{b:g}"]
            direction = ("NEGATIVE — G_spike selects AGAINST branch retention"
                         if r_pb < 0 else "non-negative — no penalty on branch retention")
            print(f"[{subset} β={b:g}] r_pb={r_pb:+.4f} (t={t:+.2f}, p={p:.3g})  "
                  f"G_spike kept={e['mean_kept']:.4f} vs dropped={e['mean_dropped']:.4f}  → {direction}")
        # d_t itself, so the correlation can be read as a tax rather than a coincidence
        for key in ("mean_d", "p95_d"):
            x = np.array([r[key] for r in rows])[mask]
            y = kept[mask]
            r_pb, t, p = point_biserial(x, y)
            sub[key] = {"r_pb": r_pb, "t": t, "p": p,
                        "mean_kept": float(x[y].mean()) if y.any() else math.nan,
                        "mean_dropped": float(x[~y].mean()) if (~y).any() else math.nan}
        report[subset] = sub

    report["marker_class_residual_tax"] = {
        name: {"mean_d": float(np.mean(v)), "p95_d": percentile(v, 95), "n": len(v)}
        for name, v in branch_gaps.items() if v
    }
    print("[residual tax under scorer_v1] " + "; ".join(
        f"{k}: mean_d={v['mean_d']:.3f} p95={v['p95_d']:.3f} (n={v['n']})"
        for k, v in report["marker_class_residual_tax"].items()))

    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[write] {args.out}")

    # --- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = Path(args.assets); assets.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, len(args.betas), figsize=(6.5 * len(args.betas), 5))
    ax = np.atleast_1d(ax)
    for j, b in enumerate(args.betas):
        x = np.array([r[f"g_spike_b{b:g}"] for r in rows])
        bins = np.linspace(float(x.min()), float(x.max()), 40)
        ax[j].hist(x[kept], bins=bins, alpha=.65, density=True,
                   label=f"branch kept (n={kept.sum()})", color="tab:green")
        ax[j].hist(x[~kept], bins=bins, alpha=.65, density=True,
                   label=f"branch dropped (n={(~kept).sum()})", color="tab:gray")
        rp = report["all"][f"beta{b:g}"]["r_pb"]
        ax[j].set_title(f"G_spike (λ=1, β={b:g}) vs branch retention\n"
                        f"32B corpus under scorer_v1 — r_pb = {rp:+.4f}")
        ax[j].set_xlabel("G_spike"); ax[j].set_ylabel("density")
        ax[j].legend(); ax[j].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(assets / "gspike_branch.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {assets}/gspike_branch.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
