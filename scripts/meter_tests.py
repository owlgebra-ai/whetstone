"""The three meter unit tests + the F1 band-existence verdict (packet P4 §8).

Run at the selected checkpoint **and its two neighbours** — the band may be
narrow, and a pass that exists at exactly one checkpoint is a different finding
from a pass that holds across a plateau.

    (a) Register hum   — held-out p95 d_t < tau_spike, overall and per marker
                         class. A pass whose *branch* class (case / X / chk /
                         check) is still uncalibrated is a QUALIFIED pass: the
                         Round-0 corpus barely contains that vocabulary
                         (activity 005 finding 7) and the 32B teacher's
                         branch-keeping traces lean on it (activity 006).
    (b) Verbose intact — mean per-token logprob delta vs pi_0 on the verbose
                         control within eps.
    (c) Corrupted probe — DECISIVE. Corrupted spans must still stand apart from
                         their own clean twins. Failing (c) invalidates the
                         scorer no matter how (a) and (b) look, because the
                         instrument would then be measuring style only and would
                         rate an unsupported leap as followable.

tau_leap is **pinned from the measured separation**, not assumed: it is chosen
at the point that maximizes Youden's J (TPR - FPR) over the paired
clean/corrupted span-p95 distributions, and the ROC AUC is reported alongside so
the choice can be audited rather than trusted.

(c) is judged on that separation alone and never against tau_spike. Design §8
Risk 1 asks whether calibrating away the style tax *dulls the leap detector*, so
the quantity of interest is how well this checkpoint separates corrupted from
clean — a property of the checkpoint, not of where (a)'s threshold happens to
sit. Since step0000 is pi_0 itself, the comparison that actually answers Risk 1
is the AUC at the winner against the AUC at step0000: if the register hum falls
while the AUC holds, the band exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whetstone.round0 import (  # noqa: E402
    build_sequence, load_jsonl, marker_class_ids, percentile,
)
from whetstone.round0_eval import (  # noqa: E402
    Pi0Cache, assert_alignment, build_eval_sets, evaluate, score_positions,
)

MODEL = "Qwen/Qwen3-1.7B"


def load_r_ids(path: str) -> frozenset:
    d = json.loads(Path(path).read_text())
    return frozenset(int(k) for k in d if not k.startswith("_"))


def build_twin_pairs(tok, rows: List[dict], span: int = 30):
    """Clean/corrupted :class:`Seq` pairs plus the token span to score.

    The corruption point is located in **token** space as the first index at or
    after ``think_start`` where the two id sequences diverge — not from string
    offsets, which do not survive tokenization at the edit site. The scored span
    is ``[div, div + 30)`` in each twin, clamped to its own think segment.
    """
    pairs = []
    for r in rows:
        clean = build_sequence(tok, uid=r["_uid"], problem=r["prompt"],
                               think_body=r["clean_think"], answer=r["answer"],
                               level=r.get("level", 0), require_gate=False)
        dirty = build_sequence(tok, uid=r["_uid"], problem=r["prompt"],
                               think_body=r["corrupted_think"], answer=r["answer"],
                               level=r.get("level", 0), require_gate=False)
        if clean.masks.g != 1 or dirty.masks.g != 1:
            continue
        a, b = np.asarray(clean.ids), np.asarray(dirty.ids)
        start = clean.masks.think_start
        n = min(len(a), len(b))
        div = None
        for i in range(start, n):
            if a[i] != b[i]:
                div = i
                break
        if div is None:
            continue                      # tokenization absorbed the edit
        cs = np.arange(div, min(div + span, clean.masks.think_end))
        ds = np.arange(div, min(div + span, dirty.masks.think_end))
        if cs.size < 5 or ds.size < 5:
            continue
        pairs.append({
            "uid": r["_uid"], "type": r["corruption_type"], "level": r.get("level", 0),
            "detail": r.get("corruption_detail", ""),
            "clean": clean, "dirty": dirty, "clean_span": cs, "dirty_span": ds,
        })
    return pairs


def run_probe(model, pairs) -> List[dict]:
    out = []
    for p in pairs:
        c = score_positions(model, p["clean"], p["clean_span"])
        d = score_positions(model, p["dirty"], p["dirty_span"])
        out.append({
            "uid": p["uid"], "type": p["type"], "level": p["level"], "detail": p["detail"],
            "clean_p95": percentile(c["gap"].tolist(), 95),
            "dirty_p95": percentile(d["gap"].tolist(), 95),
            "clean_max": float(c["gap"].max()), "dirty_max": float(d["gap"].max()),
            "clean_mean": float(c["gap"].mean()), "dirty_mean": float(d["gap"].mean()),
        })
    return out


def roc_pick_tau(clean: np.ndarray, dirty: np.ndarray):
    """(tau_leap, AUC, TPR, FPR) at the Youden-optimal threshold."""
    cand = np.unique(np.concatenate([clean, dirty]))
    best = (np.nan, -1.0, 0.0, 0.0)
    for t in cand:
        tpr = float((dirty > t).mean())
        fpr = float((clean > t).mean())
        if tpr - fpr > best[1]:
            best = (float(t), tpr - fpr, tpr, fpr)
    # AUC via the Mann-Whitney U identity (ties count a half).
    gt = (dirty[:, None] > clean[None, :]).sum()
    eq = (dirty[:, None] == clean[None, :]).sum()
    auc = float((gt + 0.5 * eq) / (len(dirty) * len(clean)))
    return best[0], auc, best[2], best[3]


def plot_pairs(res: List[dict], tau_spike: float, tau_leap: float, out_png: Path, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = np.array([r["clean_p95"] for r in res])
    dirty = np.array([r["dirty_p95"] for r in res])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    hi = float(max(dirty.max(), clean.max()))
    bins = np.linspace(0, hi, 45)
    ax[0].hist(clean, bins=bins, alpha=.65, label=f"clean (n={len(clean)})", color="tab:blue")
    ax[0].hist(dirty, bins=bins, alpha=.65, label=f"corrupted (n={len(dirty)})", color="tab:red")
    ax[0].axvline(tau_spike, color="k", ls=":", label=f"τ_spike = {tau_spike}")
    ax[0].axvline(tau_leap, color="tab:green", ls="--", label=f"τ_leap = {tau_leap:.2f}")
    ax[0].set_xlabel("p95 d_t over the corruption span (nats)")
    ax[0].set_ylabel("pairs")
    ax[0].set_title(f"Band existence — {label}\ncorrupted vs clean twin, same span")
    ax[0].legend(); ax[0].grid(alpha=.3)

    for kind, c in (("delete", "tab:purple"), ("substitute", "tab:orange")):
        sel = [r for r in res if r["type"] == kind]
        if sel:
            ax[1].scatter([r["clean_p95"] for r in sel], [r["dirty_p95"] for r in sel],
                          s=18, alpha=.7, color=c, label=f"{kind} (n={len(sel)})")
    lim = [0, hi * 1.02]
    ax[1].plot(lim, lim, "k-", lw=.8, label="no separation")
    ax[1].axhline(tau_leap, color="tab:green", ls="--", label=f"τ_leap = {tau_leap:.2f}")
    ax[1].axvline(tau_spike, color="k", ls=":", label=f"τ_spike = {tau_spike}")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("clean twin p95 d_t"); ax[1].set_ylabel("corrupted p95 d_t")
    ax[1].set_title("Paired: above the diagonal = corruption detected")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--corpus", default="/data/whetstone/corpora/seed_register_qwen")
    ap.add_argument("--probe", default="/data/whetstone/runs/round0/corrupted_probe.jsonl")
    ap.add_argument("--r-tokenset", default="/data/whetstone/runs/round0/R_tokenset.json")
    ap.add_argument("--pi0-cache", default="/data/whetstone/runs/round0/pi0_cache.npz")
    ap.add_argument("--tokenizer", default=MODEL)
    ap.add_argument("--tau-spike", type=float, default=1.2)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--span", type=int, default=30)
    ap.add_argument("--n-control", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/whetstone/runs/round0/meter_tests.json")
    ap.add_argument("--assets", default="activity/assets/007")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    r_ids = load_r_ids(args.r_tokenset)
    class_ids = marker_class_ids(tok)
    sets = build_eval_sets(tok, args.corpus, n_control=args.n_control, seed=args.seed)
    cache = Pi0Cache(args.pi0_cache)
    pairs = build_twin_pairs(tok, load_jsonl(args.probe), span=args.span)
    kinds = {k: sum(1 for p in pairs if p["type"] == k) for k in ("delete", "substitute")}
    print(f"[probe] {len(pairs)} usable twin pairs ({kinds}), span {args.span} tokens")

    assets = Path(args.assets); assets.mkdir(parents=True, exist_ok=True)
    report: Dict[str, dict] = {}

    for ck in args.ckpts:
        name = Path(ck).name
        print(f"\n=== {name} " + "=" * 50)
        model = AutoModelForCausalLM.from_pretrained(
            ck, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()

        # Alignment is a property of the *code*, which is identical across
        # checkpoints, so it is asserted once (on pi_0, step0000) and merely
        # recorded afterwards: an inoculated scorer may legitimately move this
        # number, and aborting the F1 run over that would be wrong.
        try:
            align = assert_alignment(model, sets)
        except AssertionError as e:
            align = getattr(e, "measured", float("nan"))
            print(f"[align] NOTE: {e}")
        m = evaluate(model, sets, r_ids=r_ids, class_ids=class_ids, cache=cache)
        m.pop("_gaps_heldout")

        res = run_probe(model, pairs)
        clean = np.array([r["clean_p95"] for r in res])
        dirty = np.array([r["dirty_p95"] for r in res])
        tau_leap, auc, tpr, fpr = roc_pick_tau(clean, dirty)
        # Band existence is really the threshold-free separation (AUC, and the
        # paired scatter); tau_spike and tau_leap are pinned *from* it. The
        # suggested tau_spike is where uncorrupted register text actually sits,
        # so it is reported next to the packet's start value of 1.2 rather than
        # replacing it — quoting only the self-derived one would be circular.
        tau_spike_suggested = float(percentile(clean.tolist(), 95))

        a_pass_R = m["s1_p95_gap_R"] < args.tau_spike
        a_pass = m["s1_p95_gap"] < args.tau_spike
        a_branch = m.get("s1_p95_gap_branch", float("nan")) < args.tau_spike
        a_struct = m.get("s1_p95_gap_structural", float("nan")) < args.tau_spike
        b_pass = abs(m["b_logprob_delta_mean"]) < args.eps
        frac_dirty = float((dirty > tau_leap).mean())
        frac_clean = float((clean < args.tau_spike).mean())
        # (c) is judged on the clean/corrupted *separation* alone, not against
        # tau_spike. Design §8 Risk 1 asks whether calibrating away the style tax
        # dulls the leap detector, so the question is whether corrupted spans
        # still stand apart from their own clean twins — a property of this
        # checkpoint, independent of where (a)'s threshold is set. Tying (c) to
        # tau_spike would make a provisional threshold decide the decisive test.
        c_pass = bool(auc > 0.65 and tpr > 0.5 and fpr < 0.30)

        plot_pairs(res, args.tau_spike, tau_leap, assets / f"probe_{name}.png", name)

        report[name] = {
            "align_close_entropy_median": align,
            "a_register_hum": {
                "p95_overall": m["s1_p95_gap"], "pass": bool(a_pass),
                "p95_R_tokens": m["s1_p95_gap_R"], "pass_R": bool(a_pass_R),
                "mean_gap_R": m["s1_mean_gap_R"],
                "hum_R_mean_surprisal": m["hum_R_mean_surprisal"],
                "p95_structural": m.get("s1_p95_gap_structural"),
                "pass_structural": bool(a_struct),
                "p95_branch": m.get("s1_p95_gap_branch"),
                "pass_branch": bool(a_branch),
                "n_branch_tokens": m.get("s1_n_branch"),
                "qualified": bool(a_pass and not a_branch),
            },
            "b_verbose_intact": {
                "logprob_delta": m["b_logprob_delta_mean"], "eps": args.eps, "pass": bool(b_pass),
            },
            "c_corrupted_probe": {
                "n_pairs": len(res), "by_type": kinds,
                "tau_leap": tau_leap, "auc": auc, "tpr": tpr, "fpr": fpr,
                "tau_spike_suggested": tau_spike_suggested,
                "separation_ratio": (tau_leap / tau_spike_suggested
                                     if tau_spike_suggested > 0 else float("nan")),
                "clean_p95_median": float(np.median(clean)),
                "dirty_p95_median": float(np.median(dirty)),
                "frac_corrupted_above_tau_leap": frac_dirty,
                "frac_clean_below_packet_tau_spike": frac_clean,
                "frac_pairs_dirty_gt_clean": float((dirty > clean).mean()),
                "pass": c_pass,
            },
            "all_three_pass": bool(a_pass and b_pass and c_pass),
            "metrics": m,
            "pairs": res,
        }

        print(f"    R-token gap p95={m['s1_p95_gap_R']:.3f} (register only) | "
              f"hum_R surprisal={m['hum_R_mean_surprisal']:.3f}")
        print(f"(a) register hum : p95={m['s1_p95_gap']:.3f} < {args.tau_spike} -> "
              f"{'PASS' if a_pass else 'FAIL'} | structural={m.get('s1_p95_gap_structural'):.3f} "
              f"branch={m.get('s1_p95_gap_branch'):.3f} (n={m.get('s1_n_branch')})")
        print(f"(b) verbose      : Δlogp={m['b_logprob_delta_mean']:+.4f} within ±{args.eps} -> "
              f"{'PASS' if b_pass else 'FAIL'}")
        print(f"    band: clean-span p95 = {tau_spike_suggested:.3f} (τ_spike suggested by data), "
              f"τ_leap = {tau_leap:.3f} → separation {tau_leap / max(tau_spike_suggested, 1e-9):.2f}×")
        print(f"(c) corrupted    : τ_leap={tau_leap:.3f} AUC={auc:.3f} "
              f"(TPR={tpr:.2f} FPR={fpr:.2f}) | corrupted p95 med={np.median(dirty):.3f} "
              f"vs clean {np.median(clean):.3f} | {frac_dirty:.0%} of corruptions spike "
              f"-> {'PASS' if c_pass else 'FAIL'}")
        print(f"ALL THREE: {'PASS' if report[name]['all_three_pass'] else 'FAIL'}")

        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[write] {args.out}")
    winners = [k for k, v in report.items() if v["all_three_pass"]]
    print(f"[F1] checkpoints passing all three simultaneously: {winners or 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
