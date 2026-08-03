"""Entropy audit of the starting checkpoint (design §1 precondition 1, §12.3).

Answers one question before any training happens: **does this checkpoint arrive
entropy-collapsed?** The answer selects SED's mode in Stage B — *preservation*
(healthy bimodal histogram, `Δ_max = 0.5`) vs *restoration* (arrives collapsed,
`Δ_max = 0.7`, design §4.2 / §12.6) — and the histogram this produces is reused
three more times:

  * as the **baseline** for the Stage-B sanity gate ("median entropy not below
    the audit baseline", design §4.3);
  * as the **entropy floor** for the Round-0 S3 stop (design §2);
  * re-run in ``--traces`` mode on the P3 seed register corpus to pin
    **H_pivot** = 80th percentile of the *compact-register* histogram. H_pivot
    is NOT set by this run over native traces (packet P2 Part 2 gotcha 3).

Method (design §12.3): sample one rollout per problem for ~200 level-stratified
pool problems, teacher-force each (prompt, rollout) back through the model, and
take the entropy of the **top-512 logits** per position (softmax over the 512
only — CurioSFT's convention). Report think and answer segments separately,
using the token-level parser in :mod:`whetstone.segments`.

Two alignment/memory rules this script is built around (packet P2 Part 2):

1. **Never materialize full-vocab logits for a whole sequence.** 151936 vocab ×
   16k positions × 4 bytes ≈ 10 GB. The forward runs through the base
   transformer only; ``lm_head`` is applied in position chunks, ``topk(512)``
   taken in bf16, softmax done in fp32 on the 512, entropy accumulated, chunk
   discarded.
2. **Entropy at position t comes from the logits at t−1** (next-token
   prediction). Off-by-one here mis-attributes tokens across the ``</think>``
   boundary. The ``--sanity`` output prints the entropy of the boundary tokens
   themselves as a check: ``</think>`` following a terse line should be low.

Stages are resumable; each writes its own artifact and is skipped if present
(``--force`` to redo).

Usage
-----
Full audit (generate + score + aggregate)::

    python scripts/entropy_audit.py \
        --pool /data/whetstone/data/pool/val_2k.jsonl \
        --out_dir /data/whetstone/runs/entropy_audit \
        --n 200 --compare_model Qwen/Qwen3-1.7B-Base

Audit pre-existing traces (P3 re-runs this to pin H_pivot)::

    python scripts/entropy_audit.py \
        --traces /data/whetstone/corpora/seed_register/seed_register.jsonl \
        --completion_field completion \
        --out_dir /data/whetstone/runs/entropy_audit_compact
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from whetstone.poolutil import read_jsonl, stratified_sample, write_jsonl
from whetstone.segments import (
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    blank_token_ids_for,
    parse_segments,
)

TOPK = 512                    # CurioSFT convention; also the SED bisection top-k
SEG_OTHER, SEG_THINK, SEG_ANSWER = 0, 1, 2
HIST_MAX = float(np.log(TOPK))       # 6.238 nats — max entropy of a 512-way uniform
HIST_BINS = 250


# ---------------------------------------------------------------------------
# stage 1 — probe set
# ---------------------------------------------------------------------------

def build_probe(pool_path: str, n: int, seed: int, out_path: str) -> list[dict]:
    """Proportional level-stratified sample of ``n`` problems.

    Proportional, NOT equal-per-level: DeepMath's difficulty histogram is peaked
    at 5–8 and nearly empty at levels 2–3 and 10 (activity 002 note 1), so equal
    quotas are unfillable. ``poolutil.stratified_sample`` is the v1 machinery.
    """
    rows = read_jsonl(pool_path)
    sample = stratified_sample(rows, lambda r: str(r.get("level", "_")), n,
                               random.Random(seed))
    write_jsonl(out_path, sample)
    return sample


# ---------------------------------------------------------------------------
# stage 2 — rollouts
# ---------------------------------------------------------------------------

def generate_rollouts(
    probe: list[dict],
    model: str,
    out_path: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> list[dict]:
    """One rollout per problem via vLLM, ``enable_thinking=True`` (ROADMAP rule 4).

    Stores vLLM's own ``token_ids``. Re-tokenizing the decoded text would shift
    boundary offsets — the exact bug the token-level parser exists to avoid.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in probe
    ]
    llm = LLM(
        model=model,
        max_model_len=max_tokens + 4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )
    outs = llm.generate(
        prompts,
        SamplingParams(temperature=temperature, top_p=top_p,
                       max_tokens=max_tokens, seed=seed),
    )
    rows: list[dict] = []
    for r, p, o in zip(probe, prompts, outs):
        out = o.outputs[0]
        rows.append({
            "_uid": r["_uid"],
            "level": r.get("level"),
            "prompt": r["prompt"],
            # Carried through so downstream consumers (e.g.
            # stage_register_exemplars.py) can verifier-filter without a
            # second join against the pool.
            "ground_truth": r.get("ground_truth"),
            "prompt_text": p,
            "completion": out.text,
            "completion_token_ids": list(out.token_ids),
            "finish_reason": out.finish_reason,
        })
    write_jsonl(out_path, rows)

    # Explicit teardown. vLLM's EngineCore is a separate process; if the parent
    # exits without releasing it, it is reparented to init and sits on the whole
    # GPU allocation forever (observed on turing 2026-08-02 — the next vLLM start
    # dies with "Engine core initialization failed" and the real cause, an OOM,
    # is ~200 lines up the log). Kill it here rather than relying on GC order.
    import gc

    import torch
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm, outs
    gc.collect()
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# stage 3 — teacher-forced top-512 entropy
# ---------------------------------------------------------------------------

def _chunked_topk_entropy(model, hidden, chunk: int) -> np.ndarray:
    """Entropy (nats) of the top-``TOPK`` logits at every position of ``hidden``.

    ``hidden`` is ``[1, T, H]`` from the base transformer. ``lm_head`` is applied
    per chunk so the full-vocab logits never exist for more than ``chunk``
    positions at a time.
    """
    import torch

    T = hidden.shape[1]
    out = np.empty(T, dtype=np.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits = model.lm_head(hidden[:, s:e, :])           # [1, c, V] bf16
        top = torch.topk(logits, TOPK, dim=-1).values       # [1, c, 512]
        del logits
        p = torch.softmax(top.float(), dim=-1)              # renormalized on 512
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)  # [1, c]
        out[s:e] = ent[0].detach().cpu().numpy()
        del top, p, ent
    return out


def score_traces(
    rows: list[dict],
    model_name: str,
    tok,
    *,
    chunk: int,
    max_len: int,
    label: str,
) -> dict:
    """Teacher-force each (prompt, completion) and collect per-token entropies.

    Returns concatenated arrays plus per-trace metadata. Position alignment:
    ``logits[t-1]`` predicts token ``t``, so the entropy recorded *for* token
    ``t`` is read from row ``t-1`` of the entropy array.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    base = getattr(model, "model", None)
    if base is None or not hasattr(model, "lm_head"):
        raise RuntimeError(
            f"{model_name}: expected .model + .lm_head to apply the head in chunks"
        )

    blank = blank_token_ids_for(tok)

    ent_parts, seg_parts, lvl_parts, idx_parts = [], [], [], []
    meta: list[dict] = []
    boundary_entropies: list[float] = []

    for i, r in enumerate(rows):
        prompt_ids = tok(r["prompt_text"], add_special_tokens=False).input_ids
        comp_ids = r["completion_token_ids"]
        full = list(prompt_ids) + list(comp_ids)
        truncated = False
        if len(full) > max_len:
            full = full[:max_len]
            truncated = True
        p_len = len(prompt_ids)

        m = parse_segments(full, prompt_len=p_len, blank_token_ids=blank)

        with torch.no_grad():
            ids = torch.tensor([full], device="cuda")
            hidden = base(input_ids=ids).last_hidden_state       # [1, T, H]
            ent_all = _chunked_topk_entropy(model, hidden, chunk)  # [T]
            del hidden, ids

        # Entropy FOR token t is the predictive entropy at t-1.
        # Completion tokens are t in [p_len, T); all have a valid t-1.
        T = len(full)
        pos = np.arange(p_len, T)
        ent = ent_all[pos - 1]

        seg = np.full(pos.shape, SEG_OTHER, dtype=np.int8)
        tm, am = m.think_mask, m.answer_mask
        for j, t in enumerate(pos):
            if tm[t]:
                seg[j] = SEG_THINK
            elif am[t]:
                seg[j] = SEG_ANSWER

        # Sanity: entropy of the boundary tokens themselves.
        if m.close_idx >= 0:
            boundary_entropies.append(float(ent_all[m.close_idx - 1]))

        ent_parts.append(ent)
        seg_parts.append(seg)
        lvl_parts.append(np.full(pos.shape, int(r.get("level") or 0), dtype=np.int16))
        idx_parts.append(np.full(pos.shape, i, dtype=np.int32))

        meta.append({
            "_uid": r["_uid"],
            "level": r.get("level"),
            "g": m.g,
            "reason": m.reason,
            "warnings": list(m.warnings),
            "think_len": m.think_len,
            "answer_len": m.answer_len,
            "n_completion_tokens": len(comp_ids),
            "finish_reason": r.get("finish_reason"),
            "truncated_for_audit": truncated,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{label}] scored {i + 1}/{len(rows)}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "entropy": np.concatenate(ent_parts),
        "segment": np.concatenate(seg_parts),
        "level": np.concatenate(lvl_parts),
        "trace_idx": np.concatenate(idx_parts),
        "meta": meta,
        "boundary_close_entropy": boundary_entropies,
    }


# ---------------------------------------------------------------------------
# stage 4 — aggregate + verdict
# ---------------------------------------------------------------------------

def _stats(x: np.ndarray) -> dict:
    pct = [1, 5, 10, 25, 50, 75, 80, 90, 95, 99]
    if x.size == 0:
        # Keep every key present so the verdict/print/plot paths cannot
        # KeyError on a degenerate run (e.g. every rollout gated out).
        nan = float("nan")
        return {
            "n": 0, "mean": nan, "std": nan,
            **{f"p{p}": nan for p in pct},
            "collapse_mass_lt_0.1": nan, "fork_mass_gt_1.5": nan,
            "skew": nan, "kurtosis": nan, "bimodality_coefficient": nan,
        }
    q = np.percentile(x, pct)
    mean, std = float(x.mean()), float(x.std())
    # Sarle's bimodality coefficient: > 5/9 ≈ 0.555 suggests bimodality.
    if std > 0:
        z = (x - mean) / std
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean())
        bimod = (skew ** 2 + 1.0) / kurt if kurt > 0 else float("nan")
    else:
        skew = kurt = float("nan")
        bimod = float("nan")
    return {
        "n": int(x.size),
        "mean": mean,
        "std": std,
        **{f"p{p}": float(v) for p, v in zip(pct, q)},
        "collapse_mass_lt_0.1": float((x < 0.1).mean()),
        "fork_mass_gt_1.5": float((x > 1.5).mean()),
        "skew": skew,
        "kurtosis": kurt,
        "bimodality_coefficient": bimod,
    }


def _hist(x: np.ndarray) -> dict:
    counts, edges = np.histogram(x, bins=HIST_BINS, range=(0.0, HIST_MAX))
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def aggregate(scored: dict, compare: Optional[dict]) -> dict:
    ent, seg, lvl = scored["entropy"], scored["segment"], scored["level"]
    think, answer = ent[seg == SEG_THINK], ent[seg == SEG_ANSWER]

    report = {
        "topk": TOPK,
        "max_possible_entropy_nats": HIST_MAX,
        "overall": _stats(ent),
        "think": _stats(think),
        "answer": _stats(answer),
        "per_level_think_median": {},
        "per_level_answer_median": {},
        "histograms": {
            "overall": _hist(ent),
            "think": _hist(think),
            "answer": _hist(answer),
        },
        "boundary_close_entropy": {
            "n": len(scored["boundary_close_entropy"]),
            "median": (float(np.median(scored["boundary_close_entropy"]))
                       if scored["boundary_close_entropy"] else None),
        },
    }
    for L in sorted(set(lvl.tolist())):
        tsel = think[lvl[seg == SEG_THINK] == L]
        asel = answer[lvl[seg == SEG_ANSWER] == L]
        if tsel.size:
            report["per_level_think_median"][int(L)] = float(np.median(tsel))
        if asel.size:
            report["per_level_answer_median"][int(L)] = float(np.median(asel))

    # gate / rollout health
    metas = scored["meta"]
    report["rollout_health"] = {
        "n_traces": len(metas),
        "n_gate_pass": sum(1 for m in metas if m["g"] == 1),
        "gate_pass_rate": (sum(1 for m in metas if m["g"] == 1) / len(metas)) if metas else 0.0,
        "gate_fail_reasons": {
            r: sum(1 for m in metas if m["reason"] == r)
            for r in sorted({m["reason"] for m in metas if m["g"] == 0})
        },
        "median_think_len": float(np.median([m["think_len"] for m in metas])) if metas else 0,
        "median_answer_len": float(np.median([m["answer_len"] for m in metas])) if metas else 0,
        "cap_hit_rate": (sum(1 for m in metas if m["finish_reason"] == "length") / len(metas))
                        if metas else 0.0,
    }

    if compare is not None:
        c_ent, c_seg = compare["entropy"], compare["segment"]
        report["compare"] = {
            "model": compare["model_name"],
            "note": "same traces teacher-forced through a reference (non-RL) checkpoint",
            "overall": _stats(c_ent),
            "think": _stats(c_ent[c_seg == SEG_THINK]),
            "answer": _stats(c_ent[c_seg == SEG_ANSWER]),
        }

    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> dict:
    """Preservation vs restoration.

    There is no published absolute threshold (packet P2 Part 2 step 4), so the
    rule below is an explicit, auditable heuristic and every input to it is
    reported alongside the label. The comparison against a reference
    non-RL checkpoint — when ``--compare_model`` is given — is the strongest
    single signal ("median comparable to a base model", design §1).
    """
    t = report["think"]
    checks = {
        "think_median_below_0.15_nats": t["p50"] < 0.15,
        "collapse_mass_above_0.60": t["collapse_mass_lt_0.1"] > 0.60,
        "fork_mass_below_0.10": t["fork_mass_gt_1.5"] < 0.10,
    }
    if "compare" in report:
        ref = report["compare"]["think"]["p50"]
        checks["think_median_below_half_of_reference"] = (
            ref > 0 and t["p50"] < 0.5 * ref
        )
        report.setdefault("verdict_inputs", {})["reference_think_median"] = ref
    n_hit = sum(bool(v) for v in checks.values())
    mode = "restoration" if n_hit >= 2 else "preservation"
    return {
        "mode": mode,
        "delta_max": 0.7 if mode == "restoration" else 0.5,
        "checks": {k: bool(v) for k, v in checks.items()},
        "n_checks_triggered": n_hit,
        "rule": "restoration if >=2 collapse checks trigger; heuristic, argue it "
                "in the activity file (packet P2 Part 2 step 4)",
        "think_median_nats": t["p50"],
        "think_collapse_mass": t["collapse_mass_lt_0.1"],
        "think_fork_mass": t["fork_mass_gt_1.5"],
        "bimodality_coefficient": t["bimodality_coefficient"],
    }


def make_plots(report: dict, out_dir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping PNGs", flush=True)
        return []

    paths = []
    for name in ("overall", "think", "answer"):
        h = report["histograms"][name]
        counts = np.array(h["counts"], dtype=float)
        edges = np.array(h["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        frac = counts / counts.sum() if counts.sum() else counts
        s = report[name]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(centers, frac, width=(edges[1] - edges[0]), color="#3b6ea5")
        ax.axvline(0.1, color="#c0392b", ls="--", lw=1, label="collapse < 0.1")
        ax.axvline(1.5, color="#27ae60", ls="--", lw=1, label="fork > 1.5")
        ax.axvline(s["p50"], color="k", lw=1.2, label=f"median {s['p50']:.3f}")
        ax.axvline(s["p80"], color="#8e44ad", lw=1.2,
                   label=f"p80 {s['p80']:.3f} (H_pivot recipe)")
        ax.set_xlabel("top-512 entropy (nats)")
        ax.set_ylabel("fraction of tokens")
        ax.set_title(
            f"Qwen3-1.7B entropy audit — {name} segment  "
            f"(n={s['n']:,}, collapse={s['collapse_mass_lt_0.1']:.1%}, "
            f"fork={s['fork_mass_gt_1.5']:.1%})"
        )
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, f"entropy_hist_{name}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # think vs answer overlay
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in (("think", "#3b6ea5"), ("answer", "#e67e22")):
        h = report["histograms"][name]
        counts = np.array(h["counts"], dtype=float)
        edges = np.array(h["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        frac = counts / counts.sum() if counts.sum() else counts
        ax.plot(centers, frac, color=color, lw=1.4, label=f"{name} (n={report[name]['n']:,})")
    ax.set_xlabel("top-512 entropy (nats)")
    ax.set_ylabel("fraction of tokens")
    ax.set_yscale("log")
    ax.set_title("Segment-level entropy — think vs answer (reported separately, always)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "entropy_hist_think_vs_answer.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pool", default="/data/whetstone/data/pool/val_2k.jsonl")
    ap.add_argument("--traces", default=None,
                    help="skip generation; audit these traces (P3 H_pivot mode)")
    ap.add_argument("--completion_field", default="completion")
    ap.add_argument("--out_dir", default="/data/whetstone/runs/entropy_audit")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--max_len", type=int, default=20480,
                    help="hard cap on prompt+completion during teacher forcing")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="positions per lm_head chunk (memory gotcha 1)")
    ap.add_argument("--compare_model", default=None,
                    help="reference checkpoint for the preservation/restoration call, "
                         "e.g. Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--generate_only", action="store_true",
                    help="stop after writing rollouts.jsonl. Run this first, then "
                         "re-invoke without the flag to score: vLLM does not release "
                         "GPU memory promptly, and the HF scoring pass needs the card "
                         "to itself. Scoring resumes from rollouts.jsonl.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    probe_path = os.path.join(args.out_dir, "probe.jsonl")
    roll_path = os.path.join(args.out_dir, "rollouts.jsonl")
    npz_path = os.path.join(args.out_dir, "per_token_entropy.npz")
    json_path = os.path.join(args.out_dir, "audit.json")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # --- stages 1-2: get rows with prompt_text + completion_token_ids -------
    if args.traces:
        rows = read_jsonl(args.traces)
        for r in rows:
            if "prompt_text" not in r:
                r["prompt_text"] = tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if "completion_token_ids" not in r:
                r["completion_token_ids"] = tok(
                    r[args.completion_field], add_special_tokens=False).input_ids
        print(f"[traces] {len(rows)} traces from {args.traces}", flush=True)
    else:
        if os.path.exists(roll_path) and not args.force:
            rows = read_jsonl(roll_path)
            print(f"[resume] {len(rows)} rollouts from {roll_path}", flush=True)
        else:
            probe = (read_jsonl(probe_path)
                     if os.path.exists(probe_path) and not args.force
                     else build_probe(args.pool, args.n, args.seed, probe_path))
            print(f"[probe] {len(probe)} problems -> {probe_path}", flush=True)
            rows = generate_rollouts(
                probe, args.model, roll_path,
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, seed=args.seed)
            print(f"[generate] {len(rows)} rollouts -> {roll_path}", flush=True)

    if args.generate_only:
        print("[generate_only] stopping before the scoring pass; re-invoke without "
              "--generate_only to score.", flush=True)
        return 0

    # --- stage 3: entropy ---------------------------------------------------
    print(f"[score] teacher-forcing {len(rows)} traces through {args.model}", flush=True)
    scored = score_traces(rows, args.model, tok, chunk=args.chunk,
                          max_len=args.max_len, label=args.model)

    compare = None
    if args.compare_model:
        print(f"[score] reference pass through {args.compare_model}", flush=True)
        compare = score_traces(rows, args.compare_model, tok, chunk=args.chunk,
                               max_len=args.max_len, label=args.compare_model)
        compare["model_name"] = args.compare_model

    # Raw per-token arrays — later comparisons need distributions, not just the
    # histogram (packet P2 Part 2 gotcha 4).
    save = {
        "entropy": scored["entropy"],
        "segment": scored["segment"],
        "level": scored["level"],
        "trace_idx": scored["trace_idx"],
    }
    if compare is not None:
        save["compare_entropy"] = compare["entropy"]
        save["compare_segment"] = compare["segment"]
    np.savez_compressed(npz_path, **save)
    print(f"[npz] {npz_path} ({os.path.getsize(npz_path) / 1e6:.1f} MB)", flush=True)

    # --- stage 4: aggregate -------------------------------------------------
    report = aggregate(scored, compare)
    report["config"] = vars(args)
    report["traces"] = scored["meta"]
    if args.traces:
        # Machine-readable, so P4 reads the pinned number instead of
        # re-deriving a percentile and hoping it picked the same segment.
        report["h_pivot"] = {
            "value": report["think"]["p80"],
            "recipe": "p80 of the think-segment entropy histogram (design §12.6)",
            "source_traces": args.traces,
            "completion_field": args.completion_field,
            "n_think_tokens": report["think"]["n"],
            "model": args.model,
        }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1)
    pngs = make_plots(report, args.out_dir)

    # --- print --------------------------------------------------------------
    v = report["verdict"]
    rh = report["rollout_health"]
    print()
    print("=" * 72)
    print("ENTROPY AUDIT")
    print("=" * 72)
    print(f"traces: {rh['n_traces']}  gate-pass {rh['gate_pass_rate']:.1%}  "
          f"cap-hit {rh['cap_hit_rate']:.1%}")
    print(f"median lengths — think {rh['median_think_len']:.0f} tok, "
          f"answer {rh['median_answer_len']:.0f} tok   (reported separately, always)")
    for name in ("overall", "think", "answer"):
        s = report[name]
        print(f"  {name:<8} n={s['n']:>9,}  median={s['p50']:.4f}  mean={s['mean']:.4f}  "
              f"p80={s['p80']:.4f}  p95={s['p95']:.4f}  "
              f"collapse={s['collapse_mass_lt_0.1']:.1%}  fork={s['fork_mass_gt_1.5']:.1%}  "
              f"BC={s['bimodality_coefficient']:.3f}")
    if "compare" in report:
        c = report["compare"]["think"]
        print(f"  {'REF think':<8} n={c['n']:>9,}  median={c['p50']:.4f}  "
              f"collapse={c['collapse_mass_lt_0.1']:.1%}  fork={c['fork_mass_gt_1.5']:.1%}"
              f"   [{report['compare']['model']}]")
    be = report["boundary_close_entropy"]
    print(f"  sanity: median entropy of the </think> token itself = {be['median']}"
          "   (should be LOW; if high, suspect an off-by-one)")
    print()
    if args.traces:
        # In --traces mode the think histogram IS the corpus's own histogram,
        # so p80 is the H_pivot candidate rather than a reference number. The
        # preservation/restoration verdict is deliberately not printed here: it
        # is a property of the *checkpoint*, decided once by the P2 audit over
        # native rollouts, and re-deciding it from a register corpus would be
        # reading a different distribution as if it were the same one.
        print(f"H_pivot  = {report['think']['p80']:.4f} nats"
              "   (p80 of the think histogram of --traces; design §12.6)")
        print(f"           source: {args.traces}")
        print(f"           n_think_tokens={report['think']['n']:,}, "
              f"median={report['think']['p50']:.4f}")
        print("           P4 and Stage B consume this number — record it.")
        print()
        print("NOTE: preservation-vs-restoration is NOT re-decided here; that is "
              "a property of the checkpoint,")
        print("      fixed by the P2 audit over native rollouts "
              "(activity 003: RESTORATION, Delta_max = 0.7).")
    else:
        print(f"VERDICT: {v['mode'].upper()}   ->  Stage-B Delta_max = {v['delta_max']}")
        for k, hit in v["checks"].items():
            print(f"    [{'X' if hit else ' '}] {k}")
        print(f"    ({v['n_checks_triggered']} triggered; {v['rule']})")
        print()
        print("NOTE: H_pivot is NOT set here. It is the 80th percentile of the "
              "COMPACT-REGISTER histogram; re-run with --traces on the P3 seed "
              "corpus.")
        print(f"      (p80 of this native-trace think histogram, for reference "
              f"only: {report['think']['p80']:.4f})")
    print()
    print(f"artifacts: {json_path}")
    for p in pngs:
        print(f"           {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
