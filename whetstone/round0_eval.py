"""Round-0 meter metrics: S1/S2/S3, the hum trajectory, and the meter tests.

Used by the inoculation trainer (in-process, every 10 optimizer steps) *and* by
the Part-4 meter tests (loading arbitrary checkpoints). One implementation, so a
threshold crossed during training means the same thing when it is re-checked at
the winner and its neighbours.

Everything routes through :mod:`whetstone.round0` for sequence construction and
through :mod:`whetstone.segments` for masks — packet §4's "do this identically
everywhere".

**Why pi_0 is a cache and not a resident model.** The 32 GB budget cannot hold
fp32 theta + grads + Adam moments + the SED shadow *and* a frozen pi_0 copy
(~31 GB before pi_0). But pi_0 is frozen, so everything it contributes can be
computed once: its top-512 (ids, logprobs, entropy) and its actual-token
logprob, at a fixed seeded sample of control positions. See
``scripts/precompute_pi0_cache.py``. This also makes S2 and meter test (b)
exactly reproducible across runs.

**Top-512 KL support.** ``KL(pi_theta || pi_0)`` is evaluated on pi_0's top-512
support with both sides renormalized over it. That is the design's top-512
convention (§12.3) and the only one the cache can express; it is a drift gauge,
not an exact divergence, and it under-reports mass pi_theta moves *outside*
pi_0's top-512. Over 120 optimizer steps at LR 1e-5 that tail is negligible,
but the choice is recorded because a different support gives a different kappa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from whetstone.round0 import (
    Seq,
    build_sequence,
    build_sequence_from_ids,
    load_jsonl,
    percentile,
    render_prompt,
)

TOPK = 512


# --------------------------------------------------------------------------
# Eval sets
# --------------------------------------------------------------------------

@dataclass
class EvalSets:
    """The two measurement sets, built once and reused at every eval.

    ``control_positions`` is a *fixed seeded sample* of each control trace's
    think positions. The forward still runs over the whole sequence (the
    conditioning has to be right); only the scored positions are subsampled, so
    S2/S3/(b) cost ~60k scored positions instead of ~1.2M while the sample is
    identical at every eval and across checkpoints.
    """

    heldout: List[Seq] = field(default_factory=list)
    control: List[Seq] = field(default_factory=list)
    control_positions: List[np.ndarray] = field(default_factory=list)


def build_eval_sets(
    tokenizer,
    corpus_dir: str | Path,
    *,
    n_control: int = 0,
    n_control_positions: int = 300,
    seed: int = 0,
) -> EvalSets:
    corpus = Path(corpus_dir)
    rng = np.random.default_rng(seed)

    heldout = []
    for r in load_jsonl(corpus / "heldout_register.jsonl"):
        heldout.append(
            build_sequence(
                tokenizer,
                uid=r["_uid"],
                problem=r["prompt"],
                think_body=r["compact_think"],
                answer=r["answer"],
                level=r.get("level", 0),
                require_gate=True,
            )
        )

    control, positions = [], []
    recs = load_jsonl(corpus / "verbose_control.jsonl")
    if n_control:
        recs = recs[:n_control]
    for r in recs:
        # Native rollouts carry their own token ids; re-tokenizing decoded text
        # does not round-trip at the <think> boundary (design §12.1).
        seq = build_sequence_from_ids(
            uid=r["_uid"],
            prompt_ids=render_prompt(tokenizer, r["prompt"]),
            completion_ids=r["completion_token_ids"],
            level=r.get("level", 0),
        )
        if seq.masks.g != 1:
            continue
        think = np.array([p for p in seq.think_positions if p > 0], dtype=np.int64)
        if think.size == 0:
            continue
        if think.size > n_control_positions:
            think = np.sort(rng.choice(think, size=n_control_positions, replace=False))
        control.append(seq)
        positions.append(think)

    return EvalSets(heldout=heldout, control=control, control_positions=positions)


# --------------------------------------------------------------------------
# Forward helpers
# --------------------------------------------------------------------------

@torch.no_grad()
def logits_at(model, ids: torch.Tensor, rows: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Logits for the *predicting* rows only — ``(N, V)``.

    Runs the transformer body over the full sequence, then applies ``lm_head``
    to just the requested hidden states. A 6k-token control trace would
    otherwise materialize 6212 x 151936 logits (1.9 GB) for the ~300 positions
    actually scored.

    The bf16 autocast is load-bearing, not a speed tweak: the trainee's weights
    are fp32, and fp32 SDPA cannot use the flash/mem-efficient kernels, so it
    falls back to the math backend and materializes a full (T, T) attention
    matrix — 10 GB on a 6.2k-token control trace, which OOMs the eval. It also
    matches the precision the training forward runs at.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model(input_ids=ids)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = h[0, rows, :]
        return torch.cat(
            [model.lm_head(h[s : s + chunk]) for s in range(0, h.shape[0], chunk)]
        )


@torch.no_grad()
def score_positions(
    model,
    seq: Seq,
    positions: np.ndarray,
    *,
    device: str = "cuda",
    topk: int = TOPK,
    want_top: bool = False,
) -> Dict[str, np.ndarray]:
    """Score one sequence at ``positions`` (token positions, all > 0).

    Alignment: the row predicting token ``t`` is ``t-1`` (packet §4). Getting
    this wrong shifts think/answer attribution silently, so it is written once,
    here, and asserted by the ``</think>``-entropy sanity anchor in the trainer.

    Returns arrays over ``positions``: ``surprisal``, ``gap`` (d_t),
    ``entropy`` (top-k), and — when ``want_top`` — ``top_ids`` / ``top_lp``.
    """
    ids = torch.tensor([seq.ids], device=device)
    rows = torch.tensor(positions - 1, device=device)
    logits = logits_at(model, ids, rows).float()

    logprobs = F.log_softmax(logits, dim=-1)
    actual = torch.tensor(np.asarray(seq.ids)[positions], device=device, dtype=torch.long)
    lp_actual = logprobs.gather(-1, actual.unsqueeze(-1)).squeeze(-1)
    lp_top1 = logprobs.max(dim=-1).values

    tv, ti = torch.topk(logprobs, k=min(topk, logprobs.shape[-1]), dim=-1)
    # Entropy of the top-k renormalized distribution — the design's convention
    # (§12.3), so it is on the same scale as H_pivot and the audit baseline.
    p = F.log_softmax(tv, dim=-1)
    ent = -(p.exp() * p).sum(-1)

    out = {
        "surprisal": (-lp_actual).cpu().numpy(),
        "gap": (lp_top1 - lp_actual).clamp_min(0).cpu().numpy(),
        "entropy": ent.cpu().numpy(),
        "actual_lp": lp_actual.cpu().numpy(),
    }
    if want_top:
        out["top_ids"] = ti.cpu().numpy().astype(np.int32)
        out["top_lp"] = tv.cpu().numpy().astype(np.float32)
    return out


# --------------------------------------------------------------------------
# pi_0 reference cache
# --------------------------------------------------------------------------

class Pi0Cache:
    """Frozen-pi_0 reference values at the fixed control positions."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        self.top_ids = z["top_ids"]        # (M, 512) int32
        self.top_lp = z["top_lp"]          # (M, 512) float32
        self.actual_lp = z["actual_lp"]    # (M,)     float32
        self.entropy = z["entropy"]        # (M,)     float32
        self.offsets = z["offsets"]        # (n+1,)   int64
        self.uids = [str(u) for u in z["uids"]]

    def slice(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return (
            self.top_ids[a:b],
            self.top_lp[a:b],
            self.actual_lp[a:b],
            self.entropy[a:b],
        )


def assert_alignment(model, sets: EvalSets, *, n: int = 40, device: str = "cuda",
                     max_median: float = 0.01) -> float:
    """Off-by-one alarm: ``</think>`` entropy on **native** control traces.

    The model itself emitted ``</think>`` in these rollouts, so the predicting
    distribution is near-deterministic there — activity 003's audit measured a
    median of 6.6e-05 over 182 traces. If the logits/token alignment slips by
    one, this jumps by orders of magnitude, and every think/answer attribution
    downstream is silently wrong.

    Deliberately *not* run on the compact register set: there the same quantity
    is legitimately ~0.275 nats under pi_0, so it cannot distinguish a real
    misalignment from the register's own accent.
    """
    vals = []
    for seq in sets.control[:n]:
        ci = seq.masks.close_idx
        if ci > 0:
            s = score_positions(model, seq, np.array([ci]), device=device)
            vals.append(float(s["entropy"][0]))
    med = float(np.median(vals)) if vals else math.nan
    if not (med < max_median):
        raise AssertionError(
            f"</think> entropy median on native traces = {med:.5f} nats, expected "
            f"< {max_median} (activity 003 audit: 6.6e-05). The logits/token "
            "alignment is off by one — every metric in this run would be shifted."
        )
    return med


# --------------------------------------------------------------------------
# The four monitoring curves (packet §7)
# --------------------------------------------------------------------------

def evaluate(
    model,
    sets: EvalSets,
    *,
    r_ids: frozenset,
    class_ids: Dict[str, frozenset],
    cache: Optional[Pi0Cache] = None,
    device: str = "cuda",
    topk: int = TOPK,
    audit_think_median: float = 0.02781715616583824,
) -> dict:
    """Compute S1, S2, S3 and the hum trajectory at the model's current weights.

    Returns a flat dict of scalars plus the raw heldout gap array (for the
    Part-4 histograms), ready to append to the metrics JSONL.
    """
    was_training = model.training
    model.eval()

    # --- S1 / hum: heldout register gaps -------------------------------
    gaps_all: List[float] = []
    gaps_by_class: Dict[str, List[float]] = {k: [] for k in class_ids}
    surp_r: List[float] = []
    gaps_r: List[float] = []
    surp_all: List[float] = []
    close_entropy: List[float] = []

    for seq in sets.heldout:
        pos = np.array([p for p in seq.think_positions if p > 0], dtype=np.int64)
        if pos.size == 0:
            continue
        s = score_positions(model, seq, pos, device=device, topk=topk)
        tok_ids = np.asarray(seq.ids)[pos]
        gaps_all.extend(s["gap"].tolist())
        surp_all.extend(s["surprisal"].tolist())
        for name, ids in class_ids.items():
            sel = np.isin(tok_ids, list(ids))
            if sel.any():
                gaps_by_class[name].extend(s["gap"][sel].tolist())
        sel_r = np.isin(tok_ids, list(r_ids))
        if sel_r.any():
            surp_r.extend(s["surprisal"][sel_r].tolist())
            gaps_r.extend(s["gap"][sel_r].tolist())

        # Descriptive only — NOT the alignment anchor. Measured under pi_0 the
        # median here is 0.275 nats on compact traces, against 8.0e-05 on native
        # ones (activity 007): a verbose-CoT native genuinely does not expect a
        # compact trace to end where it does. The packet's ~1e-4..0.02 anchor is
        # a statement about *native* traces, so the off-by-one assertion lives in
        # `assert_alignment` and runs on the control set.
        ci = seq.masks.close_idx
        if ci > 0:
            se = score_positions(model, seq, np.array([ci]), device=device, topk=topk)
            close_entropy.append(float(se["entropy"][0]))

    out = {
        "s1_p95_gap": percentile(gaps_all, 95),
        "s1_p99_gap": percentile(gaps_all, 99),
        "s1_mean_gap": float(np.mean(gaps_all)) if gaps_all else math.nan,
        "s1_max_gap": float(np.max(gaps_all)) if gaps_all else math.nan,
        "hum_R_mean_surprisal": float(np.mean(surp_r)) if surp_r else math.nan,
        "hum_R_n": len(surp_r),
        # Test (a) asks whether the REGISTER spikes. `s1_p95_gap` is taken over
        # every think token, so it also carries the trace's mathematical content
        # — tokens the scorer has no reason to predict and that inoculation is
        # not meant to touch. The R-restricted gap is the direct measurement of
        # the design's "register = hum (elevated mean, no spikes)" condition.
        "s1_p95_gap_R": percentile(gaps_r, 95),
        "s1_mean_gap_R": float(np.mean(gaps_r)) if gaps_r else math.nan,
        "heldout_mean_surprisal": float(np.mean(surp_all)) if surp_all else math.nan,
        "heldout_think_tokens": len(gaps_all),
        # entropy_audit.py reports the MEDIAN of this (activities 003/005 quote
        # ~1e-4..0.02); the mean is dominated by a handful of traces where the
        # compact body genuinely does not predict its own end. Report both, and
        # compare the median when using it as the off-by-one alarm.
        "close_token_entropy_median": float(np.median(close_entropy)) if close_entropy else math.nan,
        "close_token_entropy_mean": float(np.mean(close_entropy)) if close_entropy else math.nan,
    }
    for name, vals in gaps_by_class.items():
        out[f"s1_p95_gap_{name}"] = percentile(vals, 95)
        out[f"s1_mean_gap_{name}"] = float(np.mean(vals)) if vals else math.nan
        out[f"s1_n_{name}"] = len(vals)

    # --- S2 / drift + S3 / entropy + (b) verbose intact ------------------
    if cache is not None:
        kl_sum = 0.0
        n_kl = 0
        lp_delta: List[float] = []
        ent_theta: List[float] = []
        ent_pi0: List[float] = []
        ctrl_gaps: List[float] = []

        for i, (seq, pos) in enumerate(zip(sets.control, sets.control_positions)):
            c_ids, c_lp, c_actual, c_ent = cache.slice(i)

            # One forward per control trace supplies S2, S3 and (b). Scoring it
            # twice here doubled the most expensive part of every eval.
            ids_full = torch.tensor([seq.ids], device=device)
            rows = torch.tensor(pos - 1, device=device)
            logits = logits_at(model, ids_full, rows).float()
            logprobs = F.log_softmax(logits, dim=-1)

            actual = torch.tensor(
                np.asarray(seq.ids)[pos], device=device, dtype=torch.long
            )
            lp_actual = logprobs.gather(-1, actual.unsqueeze(-1)).squeeze(-1)
            # The verbose baseline d_t. tau_spike has to sit between this and
            # the register's p95, so it is measured here rather than assumed —
            # the packet's 0.750 anchor came from a different corpus.
            ctrl_gaps.extend(
                (logprobs.max(dim=-1).values - lp_actual).clamp_min(0).cpu().numpy().tolist()
            )

            tv = torch.topk(logprobs, k=min(topk, logprobs.shape[-1]), dim=-1).values
            p = F.log_softmax(tv, dim=-1)
            ent = -(p.exp() * p).sum(-1)

            # KL(pi_theta || pi_0) on pi_0's top-512 support, both renormalized.
            ids_t = torch.tensor(c_ids.astype(np.int64), device=device)
            lp0 = F.log_softmax(torch.tensor(c_lp, device=device), dim=-1)
            lpq = F.log_softmax(logprobs.gather(-1, ids_t), dim=-1)
            kl = (lpq.exp() * (lpq - lp0)).sum(-1)
            kl_sum += float(kl.sum())
            n_kl += int(kl.numel())

            lp_delta.extend((lp_actual.cpu().numpy() - c_actual).tolist())
            ent_theta.extend(ent.cpu().numpy().tolist())
            ent_pi0.extend(c_ent.tolist())

        out.update({
            "s2_kl_mean": kl_sum / max(n_kl, 1),
            "s2_n_positions": n_kl,
            "b_logprob_delta_mean": float(np.mean(lp_delta)),
            "control_p95_gap": percentile(ctrl_gaps, 95),
            "control_mean_gap": float(np.mean(ctrl_gaps)),
            "control_entropy_median": float(np.median(ent_theta)),
            "control_entropy_mean": float(np.mean(ent_theta)),
            "control_entropy_p80": percentile(ent_theta, 80),
            "pi0_entropy_median": float(np.median(ent_pi0)),
            "pi0_entropy_mean": float(np.mean(ent_pi0)),
            "pi0_entropy_p80": percentile(ent_pi0, 80),
        })
        # S3 is reported three ways on purpose. The median of this checkpoint's
        # think-entropy distribution sits at ~0.028 nats inside a 56.8%-collapse
        # -mass region (activity 003), so a 10% move on it is noise, not signal.
        # The mean is the stable statistic; all three are logged so the choice
        # is auditable rather than assumed.
        for stat in ("median", "mean", "p80"):
            base = out[f"pi0_entropy_{stat}"]
            cur = out[f"control_entropy_{stat}"]
            out[f"s3_drop_{stat}"] = (base - cur) / base if base > 0 else math.nan
        out["s3_drop_vs_audit_median"] = (
            (audit_think_median - out["control_entropy_median"]) / audit_think_median
        )

    if was_training:
        model.train()
    out["_gaps_heldout"] = gaps_all
    return out


__all__ = [
    "EvalSets",
    "Pi0Cache",
    "TOPK",
    "build_eval_sets",
    "evaluate",
    "logits_at",
    "score_positions",
]
