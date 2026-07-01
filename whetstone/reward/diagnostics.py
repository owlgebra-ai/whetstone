"""§7 diagnostic metrics — per-group signals for reward-function health.

Every training checkpoint should emit these alongside its usual eval numbers.
The single most important is ``strict_vs_lenient_gap`` (§5.4).

Consumption model:
    ``compute_group_diagnostics([breakdown_i, ...], completions=[...])`` →
    :class:`GroupDiagnostics`. Log the fields directly to your dashboard.

Notes on subtleties (§7 addendum in the design doc):
    * **Single-integer-gold benign case.** ``unique_final_answer_frac ≈ 0.25``
      on a single-integer-gold problem (e.g. gold=2 with 6/8 correct emitting
      "2") is *convergence*, not collapse. The :attr:`entropy_collapse_flag`
      gates on jaccard AND uniq — do NOT add uniq-alone alerts.
    * **Shared-setup jaccard inflation.** Forced-figure geometry problems
      inflate 5-gram jaccard on the first ~100 tokens (setup is dictated).
      Pass ``strip_setup_tokens=100`` to :func:`compute_group_diagnostics`
      when scoring such a class.
    * **Chunk-restart count as format-health proxy.** 25k chars with 10
      chunk-restarts is fine; 25k chars with 0 is monolithic collapse.
      Both :attr:`min_chunk_restarts_in_group` and
      :attr:`max_chunk_restarts_in_group` are emitted.
    * **Confident-wrong vs chaotic-correct.** Length alone can't distinguish
      "short-confident-wrong under a bad premise" from "long-chaotic-correct
      that oscillates and lands right". The ratio is tracked here so you can
      watch the "verify before commit" SFT prior decay.
    * **Rollout logging (§9.d).** Group by ``(prompt, gold)``, not prompt
      alone — see :func:`.reward.diagnostics` docstring above.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .aggregate import RewardBreakdown
from .extract import char_len, count_chunk_restarts, split_think_close
from .tiers import Tier

_TOKEN_RE = re.compile(r"\w+")

# §7 addendum: "hedging" tokens used to score the "chaotic" vs "confident"
# distinction. Case-insensitive whole-word match.
_HEDGING_RE = re.compile(
    r"\b(?:wait|actually|hmm|hold on|let me reconsider|reconsider|"
    r"on second thought|but|however|maybe|perhaps)\b",
    re.IGNORECASE,
)

# Heuristic thresholds for confident-wrong / chaotic-correct classification.
_CONFIDENT_WRONG_MAX_THINK_CHARS = 4000
_CHAOTIC_LONG_MIN_THINK_CHARS = 8000
_CONFIDENT_MAX_HEDGES = 1
_CHAOTIC_MIN_HEDGES = 3


@dataclass(frozen=True)
class GroupDiagnostics:
    """Per-group signals emitted per checkpoint (§7 table + addendum).

    All optional fields are ``None`` when the underlying counter's denominator
    is 0 (e.g. no strict winners in the group → strict_winners_fmt_ge_0_10 is
    None). The training loop should treat ``None`` as "no data this step",
    not "0".

    New fields (§7 addendum):
      * ``min_chunk_restarts_in_group`` / ``max_chunk_restarts_in_group`` —
        format-health proxy that catches monolithic-collapse ("25k chars,
        0 restarts") vs. legitimate-long-reasoning ("25k chars, 10 restarts").
      * ``confident_wrong_count`` / ``chaotic_correct_count`` — SFT "verify
        before commit" prior health. If confident-wrong grows, the prior is
        decaying.
      * ``think_jaccard_5gram_mean_post_setup`` — jaccard computed after
        dropping the first ``strip_setup_tokens`` tokens of ``<think>``, to
        avoid forced-figure setup inflating the mean.
      * ``contradiction_rate`` — fraction of accepted rollouts flagged by
        the §4.10 contradiction detector (regardless of penalty mode).
      * ``prose_templated_strict_rate`` — fraction of strict rollouts that
        went through the §6.7 fallback (diagnostic for whether the primary
        verifier misses prose finalizers on this curriculum).
    """

    n_rollouts: int
    n_strict: int
    n_lenient: int
    n_wrong: int
    strict_vs_lenient_gap: Optional[float]
    strict_winners_fmt_ge_0_10: Optional[float]
    max_wrong_fmt: Optional[float]
    median_think_chars: Optional[float]
    p95_think_chars: Optional[float]
    cap_hit_rate: Optional[float]
    zero_advantage_group_frac: Optional[float]
    think_jaccard_5gram_mean: Optional[float]
    unique_final_answer_frac: Optional[float]
    entropy_collapse_flag: bool

    # §7 addendum
    min_chunk_restarts_in_group: Optional[int] = None
    max_chunk_restarts_in_group: Optional[int] = None
    confident_wrong_count: int = 0
    chaotic_correct_count: int = 0
    think_jaccard_5gram_mean_post_setup: Optional[float] = None
    contradiction_rate: Optional[float] = None
    prose_templated_strict_rate: Optional[float] = None

    detail: Dict[str, object] = field(default_factory=dict)


# --- helpers -----------------------------------------------------------------

def _percentile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    idx = (len(xs_sorted) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = idx - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def _extract_5grams(text: str, skip_tokens: int = 0) -> set:
    """Extract lowercase 5-gram set from ``text``, optionally skipping the first ``skip_tokens``.

    ``skip_tokens > 0`` is the §7-addendum forced-figure setup mitigation:
    drop the leading tokens (which are dictated by the problem setup, not
    the model's reasoning) before computing jaccard.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if skip_tokens > 0:
        tokens = tokens[skip_tokens:]
    if len(tokens) < 5:
        return set()
    return {" ".join(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def _pairwise_jaccard_mean(texts: List[str], skip_tokens: int = 0) -> Optional[float]:
    """Mean pairwise Jaccard on 5-grams across a group.

    Returns ``None`` if fewer than 2 texts have >=5 tokens after skipping.
    Used to flag entropy collapse (§7 last row) — semantic collapse ISN'T
    caught by this (see §7 warning), so pair with a semantic-diversity
    signal in practice.
    """
    sets = [_extract_5grams(t, skip_tokens=skip_tokens) for t in texts if t]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return None
    total = 0.0
    n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                continue
            total += len(a & b) / len(union)
            n += 1
    return total / n if n else None


def _count_hedges(think: str) -> int:
    """Count hedging tokens per §7 addendum ("confident-wrong vs chaotic-correct")."""
    if not think:
        return 0
    return len(_HEDGING_RE.findall(think))


def _extract_final_answer(post_think: str) -> str:
    """Cheap final-answer key for unique-frac counting; NOT the verifier."""
    if not post_think:
        return ""
    lines = [ln.strip() for ln in post_think.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# --- main --------------------------------------------------------------------

def compute_group_diagnostics(
    breakdowns: Sequence[RewardBreakdown],
    completions: Sequence[str],
    *,
    max_completion_chars: Optional[int] = None,
    strip_setup_tokens: int = 0,
) -> GroupDiagnostics:
    """Compute §7 diagnostics for one DAPO group.

    Parameters
    ----------
    breakdowns : list[RewardBreakdown]
        Per-rollout breakdowns from :func:`compute_reward`.
    completions : list[str]
        Same-order raw completion strings — needed for jaccard / unique-frac.
    max_completion_chars : int, optional
        If provided, cap-hit rate counts rollouts whose ``|think|`` reached
        this bound AND that didn't emit ``</think>``. If omitted, cap-hit
        rate falls back to fraction-of-rollouts without ``</think>`` (safe
        upper bound but overcounts genuine termination misses).
    strip_setup_tokens : int, default 0
        §7 addendum: for forced-figure problem classes (geometry with
        canonical setup), pass e.g. ``100`` to drop the first 100 tokens
        of ``<think>`` before computing jaccard. The addendum reports
        setup inflating jaccard into the 0.05–0.10 band even when the
        mid-trace algebra is diverse.

    Returns
    -------
    GroupDiagnostics
    """
    n = len(breakdowns)
    if n == 0:
        return GroupDiagnostics(
            n_rollouts=0, n_strict=0, n_lenient=0, n_wrong=0,
            strict_vs_lenient_gap=None, strict_winners_fmt_ge_0_10=None,
            max_wrong_fmt=None, median_think_chars=None, p95_think_chars=None,
            cap_hit_rate=None, zero_advantage_group_frac=None,
            think_jaccard_5gram_mean=None, unique_final_answer_frac=None,
            entropy_collapse_flag=False,
        )
    if len(completions) != n:
        raise ValueError(
            f"compute_group_diagnostics: len(breakdowns)={n} != len(completions)={len(completions)}"
        )

    splits = [split_think_close(c) for c in completions]
    think_chars = [char_len(s.think) for s in splits]

    strict = [b for b in breakdowns if b.tier == Tier.STRICT]
    lenient = [b for b in breakdowns if b.tier == Tier.LENIENT]
    wrong = [b for b in breakdowns if b.tier == Tier.WRONG]

    strict_vs_lenient_gap: Optional[float] = None
    if strict and lenient:
        strict_vs_lenient_gap = (
            statistics.mean(b.total for b in strict)
            - statistics.mean(b.total for b in lenient)
        )

    strict_winners_fmt_ge = (
        sum(1 for b in strict if b.r_fmt >= 0.10) / len(strict) if strict else None
    )
    max_wrong_fmt = max((b.r_fmt for b in wrong), default=None)

    median_chars = statistics.median(think_chars) if think_chars else None
    p95_chars = _percentile(think_chars, 0.95)

    if max_completion_chars is not None:
        cap_hits = sum(
            1
            for s, c in zip(splits, think_chars)
            if not s.has_closed_think and c >= max_completion_chars
        )
    else:
        cap_hits = sum(1 for s in splits if not s.has_closed_think)
    cap_hit_rate = cap_hits / n

    tier_values = {b.r_acc for b in breakdowns}
    zero_adv_frac = 1.0 if len(tier_values) == 1 else 0.0

    think_texts = [s.think for s in splits]
    jaccard_mean = _pairwise_jaccard_mean(think_texts)
    # §7 addendum: post-setup-token jaccard (only meaningful when strip>0).
    jaccard_mean_post_setup = (
        _pairwise_jaccard_mean(think_texts, skip_tokens=strip_setup_tokens)
        if strip_setup_tokens > 0
        else None
    )

    finals = [_extract_final_answer(s.post_think) for s in splits]
    finals = [f for f in finals if f]
    unique_final_frac = (len(set(finals)) / len(finals)) if finals else None

    # §7 entropy-collapse: jaccard >= 0.60 AND unique_final <= 0.15.
    # Explicitly does NOT include chunk_count_std or uniq-alone — the addendum
    # calls those out as false-positive triggers on SFT rigid templates and
    # single-integer-gold benign convergence, respectively.
    jaccard_for_collapse = jaccard_mean_post_setup if jaccard_mean_post_setup is not None else jaccard_mean
    entropy_collapse = (
        jaccard_for_collapse is not None
        and jaccard_for_collapse >= 0.60
        and unique_final_frac is not None
        and unique_final_frac <= 0.15
    )

    # §7 addendum: chunk-restart count as format-health proxy.
    restarts = [count_chunk_restarts(s.think) for s in splits]
    min_restarts = min(restarts) if restarts else None
    max_restarts = max(restarts) if restarts else None

    # §7 addendum: confident-wrong vs chaotic-correct.
    confident_wrong = 0
    chaotic_correct = 0
    for b, s, chars in zip(breakdowns, splits, think_chars):
        hedges = _count_hedges(s.think)
        if (
            chars <= _CONFIDENT_WRONG_MAX_THINK_CHARS
            and hedges <= _CONFIDENT_MAX_HEDGES
            and b.tier == Tier.WRONG
        ):
            confident_wrong += 1
        elif (
            chars >= _CHAOTIC_LONG_MIN_THINK_CHARS
            and hedges >= _CHAOTIC_MIN_HEDGES
            and b.tier in (Tier.STRICT, Tier.LENIENT)
        ):
            chaotic_correct += 1

    # New tier-related rates.
    contradiction_denom = len(strict) + len(lenient)
    contradiction_rate = (
        sum(1 for b in breakdowns if b.accuracy.contradiction_detected) / contradiction_denom
        if contradiction_denom
        else None
    )
    prose_templated_strict_rate = (
        sum(1 for b in strict if b.accuracy.prose_templated_accepted) / len(strict)
        if strict
        else None
    )

    return GroupDiagnostics(
        n_rollouts=n,
        n_strict=len(strict),
        n_lenient=len(lenient),
        n_wrong=len(wrong),
        strict_vs_lenient_gap=strict_vs_lenient_gap,
        strict_winners_fmt_ge_0_10=strict_winners_fmt_ge,
        max_wrong_fmt=max_wrong_fmt,
        median_think_chars=median_chars,
        p95_think_chars=p95_chars,
        cap_hit_rate=cap_hit_rate,
        zero_advantage_group_frac=zero_adv_frac,
        think_jaccard_5gram_mean=jaccard_mean,
        unique_final_answer_frac=unique_final_frac,
        entropy_collapse_flag=entropy_collapse,
        min_chunk_restarts_in_group=min_restarts,
        max_chunk_restarts_in_group=max_restarts,
        confident_wrong_count=confident_wrong,
        chaotic_correct_count=chaotic_correct,
        think_jaccard_5gram_mean_post_setup=jaccard_mean_post_setup,
        contradiction_rate=contradiction_rate,
        prose_templated_strict_rate=prose_templated_strict_rate,
    )


def health_report(diag: GroupDiagnostics) -> Dict[str, str]:
    """Turn a :class:`GroupDiagnostics` into per-metric health labels.

    Each label is one of ``"ok"`` / ``"warn"`` / ``"fail"`` / ``"n/a"``
    per the §7 healthy ranges. Meant to be dashboardable directly.
    """
    def band(value: Optional[float], ok_lo: float, ok_hi: Optional[float], warn_lo: Optional[float] = None) -> str:
        if value is None:
            return "n/a"
        if warn_lo is not None and value < warn_lo:
            return "fail"
        if ok_hi is not None and value > ok_hi:
            return "fail"
        if value < ok_lo:
            return "warn"
        return "ok"

    out: Dict[str, str] = {}
    out["strict_vs_lenient_gap"] = band(diag.strict_vs_lenient_gap, 0.55, 0.75, warn_lo=0.45)
    out["strict_winners_fmt_ge_0_10"] = band(diag.strict_winners_fmt_ge_0_10, 0.90, None)
    out["max_wrong_fmt"] = (
        "n/a" if diag.max_wrong_fmt is None
        else ("ok" if diag.max_wrong_fmt <= 0.30 else "fail")
    )
    out["cap_hit_rate"] = (
        "n/a" if diag.cap_hit_rate is None
        else ("ok" if diag.cap_hit_rate < 0.10 else "warn")
    )
    out["entropy_collapse"] = "fail" if diag.entropy_collapse_flag else "ok"
    out["zero_advantage"] = (
        "n/a" if diag.zero_advantage_group_frac is None
        else ("ok" if diag.zero_advantage_group_frac < 0.20 else "warn")
    )
    # §7 addendum: chunk-restart proxy — flag when the min is 0 with high char length.
    out["min_chunk_restarts"] = (
        "n/a" if diag.min_chunk_restarts_in_group is None
        else ("ok" if diag.min_chunk_restarts_in_group > 0 else "warn")
    )
    # Rising confident-wrong count = SFT "verify-before-commit" prior decay.
    out["confident_wrong"] = "ok" if diag.confident_wrong_count == 0 else "warn"
    # §4.10 contradiction detection health.
    out["contradiction_rate"] = (
        "n/a" if diag.contradiction_rate is None
        else ("ok" if diag.contradiction_rate < 0.10 else "warn")
    )
    return out


__all__ = [
    "GroupDiagnostics",
    "compute_group_diagnostics",
    "health_report",
]
