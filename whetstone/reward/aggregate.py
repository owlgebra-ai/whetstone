"""Reward aggregator — ``r(τ) = r_acc(τ) + r_fmt(τ)``.

Combines the accuracy tier (§2.1), the structural sub-rewards (§3), and the
penalty stack (§4) into a single scalar, and enforces the invariants from §5:

    (I1) §5.1 — ``r_fmt >= 0.10`` when ``</think>`` is present, applied at
         *final* aggregation, not per-penalty.
    (I2) §5.2 — worst-case-strict-with-artifacts total stays above
         ``verbose-lenient + 0.30``. Provided as an executable design-contract
         check (:func:`design_contract_worst_case`).
    (I3) §5.3 — every structural bonus is acc-gated. Enforced in
         :mod:`.structure` at the sub-reward level.

The returned :class:`RewardBreakdown` carries every intermediate signal so the
training loop can dashboard §7 metrics directly from it — no need to re-parse
the completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import RewardConfig
from .extract import SplitCompletion, split_chunks, split_think_close
from .penalties import PenaltyResult, run_all_penalties
from .structure import StructureBreakdown, compute_structure
from .tiers import AccuracyResult, Tier, classify_tier


@dataclass(frozen=True)
class RewardBreakdown:
    """Complete audit trail of one rollout's reward.

    Attributes
    ----------
    total : float
        The scalar returned to DAPO: ``r_acc + r_fmt``.
    r_acc : float
        Accuracy component (§2.1). One of ``{1.0, 0.5, 0.0}`` under the
        default config.
    r_fmt : float
        Format component after floor: ``max(floor, r_struct − Σ penalties)``.
    r_struct_total : float
        Sum of structural sub-rewards *before* penalties are subtracted.
    penalties_subtracted : float
        The value that was subtracted from ``r_struct_total`` (bonuses
        contribute negatively). May be > (r_struct − floor); the floor
        clips the resulting ``r_fmt``.
    floor_applied : bool
        True iff the 0.10 (or 0.0) floor was active — i.e. raw
        ``r_struct − Σ penalties`` fell below the floor.
    tier : Tier
    accuracy : AccuracyResult
    structure : StructureBreakdown
    penalties : list[PenaltyResult]
    chunk_count : int
        Convenience: ``len(split_chunks(think))`` — cached for §7 metrics.
    """

    total: float
    r_acc: float
    r_fmt: float
    r_struct_total: float
    penalties_subtracted: float
    floor_applied: bool
    tier: Tier
    accuracy: AccuracyResult
    structure: StructureBreakdown
    penalties: List[PenaltyResult]
    chunk_count: int
    detail: Dict[str, object] = field(default_factory=dict)


def _apply_floor(
    r_struct_total: float,
    penalties_sum: float,
    has_closed_think: bool,
    cfg: RewardConfig,
) -> tuple[float, bool]:
    """Apply §5.1 floor at final aggregation.

    Returns ``(r_fmt, floor_applied)`` where ``floor_applied`` is True iff
    the raw ``r_struct − Σ`` fell below the floor.
    """
    raw = r_struct_total - penalties_sum
    floor = cfg.r_fmt_floor_with_close if has_closed_think else cfg.r_fmt_floor_without_close
    if raw < floor:
        return floor, True
    return raw, False


def compute_reward(
    completion: str,
    gold: str,
    *,
    cfg: Optional[RewardConfig] = None,
    group_chunk_count_variation: Optional[int] = None,
) -> RewardBreakdown:
    """Score one DAPO rollout.

    Parameters
    ----------
    completion : str
        Full rollout text including ``<think>...</think>``.
    gold : str
        Ground-truth answer for the prompt.
    cfg : RewardConfig, optional
        Override magnitudes / gates / feature flags. When ``None``,
        :class:`RewardConfig` defaults are used (§2.3, §4).
    group_chunk_count_variation : int, optional
        Chunk-count variation across the DAPO group (§4.5 (c)). Pass
        ``max(chunk_count) − min(chunk_count)`` for the group; omit
        when scoring a single rollout in isolation.

    Returns
    -------
    RewardBreakdown
        ``total`` is the DAPO reward. Every sub-component is preserved
        for §7 diagnostics and postmortems.
    """
    cfg = cfg or RewardConfig()
    split: SplitCompletion = split_think_close(completion)
    # cfg is forwarded so classify_tier sees the §6.7 prose-templated flag and
    # §4.10 contradiction_mode (strict-gate demotion) — see tiers.classify_tier.
    accuracy = classify_tier(
        completion, gold, split=split, last_k_chars=cfg.lenient_last_k_chars, cfg=cfg
    )
    structure = compute_structure(split, accuracy.value, cfg=cfg)
    penalties, penalties_sum = run_all_penalties(
        split, cfg=cfg, group_chunk_count_variation=group_chunk_count_variation
    )

    r_fmt, floor_applied = _apply_floor(
        structure.total, penalties_sum, split.has_closed_think, cfg
    )
    total = accuracy.value + r_fmt

    return RewardBreakdown(
        total=total,
        r_acc=accuracy.value,
        r_fmt=r_fmt,
        r_struct_total=structure.total,
        penalties_subtracted=penalties_sum,
        floor_applied=floor_applied,
        tier=accuracy.tier,
        accuracy=accuracy,
        structure=structure,
        penalties=penalties,
        chunk_count=len(split_chunks(split.think)),
        detail={
            "cfg_version": cfg.version,
            "has_closed_think": split.has_closed_think,
        },
    )


# --- Design-contract checks (§5.2) --------------------------------------------

def design_contract_worst_case(cfg: Optional[RewardConfig] = None) -> Dict[str, float]:
    """Compute the §5.2 worst-case invariant on paper (using magnitudes only).

    Returns a dict with:
        * ``strict_worst_case_total`` — a strict-correct rollout that trips
          every penalty simultaneously, floored at 0.10.
        * ``verbose_lenient_baseline`` — a lenient-correct rollout with the
          base ``has_closed_think`` structural reward and no penalties.
        * ``margin`` — ``strict_worst_case_total − verbose_lenient_baseline``.
        * ``passes`` — True iff margin >= 0.30 per §5.2.

    Run this at import time (in the training script) to fail-fast on
    magnitude-configuration mistakes.
    """
    cfg = cfg or RewardConfig()

    # Strict correct rollout with </think> present but every penalty tripped.
    # r_struct base: closed_think (0.15). Ignore terminal/boxed/bare/short_clean
    # because a rollout that trips every penalty likely isn't clean-terminal.
    r_struct_worst = cfg.struct_closed_think
    penalties_worst = (
        cfg.length_pen_max
        + cfg.placeholder_pen_max
        + cfg.repetition_pen
        + cfg.monolithic_pen_max
        + cfg.post_tail_repeat_pen_max
        + cfg.register_leak_pen
        + cfg.sentinel_pen
        # §4.10 weak contradiction — strict-gate mode contributes 0 here because
        # it manifests as a tier demotion, not a subtracted magnitude.
        + (cfg.contradiction_pen if cfg.contradiction_mode == "weak" else 0.0)
        + cfg.malformed_boxed_pen_max  # §6.10
        + (cfg.meta_pattern_pen if cfg.enable_meta_pattern_pen else 0.0)
        - cfg.closed_think_bonus  # closure bonus is a negative penalty
    )
    raw = r_struct_worst - penalties_worst
    r_fmt_worst = max(cfg.r_fmt_floor_with_close, raw)
    strict_worst_case_total = cfg.r_acc_strict + r_fmt_worst

    # Verbose lenient baseline: lenient acc + closed_think only.
    verbose_lenient_baseline = cfg.r_acc_lenient + cfg.struct_closed_think

    margin = strict_worst_case_total - verbose_lenient_baseline
    return {
        "strict_worst_case_total": strict_worst_case_total,
        "verbose_lenient_baseline": verbose_lenient_baseline,
        "margin": margin,
        "passes": margin >= 0.30,
    }


__all__ = [
    "RewardBreakdown",
    "compute_reward",
    "design_contract_worst_case",
]
