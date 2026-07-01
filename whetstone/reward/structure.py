"""``r_struct`` sub-rewards (§3).

Each sub-reward is its own function so the smoke-test checklist in §8 can hit
every branch independently.

Design contract (§3, §4.4, §5.3):
    Every *bonus* — ``short_clean_bonus`` and ``chunk_restart_present`` —
    gates on ``acc >= 0.5``. This is a **behavioural** gate: a wrong rollout
    NEVER earns a bonus regardless of how compact/clean its structure is.
    Failing this gate makes compact-wrong outrank verbose-correct on the
    format channel (§9.3).

Baseline structural signals (``has_closed_think``, ``terminal_answer_boxed``,
``terminal_answer_bare``) are NOT gated on accuracy — they measure well-
formedness of the completion shape, which is orthogonal to correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (
    BONUS_ACC_GATE,
    CHUNK_RESTART_MIN_COUNT,
    RewardConfig,
    SHORT_CLEAN_CHUNK_COUNT_MAX,
    STRUCT_CHUNK_RESTART_PRESENT,
    STRUCT_CLOSED_THINK,
    STRUCT_SHORT_CLEAN_BONUS,
    STRUCT_TERMINAL_BARE,
    STRUCT_TERMINAL_BOXED,
)
from .extract import (
    SplitCompletion,
    count_chunk_restarts,
    split_chunks,
    terminal_is_bare,
    terminal_is_boxed,
)


@dataclass(frozen=True)
class StructureBreakdown:
    """Full breakdown of ``r_struct`` (§3) with each sub-reward isolated.

    Every field is either 0 or the specific magnitude from :mod:`.config`.
    Sum = total ``r_struct`` before penalties.
    """

    closed_think: float
    terminal_boxed: float
    terminal_bare: float
    short_clean_bonus: float
    chunk_restart_present: float

    # Sub-flags — for §7 diagnostics and smoke tests. Do NOT re-derive from
    # magnitudes in downstream code (rounding + config overrides would drift).
    has_closed_think: bool
    is_terminal_boxed: bool
    is_terminal_bare: bool
    is_short_clean: bool
    has_chunk_restart: bool

    @property
    def total(self) -> float:
        return (
            self.closed_think
            + self.terminal_boxed
            + self.terminal_bare
            + self.short_clean_bonus
            + self.chunk_restart_present
        )


def _has_closed_think(split: SplitCompletion, cfg: RewardConfig) -> float:
    return cfg.struct_closed_think if split.has_closed_think else 0.0


def _terminal_boxed(split: SplitCompletion, cfg: RewardConfig) -> float:
    # §6.11: operate on final_block, not raw post_think.
    return cfg.struct_terminal_boxed if terminal_is_boxed(split.final_block) else 0.0


def _terminal_bare(split: SplitCompletion, cfg: RewardConfig) -> float:
    # §6.11: operate on final_block, not raw post_think.
    return cfg.struct_terminal_bare if terminal_is_bare(split.final_block) else 0.0


def _short_clean_bonus(
    split: SplitCompletion,
    r_acc_value: float,
    cfg: RewardConfig,
) -> tuple[float, bool]:
    """§3 short-clean bonus (acc-gated).

    Fires iff:
      (1) ``r_acc >= BONUS_ACC_GATE`` (§3, §4.4, §5.3)
      (2) chunk count of the ``<think>`` body is <= 25 (chunk-count gating is
          more robust than think-token gating — §3 note)
      (3) terminal answer is boxed OR bare in the §6.11 final block
    """
    if r_acc_value < cfg.bonus_acc_gate:
        return 0.0, False
    chunk_count = len(split_chunks(split.think))
    if chunk_count > cfg.short_clean_chunk_count_max:
        return 0.0, False
    if not (terminal_is_boxed(split.final_block) or terminal_is_bare(split.final_block)):
        return 0.0, False
    return cfg.struct_short_clean_bonus, True


def _chunk_restart_present(
    split: SplitCompletion,
    r_acc_value: float,
    cfg: RewardConfig,
) -> tuple[float, bool]:
    """§3 chunk_restart_present (acc-gated register-preservation signal).

    Rewards a Gemma-4-style numbered compact register. For base models with a
    different native register, override :attr:`RewardConfig.struct_chunk_restart_present`
    to zero and add the equivalent structural marker elsewhere.
    """
    if r_acc_value < cfg.bonus_acc_gate:
        return 0.0, False
    restarts = count_chunk_restarts(split.think)
    if restarts < CHUNK_RESTART_MIN_COUNT:
        return 0.0, False
    return cfg.struct_chunk_restart_present, True


def compute_structure(
    split: SplitCompletion,
    r_acc_value: float,
    cfg: Optional[RewardConfig] = None,
) -> StructureBreakdown:
    """Return the full ``r_struct`` breakdown for one rollout."""
    cfg = cfg or RewardConfig()

    closed = _has_closed_think(split, cfg)
    boxed = _terminal_boxed(split, cfg)
    bare = _terminal_bare(split, cfg)
    short_clean_v, short_clean_flag = _short_clean_bonus(split, r_acc_value, cfg)
    restart_v, restart_flag = _chunk_restart_present(split, r_acc_value, cfg)

    return StructureBreakdown(
        closed_think=closed,
        terminal_boxed=boxed,
        terminal_bare=bare,
        short_clean_bonus=short_clean_v,
        chunk_restart_present=restart_v,
        has_closed_think=split.has_closed_think,
        is_terminal_boxed=bool(boxed),
        is_terminal_bare=bool(bare),
        is_short_clean=short_clean_flag,
        has_chunk_restart=restart_flag,
    )


__all__ = ["StructureBreakdown", "compute_structure"]
