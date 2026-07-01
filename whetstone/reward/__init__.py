"""WHETSTONE Stage-5 DAPO reward package.

Implements the reward function specified in ``WHETSTONE_STAGE5_REWARD_DESIGN.md``.
The public entrypoints are :func:`compute_reward` (per-rollout scoring) and
:func:`compute_group_diagnostics` (per-group §7 metrics).

Module layout
-------------
- :mod:`whetstone.reward.config`      — magnitude constants (§2.3, §4.*)
- :mod:`whetstone.reward.extract`     — post-``</think>`` extraction, chunking
- :mod:`whetstone.reward.tiers`       — ``r_acc`` strict / lenient / wrong (§2.1)
- :mod:`whetstone.reward.structure`   — ``r_struct`` sub-rewards (§3)
- :mod:`whetstone.reward.penalties`   — §4 penalty catalogue
- :mod:`whetstone.reward.aggregate`   — ``compute_reward`` + ``RewardBreakdown``
- :mod:`whetstone.reward.diagnostics` — §7 group-level metrics

Design invariants (from §5) — enforced at aggregation:
    (I1) ``r_fmt >= 0.10`` when ``</think>`` is present
    (I2) worst-case-strict-with-artifacts total >= verbose-lenient + 0.30
    (I3) every structural bonus gates on ``acc >= 0.5``
"""

from .aggregate import RewardBreakdown, compute_reward
from .diagnostics import GroupDiagnostics, compute_group_diagnostics
from .tiers import Tier, classify_tier

__all__ = [
    "RewardBreakdown",
    "GroupDiagnostics",
    "Tier",
    "classify_tier",
    "compute_reward",
    "compute_group_diagnostics",
]
