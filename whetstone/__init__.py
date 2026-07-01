"""WHETSTONE reference utilities.

The deterministic verifier lives in :mod:`whetstone.verify`; the Stage-5 DAPO
reward function lives in :mod:`whetstone.reward` and is re-exported here so
training scripts can ``from whetstone import compute_reward``.
"""

from .reward import (
    GroupDiagnostics,
    RewardBreakdown,
    Tier,
    compute_group_diagnostics,
    compute_reward,
)
from .verify import extract_answer, verify_response

__all__ = [
    "verify_response",
    "extract_answer",
    "compute_reward",
    "compute_group_diagnostics",
    "RewardBreakdown",
    "GroupDiagnostics",
    "Tier",
]
