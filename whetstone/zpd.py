"""ZPD band-pass token weighting for Stage B (design §4.1, packet P6).

The direct fix for v1's Diagnosis #1. v1 up-weighted tokens by surprisal, which
concentrates gradient on exactly the tokens the student cannot yet reach; the
band-pass instead gates those off and boosts *bounded* novelty inside the
reachable zone:

    gate_t = sigmoid(kappa * (log pi_S(tau_t) - gamma))
    nov_t  = 1 + alpha_nov * min(S_t, s_cap)
    w_t    = gate_t * nov_t

with ``S_t = -log pi_S(tau_t)``, so ``log pi_S = -S_t``. The gate is ~1 where the
student already assigns reasonable probability and ~0 on residual spikes; the
novelty factor is capped at ``s_cap`` so a single unreachable token cannot buy
unbounded weight.

Defaults are design §12.6's: kappa=1, alpha_nov=0.5, s_cap=4 nats, and gamma
init ln(1e-4). gamma is pinned by measurement per corpus and per round
(``scripts/stageb_pin_gamma.py``) — activity 009 measured -9.2103 on the golden
corpus under the original checkpoint and found masking almost flat in gamma
(1.3% at -11.5, 4.9% at -5), so the init stands.

This module is the single definition, imported by both the pinning script and
the trainer. Two implementations of the same formula is how a gate measured at
one shape gets trained at another.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

KAPPA_DEFAULT = 1.0
ALPHA_NOV_DEFAULT = 0.5
S_CAP_DEFAULT = 4.0
GAMMA_INIT = math.log(1e-4)          # ~ -9.2103


def band_pass(
    s: np.ndarray,
    gamma: float = GAMMA_INIT,
    kappa: float = KAPPA_DEFAULT,
    alpha_nov: float = ALPHA_NOV_DEFAULT,
    s_cap: float = S_CAP_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gate, nov, w)`` for an array of surprisals ``S_t`` in nats."""
    s = np.asarray(s, dtype=np.float64)
    gate = 1.0 / (1.0 + np.exp(-kappa * (-s - gamma)))
    nov = 1.0 + alpha_nov * np.minimum(s, s_cap)
    return gate, nov, gate * nov


def masked_threshold(gamma: float, kappa: float = KAPPA_DEFAULT,
                     cut: float = 0.1) -> float:
    """Surprisal above which ``gate < cut``.

    ``sigmoid(k(-S - gamma)) < cut``  <=>  ``S > -gamma + ln((1-cut)/cut) / k``.
    Note the **plus**: the sign here was wrong once already (activity 009), and
    it put the histogram's threshold line where the gate is 0.90.
    """
    return -gamma + math.log((1.0 - cut) / cut) / kappa


def sequence_normalizer(w: np.ndarray, n_completion: int,
                        floor_frac: float = 0.25) -> float:
    """Per-sequence loss denominator: ``max(sum(w), floor_frac * n_completion)``.

    Design §4.1 normalizes by ``sum(w)``, not by token count, so a sequence is
    not penalised for having tokens gated off. The floor stops the degenerate
    end of that: a sequence whose tokens are *almost all* gated off would divide
    its few survivors by a tiny denominator and shout over every other sequence
    in the batch. Trainers log how often the floor binds — if it binds often,
    gamma is wrong for the corpus, and that is a finding rather than a knob.
    """
    return max(float(np.sum(w)), floor_frac * float(n_completion))


__all__ = [
    "ALPHA_NOV_DEFAULT",
    "GAMMA_INIT",
    "KAPPA_DEFAULT",
    "S_CAP_DEFAULT",
    "band_pass",
    "masked_threshold",
    "sequence_normalizer",
]
