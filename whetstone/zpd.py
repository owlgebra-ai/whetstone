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


#: Weight floor applied to register-card tokens inside the think segment.
#: See :func:`band_pass` for why it exists and why it is 1.0 rather than higher.
REGISTER_FLOOR_DEFAULT = 1.0


def band_pass(
    s: np.ndarray,
    gamma: float = GAMMA_INIT,
    kappa: float = KAPPA_DEFAULT,
    alpha_nov: float = ALPHA_NOV_DEFAULT,
    s_cap: float = S_CAP_DEFAULT,
    *,
    floor_mask: np.ndarray | None = None,
    floor: float = REGISTER_FLOOR_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gate, nov, w)`` for an array of surprisals ``S_t`` in nats.

    ``floor_mask`` marks positions where ``w`` is raised to at least ``floor``
    — the **register-whitelist floor**, an attested deviation from design §4.1
    added by activity 009 finding 7 and ratified by the user 2026-08-05.

    Why it is needed: every trace in the Stage-A corpus opens ``<think>\\n goal
    :``, and ``goal`` sits ~40 nats outside the original student, so the gate
    masked it in **100.0%** of 2,414 traces. `chk` was 98.4% masked. The student
    was therefore never taught the tokens that open and close a register trace,
    and at generation time its best continuation after ``<think>\\n`` became
    ``</think>`` — an empty scratchpad. No gamma fixes this: gamma must reach
    ~-42 before ``goal`` carries weight, at which point corpus masking is 0.00%
    and the gate is off entirely, which is v1's Diagnosis #1 restored.

    Why it is defensible rather than a hack: the register is **specified, not
    discovered** (CLAUDE.md). The band-pass exists to stop gradient being spent
    on *reasoning* the student cannot reach; it was never meant to refuse to
    teach notation that a human wrote down in the card. The floor applies only
    to card §2's ~15 marker types, and only inside the think segment — a marker
    in the answer is leakage (F3d) and must never be boosted.

    Why a **floor** and not a replacement: mid-line markers already earn more
    than 1.0 from the ordinary band-pass (` ⇒` sits at w 1.72). Overwriting
    would *demote* them. ``np.maximum`` is monotone, so no token loses weight.

    Why 1.0 and not the novelty maximum of 3.0: these tokens carry a ~40-nat CE
    all by themselves. At w=1.0 ``goal`` is already ~7% of its sequence's loss;
    at 3.0 it would be ~20%, and concentrating gradient on the least-predictable
    tokens is precisely the v1 failure this whole stage exists to fix.
    """
    s = np.asarray(s, dtype=np.float64)
    gate = 1.0 / (1.0 + np.exp(-kappa * (-s - gamma)))
    nov = 1.0 + alpha_nov * np.minimum(s, s_cap)
    w = gate * nov
    if floor_mask is not None:
        w = np.where(np.asarray(floor_mask, dtype=bool), np.maximum(w, floor), w)
    return gate, nov, w


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


def register_floor_mask(ids, prompt_len: int, think_start: int, think_end: int,
                        whitelist_ids) -> np.ndarray:
    """Boolean mask over **completion** positions: card tokens inside think.

    Restricted to the think span on purpose. The same token in the answer
    segment is register leakage, which F3d requires to be ~0 — flooring it there
    would train the model to do the thing the gate is checking for.
    """
    n = len(ids) - prompt_len
    m = np.zeros(n, dtype=bool)
    for i in range(think_start, think_end):
        if ids[i] in whitelist_ids:
            m[i - prompt_len] = True
    return m


__all__ = [
    "ALPHA_NOV_DEFAULT",
    "GAMMA_INIT",
    "KAPPA_DEFAULT",
    "REGISTER_FLOOR_DEFAULT",
    "S_CAP_DEFAULT",
    "band_pass",
    "masked_threshold",
    "register_floor_mask",
    "sequence_normalizer",
]
