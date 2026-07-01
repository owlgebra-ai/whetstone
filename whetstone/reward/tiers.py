"""``r_acc`` tier classifier — strict / lenient / wrong (§2.1).

The three-tier structure is the single largest lever in the reward function.
This module keeps the classification logic tight and easy to audit against
§2.1 invariants:

    (1) Both strict and lenient use **post-``</think>`` extraction only**.
        ``whetstone.verify.verify_response`` already enforces this via
        ``_strip_think`` — we do not scan ``<think>`` here.
    (2) Lenient additionally requires the gold to appear in the *last-K*
        window of the post-think (or as the last-occurring numeric match)
        — otherwise a multi-answer markdown dump that happens to contain
        gold would score as lenient.
    (3) Strict requires the final block to be *clean* (bare numeric / boxed
        with no prose) — see :func:`whetstone.reward.extract.is_clean_post_think`.
    (4) §6.7: prose-templated finalizers (``**Answer:** X``, ``Answer: X``,
        ``The X is N.``) get a strict-tier fallback after the base verifier
        rejects. Without this ~10–30% of problem classes have a structurally
        unreachable strict ceiling.
    (5) §4.10 strict-gate mode: if enabled, a mismatch between the last
        numeric in ``<think>`` tail and the terminal answer demotes the
        rollout from strict to lenient (redemption-recap attractor mitigation).

Verifier hygiene (§6) is delegated to :func:`whetstone.verify.verify_response`.
Do not add answer normalisation here — extend the verifier instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from whetstone.verify import verify_response

from .config import LENIENT_LAST_K_CHARS, R_ACC_LENIENT, R_ACC_STRICT, R_ACC_WRONG, RewardConfig
from .extract import (
    SplitCompletion,
    extract_last_numeric_in_think_tail,
    extract_terminal_answer_from_post_think,
    is_clean_post_think,
    last_k_window,
    numerical_agree,
    prose_templated_matches_gold,
    split_think_close,
)


class Tier(str, Enum):
    """Accuracy tier assigned to a rollout."""

    STRICT = "strict"
    LENIENT = "lenient"
    WRONG = "wrong"


@dataclass(frozen=True)
class AccuracyResult:
    """Full outcome of tier classification.

    ``value`` is what feeds into ``compute_reward``. ``tier`` and the sub-flags
    are for diagnostics and §7 metrics — do not use them to compute reward
    directly (that path must go through :func:`classify_tier`).

    Extra fields (§6.7, §4.10):
      * ``prose_templated_accepted`` — the primary verifier rejected but §6.7
        prose-templated fallback matched. Only set on strict rollouts that
        would otherwise have been wrong.
      * ``contradiction_detected`` — think-tail last numeric disagrees with
        the terminal answer. Fires regardless of the config's contradiction
        mode; the mode governs *reaction*, not detection.
      * ``strict_gate_demoted`` — the rollout was accepted as strict-clean
        but demoted to lenient because of ``contradiction_detected`` under
        the ``strict_gate`` contradiction mode.
    """

    tier: Tier
    value: float
    verifier_accepted: bool
    post_clean: bool
    gold_in_last_k: bool
    prose_templated_accepted: bool = False
    contradiction_detected: bool = False
    strict_gate_demoted: bool = False


def _gold_in_last_k(post_think: str, gold: str, k: int) -> bool:
    """Return True iff ``gold`` occurs in the last-K window of ``post_think``.

    §2.1 note (2): "require the gold to appear in the *last-K* window of the
    post-think (or as the last-occurring numeric match)". We satisfy the first
    clause literally — the second clause is redundant when K is large enough
    because the last-occurring numeric match is inside the last-K window by
    construction.
    """
    if not gold or not post_think:
        return False
    window = last_k_window(post_think, k)
    if not window:
        return False
    # Bare-integer golds use a word-boundary + non-backslash-trail check per
    # §6.3 to reject ``24\sqrt{2}`` matching gold ``24``.
    if re.fullmatch(r"-?\d+", gold.strip()):
        pattern = r"\b" + re.escape(gold.strip()) + r"(?!\\)"
        return re.search(pattern, window) is not None
    return gold.strip() in window


def _detect_contradiction(split: SplitCompletion, cfg: RewardConfig) -> bool:
    """§4.10: True iff last numeric in ``<think>`` tail disagrees with terminal answer.

    Returns False when either side is missing (can't detect contradiction
    with insufficient evidence — avoids false-positives on cap-hit / non-
    numeric rollouts).
    """
    if cfg.contradiction_mode == "off":
        return False
    think_last = extract_last_numeric_in_think_tail(split.think, cfg.contradiction_think_tail_chars)
    post_terminal = extract_terminal_answer_from_post_think(split.post_think)
    if think_last is None or post_terminal is None:
        return False
    return not numerical_agree(
        think_last,
        post_terminal,
        rel_tol=cfg.contradiction_numeric_rel_tol,
        abs_tol=cfg.contradiction_numeric_abs_tol,
    )


def classify_tier(
    completion: str,
    gold: str,
    *,
    split: Optional[SplitCompletion] = None,
    last_k_chars: int = LENIENT_LAST_K_CHARS,
    cfg: Optional[RewardConfig] = None,
) -> AccuracyResult:
    """Classify a rollout into strict / lenient / wrong.

    Order of operations:
      1. Primary verifier → accepted / rejected.
      2. If rejected AND §6.7 fallback enabled, try prose-templated extractor
         on the final block's terminal line. On match, treat as accepted +
         set ``prose_templated_accepted``.
      3. Classify: accepted + clean-final-block → strict; accepted + gold-in-
         last-K → lenient; else wrong.
      4. §4.10 strict-gate: if enabled and contradiction detected on a strict
         rollout, demote to lenient.

    Parameters
    ----------
    completion : str
        Full rollout text including ``<think>...</think>`` (if present).
    gold : str
        Ground-truth answer string. May be numeric, symbolic, LaTeX, or
        a letter for MCQ; canonicalization lives in the verifier.
    split : SplitCompletion, optional
        Pre-computed ``</think>`` split (avoids re-splitting).
    last_k_chars : int
        Size of the lenient last-K window (§6.1 K ≈ 500).
    cfg : RewardConfig, optional
        Feature flags for §6.7 fallback and §4.10 strict-gate. Falls back
        to :class:`RewardConfig` defaults when omitted.

    Returns
    -------
    AccuracyResult
    """
    cfg = cfg or RewardConfig()
    if split is None:
        split = split_think_close(completion)

    accepted = verify_response(completion, gold) if gold else False
    prose_templated_accepted = False

    if not accepted and cfg.enable_prose_templated_extractor and gold:
        # §6.7 strict fallback — try prose-templated finalizer on the final block.
        if prose_templated_matches_gold(split.final_block, gold) is not None:
            accepted = True
            prose_templated_accepted = True

    # §6.11: cleanliness scored on the final_block, not raw post-think.
    post_clean = is_clean_post_think(split.final_block)
    gold_present = _gold_in_last_k(split.post_think, gold, last_k_chars)

    contradiction = _detect_contradiction(split, cfg) if accepted else False

    if accepted and post_clean:
        # §4.10 strict-gate: demote strict → lenient on contradiction.
        if contradiction and cfg.contradiction_mode == "strict_gate":
            return AccuracyResult(
                tier=Tier.LENIENT,
                value=R_ACC_LENIENT,
                verifier_accepted=True,
                post_clean=True,
                gold_in_last_k=gold_present,
                prose_templated_accepted=prose_templated_accepted,
                contradiction_detected=True,
                strict_gate_demoted=True,
            )
        return AccuracyResult(
            tier=Tier.STRICT,
            value=R_ACC_STRICT,
            verifier_accepted=True,
            post_clean=True,
            gold_in_last_k=gold_present,
            prose_templated_accepted=prose_templated_accepted,
            contradiction_detected=contradiction,
        )

    if accepted and gold_present:
        return AccuracyResult(
            tier=Tier.LENIENT,
            value=R_ACC_LENIENT,
            verifier_accepted=True,
            post_clean=False,
            gold_in_last_k=True,
            prose_templated_accepted=prose_templated_accepted,
            contradiction_detected=contradiction,
        )

    # verify_response accepted BUT gold isn't in the last-K window: reject
    # per §2.1 note (2). This is the multi-answer markdown-dump safeguard.
    return AccuracyResult(
        tier=Tier.WRONG,
        value=R_ACC_WRONG,
        verifier_accepted=accepted,
        post_clean=post_clean,
        gold_in_last_k=gold_present,
        prose_templated_accepted=prose_templated_accepted,
        contradiction_detected=contradiction,
    )


__all__ = ["Tier", "AccuracyResult", "classify_tier"]
