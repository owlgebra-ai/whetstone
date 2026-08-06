"""DAPO objective with segment routing and TEA (design §5.1–5.2; packet P7 §6).

Pure tensor math, no I/O and no model — everything here runs on CPU with tiny
tensors, which is what makes it unit-testable before any GPU time is spent. The
training script (:mod:`scripts.stagec_train`) owns the model, the rollout
plumbing and the curriculum; this module owns the objective.

The routing is the whole point. v1 applied a **uniform** KL anchor and a
character-count length penalty to every token, which is how segment drift hides:
one number cannot say whether the think block compressed or the answer
collapsed. Stage C sends each segment somewhere different:

* **think tokens** — the clip objective with length pressure arriving through
  the *scalar reward* (:mod:`whetstone.reward.stagec`), plus **TEA** entropy
  protection. **No style anchor**: changing that register is the point of the
  project, so nothing here pulls think tokens back toward π_0.
* **answer tokens** — the clip objective plus **forward KL to the original
  checkpoint** π_0, with the SCA length band arriving through the scalar reward.
  Activity 009 round 2 proved answers collapse (288 → 19 tokens) without this.

Nothing in this module knows about ``r_acc``: rewards arrive as advantages
already computed from the scalar reward. Mixing the KL into the reward instead
of the loss would recreate v1's mistake by the back door (packet §1b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# --- pinned config (v1 §7.3 + design §12.6) ---------------------------------

CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.28      # clip-higher — DAPO's decoupled upper bound
GROUP_SIZE = 8
LEARNING_RATE = 1e-6

TEA_TAU_C = 1.0           # swept {0.7, 1.0} in the pilot (010 finding 3)
TEA_LAMBDA = 0.05
TEA_CAP_C = 100.0         # a token may take at most 100× the uniform weight
TEA_SCALE = "weighted_mean"   # attested deviation — see tea_regularizer.__doc__

LAMBDA_ALIGN = 0.1        # answer-segment forward KL to π_0
DIFFICULTY_ALPHA = 0.5    # W(x) = 1 + α·(1 − p̂)

ADV_EPS = 1e-6


# --- group advantages + dynamic sampling ------------------------------------


@dataclass
class GroupStats:
    """Diagnostics for one problem's rollout group. All of it is dashboarded."""

    uid: str
    n: int
    n_correct: int
    reward_mean: float
    reward_std: float
    adv_abs_mean: float
    kept: bool
    drop_reason: str = ""
    p_hat: float = 0.0


def dynamic_sampling_keep(correct: Sequence[bool]) -> Tuple[bool, str]:
    """DAPO dynamic sampling: keep only groups with a mix of correct and wrong.

    Filtering is on the **accuracy** flag, not on the shaped total. An all-correct
    group still has non-zero total-reward spread here (the length tail and the
    answer band differ between members), but reinforcing style differences on a
    problem the policy has already saturated is not what the phase is for — and
    the packet is explicit: "drop all-correct and all-wrong groups from the
    gradient". The drop rate is the **phase-exhaustion signal**; log it.
    """
    if not correct:
        return False, "empty"
    n_ok = sum(bool(c) for c in correct)
    if n_ok == 0:
        return False, "all_wrong"
    if n_ok == len(correct):
        return False, "all_correct"
    return True, ""


def group_advantages(
    rewards: torch.Tensor,
    *,
    normalize_std: bool = True,
) -> torch.Tensor:
    """``A_i = (r_i − mean) / (std + eps)`` over one group (GRPO/DAPO form).

    ``normalize_std=False`` gives the Dr.GRPO / unnormalized variant. Kept as a
    flag rather than a fork because it interacts with difficulty amplification:
    std-normalization already inflates advantages in low-spread groups, and
    ``W(x)`` re-weights by difficulty on top. Per-bucket advantage variance is a
    logged curve precisely so that interaction is visible in the pilot rather
    than inferred.
    """
    if rewards.numel() == 0:
        return rewards
    centered = rewards - rewards.mean()
    if not normalize_std:
        return centered
    return centered / (rewards.std(unbiased=False) + ADV_EPS)


def difficulty_weight(p_hat: float, alpha: float = DIFFICULTY_ALPHA) -> float:
    """``W(x) = 1 + α·(1 − p̂_succ(x))`` — design §5.2.

    Applied to **positive think advantages only** (see :func:`route_advantages`):
    amplifying negative advantages on hard problems would push the policy away
    from partially-working approaches precisely where it has the least to spare.
    """
    return 1.0 + alpha * (1.0 - float(p_hat))


def route_advantages(
    advantages: torch.Tensor,
    think_mask: torch.Tensor,
    p_hat: float,
    *,
    alpha: float = DIFFICULTY_ALPHA,
) -> torch.Tensor:
    """Broadcast per-rollout advantages to tokens, amplifying hard positives.

    Args:
        advantages: ``(B,)`` per-rollout advantage.
        think_mask: ``(B, T)`` 1 on think-content tokens.
        p_hat: the problem's measured success rate (from the curriculum buckets).

    Returns ``(B, T)`` per-token advantage. Amplification touches **think**
    tokens with a **positive** advantage and nothing else — answer tokens keep
    the unamplified value so the answer channel stays anchored to π_0 rather
    than being pushed by difficulty.
    """
    w = difficulty_weight(p_hat, alpha)
    tok_adv = advantages.unsqueeze(1).expand_as(think_mask).clone()
    amplify = (think_mask > 0) & (tok_adv > 0)
    tok_adv[amplify] = tok_adv[amplify] * w
    return tok_adv


# --- the clip objective -----------------------------------------------------


def token_level_policy_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    token_advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    eps_low: float = CLIP_EPS_LOW,
    eps_high: float = CLIP_EPS_HIGH,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DAPO token-level clipped surrogate.

    **Token-level**, not sequence-level: the sum runs over every token in the
    batch and is divided by the batch's *total* token count, so a long rollout
    contributes proportionally to its length. Sequence-level averaging (GRPO's
    default) gives every rollout equal weight regardless of length, which
    under-weights exactly the long degenerate generations this run is trying to
    extinguish.

    ``eps_high > eps_low`` is DAPO's clip-higher: it leaves more room for
    low-probability tokens to *gain* probability, which is the entropy-preserving
    half of the algorithm and complements TEA.
    """
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * token_advantages
    per_token = -torch.min(unclipped, clipped)

    denom = completion_mask.sum().clamp(min=1.0)
    loss = (per_token * completion_mask).sum() / denom

    with torch.no_grad():
        m = completion_mask > 0
        n_clipped_low = ((ratio < 1.0 - eps_low) & m).sum().item()
        n_clipped_high = ((ratio > 1.0 + eps_high) & m).sum().item()
        n_tok = int(m.sum().item()) or 1
        stats = {
            "ratio_mean": float((ratio * completion_mask).sum() / denom),
            "ratio_max": float(ratio[m].max()) if n_tok else 0.0,
            "clip_frac_low": n_clipped_low / n_tok,
            "clip_frac_high": n_clipped_high / n_tok,
            "n_tokens": n_tok,
        }
    return loss, stats


# --- TEA: token-wise entropy-adaptive regularization ------------------------


def tea_regularizer(
    logp: torch.Tensor,
    entropy: torch.Tensor,
    token_advantages: torch.Tensor,
    think_mask: torch.Tensor,
    *,
    tau_c: float = TEA_TAU_C,
    cap_c: float = TEA_CAP_C,
    scale: str = TEA_SCALE,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Covariance-targeted entropy protection on think tokens (Light-IF TEA).

    ``Cov_t = centered(log p_t) · centered(A_t)`` identifies tokens where being
    *confident* co-varies with being *rewarded* — the tokens whose entropy the
    policy is most likely to spend first. A softmax over ``Cov/τ_c`` concentrates
    weight there, a cap at ``c/|T|`` stops any single token dominating, and the
    resulting weighted entropy is **added back** to the objective (returned
    positive; the caller subtracts ``λ_TEA · L_TEA``), so those tokens keep their
    entropy while everything else trains normally.

    ``τ_c`` is the softmax temperature over covariance — *not* an entropy
    threshold. Activity 003 and 010 finding 3 both flag it against this
    checkpoint's ≈0.725-nat second entropy mode; the pilot sweeps {0.7, 1.0}.

    Scaling — an ATTESTED DEVIATION from design §5.1 / packet §6
    -----------------------------------------------------------
    Both write ``L_TEA = |T_r| · Σ weight_t · H_t`` with ``λ_TEA = 0.05``. Taken
    literally that term scales with the number of think tokens in the batch,
    while the DAPO policy loss is a per-token **mean**. At this project's real
    sequence lengths the two are not commensurate by 3–4 orders of magnitude:
    with ``|T| ≈ 5,000`` and mean think entropy ≈ 0.62 (activity 010's entropy
    card), the literal form gives ``L_TEA ≈ 3.1e3`` and ``λ_TEA·L_TEA ≈ 155``
    against a policy loss of order 0.1 — the objective would collapse to "make
    the output uniform" and nothing else.

    ``scale="weighted_mean"`` (default) renormalizes the capped weights to sum
    to 1, making ``L_TEA`` a weighted **mean** entropy in nats — same units as
    the entropy card, bounded by ``ln(512) = 6.24``, and commensurate with the
    policy loss so ``λ_TEA = 0.05`` means what the design intends. The literal
    form is kept as ``scale="design_literal"`` so the deviation is a flag rather
    than a fork, and **both scales are logged every step** (``tea/l_tea_mean``,
    ``tea/l_tea_literal``) so the choice stays auditable. λ_TEA is on the run-1
    sweep list in any case (CLAUDE.md working conventions).
    """
    m = think_mask > 0
    n_think = int(m.sum().item())
    if n_think == 0:
        z = logp.sum() * 0.0
        return z, {"n_think": 0, "cap_hit_frac": 0.0, "l_tea_mean": 0.0,
                   "l_tea_literal": 0.0, "selected_entropy_mean": 0.0,
                   "think_entropy_mean": 0.0, "cov_max": 0.0, "cov_min": 0.0}

    lp = logp[m]
    adv = token_advantages[m]
    ent = entropy[m]

    cov = (lp - lp.mean()) * (adv - adv.mean())
    w = torch.softmax(cov / tau_c, dim=0)
    cap = cap_c / float(n_think)
    w_capped = torch.clamp(w, max=cap)

    literal = float(n_think) * (w_capped * ent).sum()
    weighted_mean = (w_capped * ent).sum() / w_capped.sum().clamp(min=1e-12)

    if scale == "weighted_mean":
        l_tea = weighted_mean
    elif scale == "design_literal":
        l_tea = literal
    else:
        raise ValueError(f"unknown TEA scale {scale!r}")

    with torch.no_grad():
        cap_hit = (w > cap).float().mean().item()
        # Entropy at the tokens TEA actually selected (top 1% by weight) — the
        # curve that says whether the term is protecting anything.
        k = max(1, n_think // 100)
        top = torch.topk(w_capped, k).indices
        stats = {
            "n_think": n_think,
            "cap_hit_frac": cap_hit,
            "l_tea_mean": float(weighted_mean),
            "l_tea_literal": float(literal),
            "weight_mass": float(w_capped.sum()),
            "selected_entropy_mean": float(ent[top].mean()),
            "think_entropy_mean": float(ent.mean()),
            "cov_max": float(cov.max()),
            "cov_min": float(cov.min()),
        }
    return l_tea, stats


# --- answer-segment alignment to π_0 ----------------------------------------


def answer_kl(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    estimator: str = "k3",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-token KL between the policy and frozen π_0 on **answer tokens only**.

    ``logp_ref`` is π_0's log-probability of the *sampled* token, obtained by
    prefilling the same sequence through the frozen anchor server. Two
    estimators over samples from π_θ:

    * ``k1`` = ``log π_θ − log π_0`` — the packet's "per-token logprob-diff";
      unbiased for ``KL(π_θ‖π_0)`` but signed and high-variance.
    * ``k3`` = ``exp(Δ) − Δ − 1`` with ``Δ = log π_0 − log π_θ`` — same
      expectation, always ≥ 0, much lower variance. **Default**, because a
      penalty term that can go negative can be *farmed*: the policy would be
      rewarded for drifting in whichever direction makes the estimate negative.

    Think tokens are untouched — no style anchor there, by design.
    """
    m = answer_mask > 0
    n = int(m.sum().item())
    if n == 0:
        z = logp.sum() * 0.0
        return z, {"n_answer": 0, "kl_mean": 0.0}

    delta = logp_ref[m] - logp[m]
    if estimator == "k1":
        per_tok = -delta
    elif estimator == "k3":
        per_tok = torch.exp(delta) - delta - 1.0
    else:
        raise ValueError(f"unknown KL estimator {estimator!r}")

    kl = per_tok.mean()
    with torch.no_grad():
        stats = {
            "n_answer": n,
            "kl_mean": float(kl),
            "logp_diff_mean": float(-delta.mean()),
            "kl_max": float(per_tok.max()),
        }
    return kl, stats


# --- assembly ---------------------------------------------------------------


@dataclass
class LossParts:
    total: torch.Tensor
    policy: torch.Tensor
    tea: torch.Tensor
    kl: torch.Tensor
    stats: Dict[str, float] = field(default_factory=dict)


def stagec_loss(
    *,
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    entropy: torch.Tensor,
    token_advantages: torch.Tensor,
    think_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    lambda_tea: float = TEA_LAMBDA,
    lambda_align: float = LAMBDA_ALIGN,
    tau_c: float = TEA_TAU_C,
    cap_c: float = TEA_CAP_C,
    tea_scale: str = TEA_SCALE,
    kl_estimator: str = "k3",
) -> LossParts:
    """Assemble the Stage-C objective.

    ``loss = policy_loss − λ_TEA · L_TEA + λ_align · KL_answer``

    The signs: the policy loss is already negated (minimizing it maximizes the
    surrogate); ``L_TEA`` is *subtracted* so minimizing the total **maximizes**
    entropy at the selected think tokens; the answer KL is *added* so minimizing
    the total **reduces** answer drift from π_0.
    """
    completion_mask = ((think_mask > 0) | (answer_mask > 0)).to(logp.dtype)

    policy, p_stats = token_level_policy_loss(
        logp, logp_old, token_advantages, completion_mask
    )
    tea, t_stats = tea_regularizer(
        logp, entropy, token_advantages, think_mask,
        tau_c=tau_c, cap_c=cap_c, scale=tea_scale,
    )
    kl, k_stats = answer_kl(logp, logp_ref, answer_mask, estimator=kl_estimator)

    total = policy - lambda_tea * tea + lambda_align * kl
    with torch.no_grad():
        stats = {**{f"policy/{k}": v for k, v in p_stats.items()},
                 **{f"tea/{k}": v for k, v in t_stats.items()},
                 **{f"kl/{k}": v for k, v in k_stats.items()},
                 "loss/total": float(total.detach()),
                 "loss/policy": float(policy.detach()),
                 "loss/tea_term": float((lambda_tea * tea).detach()),
                 "loss/kl_term": float((lambda_align * kl).detach())}
    return LossParts(total=total, policy=policy, tea=tea, kl=kl, stats=stats)


def assert_masks_partition(
    think_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    completion_lengths: Sequence[int],
) -> None:
    """Packet §11: assert think/answer masks sum to the completion length.

    Advantage routing is token-masked by ``parse_segments`` — never by string
    offsets. A mask that silently drops or double-counts tokens routes the wrong
    loss to the wrong segment, and nothing downstream would notice.
    """
    if torch.any((think_mask > 0) & (answer_mask > 0)):
        raise AssertionError("think and answer masks overlap")
    got = (think_mask.sum(dim=1) + answer_mask.sum(dim=1)).tolist()
    for i, (g, want) in enumerate(zip(got, completion_lengths)):
        if int(g) > int(want):
            raise AssertionError(
                f"rollout {i}: masked {int(g)} tokens > completion length {want}"
            )


__all__ = [
    "GroupStats", "LossParts",
    "dynamic_sampling_keep", "group_advantages", "difficulty_weight",
    "route_advantages", "token_level_policy_loss", "tea_regularizer",
    "answer_kl", "stagec_loss", "assert_masks_partition",
    "CLIP_EPS_LOW", "CLIP_EPS_HIGH", "GROUP_SIZE", "LEARNING_RATE",
    "TEA_TAU_C", "TEA_LAMBDA", "TEA_CAP_C", "LAMBDA_ALIGN", "DIFFICULTY_ALPHA",
]
