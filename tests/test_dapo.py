"""Tests for the Stage-C DAPO objective (packet P7 §6).

The properties worth testing here are the ones that fail *silently* in training:
a sign error on TEA (which would destroy entropy while the curve labelled
"entropy protection" ticked up), a KL applied to the wrong segment (which would
anchor the think block and defeat the whole project), or masks that do not
partition the completion.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from whetstone.dapo import (  # noqa: E402
    CLIP_EPS_HIGH,
    CLIP_EPS_LOW,
    answer_kl,
    assert_masks_partition,
    difficulty_weight,
    dynamic_sampling_keep,
    group_advantages,
    route_advantages,
    stagec_loss,
    tea_regularizer,
    token_level_policy_loss,
)


# --- dynamic sampling -------------------------------------------------------

def test_dynamic_sampling_drops_saturated_and_hopeless_groups() -> None:
    assert dynamic_sampling_keep([True] * 8) == (False, "all_correct")
    assert dynamic_sampling_keep([False] * 8) == (False, "all_wrong")
    keep, reason = dynamic_sampling_keep([True, False] * 4)
    assert keep is True and reason == ""


def test_dynamic_sampling_keeps_a_single_success() -> None:
    """1/8 is the hardest usable group — it must not be dropped as 'all wrong'."""
    assert dynamic_sampling_keep([True] + [False] * 7)[0] is True
    assert dynamic_sampling_keep([False] * 7 + [True])[0] is True


# --- advantages -------------------------------------------------------------

def test_group_advantages_are_zero_mean() -> None:
    r = torch.tensor([1.35, 1.10, 0.10, 0.0, 1.2, 0.1, 0.1, 1.35])
    a = group_advantages(r)
    assert torch.allclose(a.mean(), torch.tensor(0.0), atol=1e-5)
    assert a[0] > 0 and a[3] < 0


def test_uniform_group_gives_zero_advantage() -> None:
    """The reason dynamic sampling exists — such a group teaches nothing."""
    a = group_advantages(torch.full((8,), 1.35))
    assert torch.allclose(a, torch.zeros(8), atol=1e-5)


def test_difficulty_weight_amplifies_hard_problems() -> None:
    assert difficulty_weight(1.0) == pytest.approx(1.0)     # saturated: no boost
    assert difficulty_weight(0.5) == pytest.approx(1.25)
    assert difficulty_weight(0.125) == pytest.approx(1.4375)  # 1/8 group


def test_amplification_touches_only_positive_think_advantages() -> None:
    adv = torch.tensor([1.0, -1.0])
    think = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    ans = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    out = route_advantages(adv, think, p_hat=0.0)   # W = 1.5

    assert out[0, 0] == pytest.approx(1.5), "positive think advantage not amplified"
    assert out[0, 2] == pytest.approx(1.0), "answer token must stay unamplified"
    assert out[1, 0] == pytest.approx(-1.0), "negative advantage must not amplify"
    assert out[1, 2] == pytest.approx(-1.0)
    assert ans.sum() > 0  # fixture sanity


# --- clip objective ---------------------------------------------------------

def test_policy_loss_pushes_up_positive_advantage_tokens() -> None:
    logp_old = torch.log(torch.tensor([[0.5, 0.5]]))
    logp = logp_old.clone().requires_grad_(True)
    adv = torch.tensor([[1.0, 1.0]])
    mask = torch.ones(1, 2)
    loss, _ = token_level_policy_loss(logp, logp_old, adv, mask)
    loss.backward()
    # Minimizing the loss must increase logp where the advantage is positive.
    assert (logp.grad < 0).all()


def test_policy_loss_pushes_down_negative_advantage_tokens() -> None:
    logp_old = torch.log(torch.tensor([[0.5, 0.5]]))
    logp = logp_old.clone().requires_grad_(True)
    loss, _ = token_level_policy_loss(
        logp, logp_old, torch.tensor([[-1.0, -1.0]]), torch.ones(1, 2)
    )
    loss.backward()
    assert (logp.grad > 0).all()


def test_clip_higher_is_asymmetric() -> None:
    assert CLIP_EPS_HIGH > CLIP_EPS_LOW


def test_token_level_normalization_weights_long_rollouts_more() -> None:
    """A 10-token rollout must contribute ~10× a 1-token rollout, not 1×."""
    logp_old = torch.zeros(2, 10)
    logp = torch.zeros(2, 10)
    adv = torch.zeros(2, 10)
    adv[0, :] = 1.0     # long rollout, all 10 tokens active
    adv[1, 0] = 1.0     # short rollout, 1 token active
    mask = torch.zeros(2, 10)
    mask[0, :] = 1.0
    mask[1, 0] = 1.0
    loss, stats = token_level_policy_loss(logp, logp_old, adv, mask)
    assert stats["n_tokens"] == 11
    # sum(-1*1 over 11 active) / 11 = -1.0 exactly; sequence-level averaging
    # would have produced the same here, so assert the denominator directly.
    assert loss == pytest.approx(-1.0)


def test_masked_tokens_do_not_contribute() -> None:
    logp = torch.zeros(1, 4)
    adv = torch.tensor([[1.0, 1.0, 99.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss, stats = token_level_policy_loss(logp, logp, adv, mask)
    assert stats["n_tokens"] == 2
    assert loss == pytest.approx(-1.0)   # the 99s are masked out


# --- TEA --------------------------------------------------------------------

def test_tea_term_increases_entropy_when_subtracted() -> None:
    """The sign test. `loss = policy − λ·L_TEA` must RAISE entropy."""
    ent = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
    logp = torch.tensor([[-0.1, -2.0, -0.3, -1.0]])
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    think = torch.tensor([[1, 1, 1, 1]])
    l_tea, _ = tea_regularizer(logp, ent, adv, think)
    (-0.05 * l_tea).backward()
    assert (ent.grad < 0).all(), (
        "subtracting λ·L_TEA must reward higher entropy; a positive gradient "
        "here means TEA is destroying the entropy it claims to protect"
    )


def test_tea_only_sees_think_tokens() -> None:
    ent = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
    logp = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    think = torch.tensor([[1, 1, 0, 0]])
    _, stats = tea_regularizer(logp, ent, adv, think)
    assert stats["n_think"] == 2
    assert stats["think_entropy_mean"] == pytest.approx(1.0), \
        "TEA read answer-segment entropy"


def test_tea_concentrates_weight_on_high_covariance_tokens() -> None:
    """Token 0 is confident AND rewarded — the one whose entropy gets spent."""
    n = 50
    logp = torch.full((1, n), -2.0)
    adv = torch.full((1, n), -0.1)
    logp[0, 0] = -0.01      # very confident
    adv[0, 0] = 3.0         # and highly rewarded
    ent = torch.rand(1, n)
    ent[0, 0] = 2.0
    think = torch.ones(1, n, dtype=torch.long)
    _, stats = tea_regularizer(logp, ent, adv, think, tau_c=1.0)
    assert stats["selected_entropy_mean"] > float(ent.mean()), \
        "TEA did not select the high-covariance token"


def test_tea_cap_bounds_any_single_token() -> None:
    n = 200
    logp = torch.full((1, n), -3.0)
    adv = torch.full((1, n), -1.0)
    logp[0, 0], adv[0, 0] = 0.0, 50.0        # would dominate a raw softmax
    think = torch.ones(1, n, dtype=torch.long)
    _, stats = tea_regularizer(logp, torch.ones(1, n), adv, think, cap_c=100.0)
    assert stats["cap_hit_frac"] > 0, "the cap never engaged on a spiked batch"


def test_tea_is_scale_free_under_uniform_weights() -> None:
    """Uniform weights must give exactly mean(H), so λ_TEA reads the same at
    any batch size."""
    for n in (16, 256):
        logp = torch.zeros(1, n)      # zero covariance everywhere -> uniform
        adv = torch.zeros(1, n)
        ent = torch.full((1, n), 0.7)
        think = torch.ones(1, n, dtype=torch.long)
        l_tea, _ = tea_regularizer(logp, ent, adv, think)
        assert float(l_tea) == pytest.approx(0.7, abs=1e-5)


def test_tea_is_inert_with_no_think_tokens() -> None:
    l_tea, stats = tea_regularizer(
        torch.zeros(1, 4), torch.ones(1, 4), torch.ones(1, 4), torch.zeros(1, 4)
    )
    assert float(l_tea) == 0.0 and stats["n_think"] == 0


# --- answer KL --------------------------------------------------------------

def test_answer_kl_is_zero_when_policy_matches_pi0() -> None:
    lp = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
    kl, stats = answer_kl(lp, lp.clone(), torch.tensor([[0, 0, 1, 1]]))
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert stats["n_answer"] == 2


def test_k3_estimator_is_never_negative() -> None:
    """A signed penalty can be farmed; k3 cannot go below zero."""
    torch.manual_seed(0)
    for _ in range(50):
        lp = torch.randn(1, 8)
        ref = torch.randn(1, 8)
        kl, _ = answer_kl(lp, ref, torch.ones(1, 8), estimator="k3")
        assert float(kl) >= -1e-6


def test_k1_estimator_can_go_negative() -> None:
    """Documents why k3 is the default."""
    lp = torch.tensor([[-3.0]])
    ref = torch.tensor([[-0.1]])
    kl, _ = answer_kl(lp, ref, torch.ones(1, 1), estimator="k1")
    assert float(kl) < 0


def test_answer_kl_ignores_think_tokens() -> None:
    """The no-style-anchor invariant: think tokens must not be pulled to π_0."""
    lp = torch.tensor([[-5.0, -5.0, -0.5, -0.5]], requires_grad=True)
    ref = torch.tensor([[-0.1, -0.1, -0.5, -0.5]])
    kl, _ = answer_kl(lp, ref, torch.tensor([[0, 0, 1, 1]]))
    kl.backward()
    assert torch.allclose(lp.grad[0, :2], torch.zeros(2)), \
        "the answer KL leaked a gradient onto think tokens"


# --- assembly ---------------------------------------------------------------

def test_stagec_loss_routes_each_term_to_its_segment() -> None:
    B, T = 2, 6
    think = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    answer = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0]])
    logp = torch.full((B, T), -0.7, requires_grad=True)
    parts = stagec_loss(
        logp=logp,
        logp_old=torch.full((B, T), -0.7),
        logp_ref=torch.full((B, T), -0.9),
        entropy=torch.full((B, T), 0.6),
        token_advantages=torch.tensor(
            [[1.0] * 6, [-1.0] * 6]),
        think_mask=think,
        answer_mask=answer,
    )
    assert set(parts.stats) >= {"loss/total", "loss/policy", "loss/tea_term",
                                "loss/kl_term", "tea/n_think", "kl/n_answer"}
    assert parts.stats["tea/n_think"] == 5
    assert parts.stats["kl/n_answer"] == 4
    parts.total.backward()
    assert logp.grad is not None


def test_mask_partition_assert_catches_overlap() -> None:
    with pytest.raises(AssertionError, match="overlap"):
        assert_masks_partition(torch.tensor([[1, 1]]), torch.tensor([[1, 0]]), [2])


def test_mask_partition_assert_catches_overcount() -> None:
    with pytest.raises(AssertionError, match="masked"):
        assert_masks_partition(torch.tensor([[1, 1]]), torch.tensor([[0, 0]]), [1])


def test_mask_partition_accepts_boundary_tokens_being_excluded() -> None:
    """<think>/</think>/EOS belong to neither segment — under-count is legal."""
    assert_masks_partition(torch.tensor([[0, 1, 1, 0]]),
                           torch.tensor([[0, 0, 0, 1]]), [4])
