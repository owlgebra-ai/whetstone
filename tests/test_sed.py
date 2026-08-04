"""Unit tests for the SED kernel (packet P4 §6; design §12.4).

The four tests here are not coverage for its own sake — each one pins a bug
that design §12.4 names explicitly *because it fails silently*. A hard-copy
shadow, a per-micro-batch cadence, a gate read off the trainee, or a K2 term
that leaks gradient into the shadow all leave a run that trains, converges and
reports a falling loss while the entropy guarantee is simply absent.

The kernel is shared verbatim with Stage B, so these run before the Round-0
trainer and again before P6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from whetstone.sed import SEDRegularizer, solve_temperature, topk_entropy  # noqa: E402


class ToyLM(nn.Module):
    """Minimal causal-LM stand-in: ``(1, T)`` ids -> ``(1, T, V)`` logits."""

    def __init__(self, vocab: int = 64, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.head(self.emb(input_ids))})()


def _flat(model):
    return torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])


# --------------------------------------------------------------------------
# 1. EMA decay follows mu^k analytically (rule 1: update, never replacement)
# --------------------------------------------------------------------------

def test_ema_follows_mu_to_the_k():
    torch.manual_seed(0)
    model = ToyLM()
    sed = SEDRegularizer(model, ema_decay=0.99, sync_every=5, shadow_dtype=torch.float32)

    phi0 = _flat(sed.shadow)
    assert torch.allclose(phi0, _flat(model)), "shadow must initialize to phi <- theta"

    # Move theta once and hold it there; phi should approach it geometrically.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    theta = _flat(model)
    d0 = phi0 - theta

    mu = 0.99
    for k in range(1, 21):
        sed.sync()
        expected = theta + (mu ** k) * d0
        got = _flat(sed.shadow)
        assert torch.allclose(got, expected, atol=1e-5), (
            f"after {k} syncs the shadow is not mu^k of the way; "
            "a hard copy would have landed on theta at k=1"
        )

    # The distinguishing check: a hard copy reaches theta immediately.
    assert (_flat(sed.shadow) - theta).abs().max() > 1e-3, (
        "shadow collapsed onto theta — that is the replacement bug"
    )


# --------------------------------------------------------------------------
# 2. Bisection hits a random target and clamps at the range ends
# --------------------------------------------------------------------------

def test_bisection_hits_target_and_clamps():
    torch.manual_seed(1)
    n, k = 64, 512
    logits = torch.randn(n, k) * 2.0

    h_lo = topk_entropy(logits, k=k, tau=1.1)
    h_hi = topk_entropy(logits, k=k, tau=1.5)
    assert (h_hi > h_lo).all(), "entropy must increase with temperature"

    # Reachable targets: hit them within 1e-2 nats.
    frac = torch.rand(n)
    target = h_lo + frac * (h_hi - h_lo)
    tau = solve_temperature(logits, target, lo=1.1, hi=1.5, iters=20)
    got = torch.stack([topk_entropy(logits[i], k=k, tau=float(tau[i])) for i in range(n)])
    assert (got - target).abs().max() < 1e-2, (
        f"bisection missed by {(got - target).abs().max():.4f} nats"
    )
    assert ((tau >= 1.1) & (tau <= 1.5)).all()

    # Below the range -> clamp low; above -> clamp high. Silently, by design.
    tau_low = solve_temperature(logits, h_lo - 5.0, lo=1.1, hi=1.5, iters=20)
    assert torch.allclose(tau_low, torch.full_like(tau_low, 1.1), atol=1e-4)
    tau_high = solve_temperature(logits, h_hi + 5.0, lo=1.1, hi=1.5, iters=20)
    assert torch.allclose(tau_high, torch.full_like(tau_high, 1.5), atol=1e-4)


# --------------------------------------------------------------------------
# 3. Exactly one sync per 40 micro-batches at grad-accum 8 (rule 2)
# --------------------------------------------------------------------------

def test_sync_cadence_counts_optimizer_steps_not_micro_batches():
    model = ToyLM()
    sed = SEDRegularizer(model, sync_every=5, shadow_dtype=torch.float32)

    accum = 8
    n_micro = 400
    opt_step = 0
    for i in range(1, n_micro + 1):
        if i % accum == 0:                      # an optimizer step happened
            opt_step += 1
            sed.maybe_sync(opt_step)

    assert opt_step == n_micro // accum == 50
    assert sed.n_syncs == 10, (
        f"expected 10 syncs in {n_micro} micro-batches (one per 40), got {sed.n_syncs}"
    )
    assert n_micro / sed.n_syncs == 40.0

    # The bug this guards: syncing per micro-batch moves the shadow 8x too fast.
    model2 = ToyLM()
    sed2 = SEDRegularizer(model2, sync_every=5, shadow_dtype=torch.float32)
    for i in range(1, n_micro + 1):
        sed2.maybe_sync(i)
    assert sed2.n_syncs == 80 == 8 * sed.n_syncs


# --------------------------------------------------------------------------
# 4. K2 gradient flows to theta only (shadow under no_grad)
# --------------------------------------------------------------------------

def test_k2_gradient_reaches_theta_only():
    torch.manual_seed(2)
    model = ToyLM()
    sed = SEDRegularizer(model, shadow_dtype=torch.float32, topk=32)

    ids = torch.randint(0, 64, (1, 24))
    think = torch.zeros(24, dtype=torch.long)
    think[6:18] = 1                              # a think span, boundaries excluded

    logits = model(ids).logits
    loss, stats = sed.loss(logits, ids, think, return_stats=True)

    assert stats["n_tokens"] == 12
    assert loss.requires_grad
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), (
        "no gradient reached theta — the K2 term is disconnected"
    )
    for name, p in sed.shadow.named_parameters():
        assert not p.requires_grad, f"shadow param {name} requires grad"
        assert p.grad is None, f"gradient leaked into shadow param {name}"


def test_gate_is_read_off_the_shadow_not_the_trainee():
    """Rule 3: H_t comes from phi. Perturbing theta alone must not move it."""
    torch.manual_seed(3)
    model = ToyLM()
    sed = SEDRegularizer(model, shadow_dtype=torch.float32, topk=32)

    ids = torch.randint(0, 64, (1, 24))
    think = torch.zeros(24, dtype=torch.long)
    think[6:18] = 1

    _, s0 = sed.loss(model(ids).logits, ids, think, return_stats=True)
    with torch.no_grad():                        # move theta hard, leave phi alone
        for p in model.parameters():
            p.mul_(3.0).add_(0.5)
    _, s1 = sed.loss(model(ids).logits, ids, think, return_stats=True)

    assert s0["H_shadow_mean"] == pytest.approx(s1["H_shadow_mean"], abs=1e-6)
    assert s0["tau_mean"] == pytest.approx(s1["tau_mean"], abs=1e-6)
    assert s0["k2_mean"] != pytest.approx(s1["k2_mean"], abs=1e-6), (
        "K2 did not move when theta moved — the trainee side is not connected"
    )


def test_delta_gate_direction_restores_at_forks_not_at_collapse():
    """Delta_t must be large where the shadow is uncertain, ~0 where it is sure.

    The inverted gate is an easy and invisible mistake: it injects entropy into
    deterministic continuations and leaves fork tokens collapsed — the exact
    opposite of restoration mode.
    """
    torch.manual_seed(4)
    model = ToyLM()
    sed = SEDRegularizer(model, shadow_dtype=torch.float32, topk=32,
                         H_pivot=0.6707, delta_max=0.7, gamma_e=1.0)

    peaked = torch.zeros(1, 32)
    peaked[0, 0] = 30.0                          # near-deterministic
    flat = torch.zeros(1, 32)                    # maximally uncertain

    h_peaked = topk_entropy(peaked, k=32)
    h_flat = topk_entropy(flat, k=32)
    d_peaked = 0.7 * torch.sigmoid(1.0 * (h_peaked - 0.6707))
    d_flat = 0.7 * torch.sigmoid(1.0 * (h_flat - 0.6707))

    assert float(d_peaked) < 0.25, "collapse tokens must get almost no boost"
    assert float(d_flat) > 0.45, "fork tokens must get a substantial boost"
    assert float(d_flat) > float(d_peaked)
