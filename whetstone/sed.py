"""SED — entropy-preserving self-distillation (CurioSFT; design §4.2, §12.4).

Shared **verbatim** by Round 0 (packet P4 Part 2) and Stage B (packet P6): the
same kernel, a different trainee and a different EMA copy. It is the mechanism
that stops SFT from collapsing the entropy the later RL stages need — v1's
Diagnosis #3.

The idea in one line: distill the trainee toward a *temperature-raised* copy of
its own slow-moving EMA shadow, where the temperature is chosen per token so
the target's entropy lands a gated amount above the shadow's own.

    H_t     = entropy of pi_phi at token t          (top-k, k = 512)
    Delta_t = delta_max * sigmoid(gamma_e * (H_t - H_pivot))
    tau_hat = tau such that H(pi_phi / tau) == H_t + Delta_t   (bisection)
    K2      = 0.5 * (log pi_theta(y_t) - log pi_phi,tau_hat(y_t))^2

The gate direction is deliberate and easy to get backwards. `Delta_t` is *large*
where the shadow is already uncertain (fork tokens, H_t above H_pivot) and
*vanishes* where it is confident (H_t below H_pivot). Entropy is restored at
decision points and not injected into deterministic continuations — pushing
entropy into the middle of a word is how this term turns into noise.

Every implementation rule below is a named bug from design §12.4. They are
listed there because each one fails *silently*: the loss still decreases, the
run still finishes, and the entropy guarantee is simply absent.

1.  **EMA update, never replacement.** ``phi <- 0.99*phi + 0.01*theta`` every 5
    *optimizer* steps, ``phi`` initialized to ``theta``. A hard copy every 5
    steps makes the shadow a 5-step-lagged clone — a fast-moving target, and
    the stabilization vanishes. At mu=0.99 on a 5-step cadence the averaging
    horizon is ~1/(1-mu) = 100 syncs = 500 optimizer steps.
2.  **Count optimizer steps, not micro-batches.** With grad-accum 8, "every 5
    steps" is every 40 forward/backward passes. Syncing per micro-batch moves
    the shadow 8x too fast.
3.  **Gate and temperature search run on the shadow's logits**, never the
    trainee's — one forward of ``phi`` per batch supplies both.
4.  **Round 0 and Stage B each own their EMA copy.** Never shared, never
    carried across stages.

Scope note on top-k: k=512 bounds the *entropy* computation and the bisection
(design §12.3 measures entropy this way, so H_pivot is on that scale). The two
K2 logprobs are exact full-vocab log-softmax values, so both sides of the
difference live on the same support — mixing a full-vocab trainee logprob with
a top-512-renormalized target logprob would add a systematic offset that grows
with how much mass sits outside the top 512.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pinned by activity 005 (1,200 Qwen3-1.7B register traces, 243k tokens).
H_PIVOT_DEFAULT = 0.6707
# Pinned by activity 003's audit verdict: SED runs in RESTORATION mode.
DELTA_MAX_DEFAULT = 0.7


def row_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    rows: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Logits at the *predicting* rows only — ``(N, V)``.

    Runs the transformer body over the whole sequence (the conditioning has to
    be right) and applies ``lm_head`` to just the rows that carry loss. Round 0
    supervises think tokens only — a median 150 of a 1003-token sequence — so
    computing the full ``(1, T, V)`` tensor spends most of its memory on prompt
    and answer positions that get no gradient. On this corpus's longest record
    (4,491 tokens) the full tensor is 1.36 GB before autograd saves a copy,
    which does not fit beside fp32 weights + Adam moments on a 32 GB card.

    Falls back to a plain forward for models without the ``.model``/``.lm_head``
    split (the toy models in the unit tests).
    """
    if not (hasattr(model, "model") and hasattr(model, "lm_head")):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[0, rows, :]
    out = model.model(input_ids=input_ids, attention_mask=attention_mask)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h[0, rows, :]
    return torch.cat([model.lm_head(h[s : s + chunk]) for s in range(0, h.shape[0], chunk)])


def topk_entropy(logits: torch.Tensor, k: int = 512, tau: float = 1.0) -> torch.Tensor:
    """Entropy (nats) of the top-``k``-renormalized distribution at temperature ``tau``.

    Args:
        logits: ``(..., V)``.
    Returns:
        ``(...)`` entropies.
    """
    v = torch.topk(logits.float(), k=min(k, logits.shape[-1]), dim=-1).values
    logp = F.log_softmax(v / tau, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def solve_temperature(
    topk_logits: torch.Tensor,
    target_entropy: torch.Tensor,
    *,
    lo: float = 1.1,
    hi: float = 1.5,
    iters: int = 20,
) -> torch.Tensor:
    """Per-token bisection for ``tau`` with ``H(softmax(logits/tau)) == target``.

    Entropy is monotonically increasing in ``tau``, so plain bisection converges.
    Targets outside ``[H(lo), H(hi)]`` **clamp silently at the range ends**
    (design §12.4) — the range is a deliberate bound on how hard this term may
    push, not an assertion about the data.

    Args:
        topk_logits: ``(N, k)`` — the shadow's top-k logits per token.
        target_entropy: ``(N,)``.
    Returns:
        ``(N,)`` temperatures in ``[lo, hi]``.
    """
    x = topk_logits.float()
    lo_t = torch.full_like(target_entropy, lo)
    hi_t = torch.full_like(target_entropy, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        logp = F.log_softmax(x / mid.unsqueeze(-1), dim=-1)
        h = -(logp.exp() * logp).sum(-1)
        too_cold = h < target_entropy      # need more temperature
        lo_t = torch.where(too_cold, mid, lo_t)
        hi_t = torch.where(too_cold, hi_t, mid)
    return 0.5 * (lo_t + hi_t)


class SEDRegularizer:
    """EMA shadow + entropy-gated self-distillation loss.

    Args:
        model: the trainee ``theta``. The shadow is a detached deep copy of it.
        ema_decay: ``mu`` in ``phi <- mu*phi + (1-mu)*theta``.
        sync_every: EMA cadence in **optimizer** steps.
        tau_range: bisection bounds for ``tau_hat``.
        topk: k for the entropy computation and the bisection.
        H_pivot: gate centre, from the compact-register entropy histogram.
        delta_max: maximum entropy boost (0.7 = restoration mode).
        gamma_e: gate sharpness. **Declared placeholder** — it has no
            design-table anchor; 1.0 is the start value pinned by activity 007.
        shadow_dtype: bf16 keeps the 1.7B shadow at 3.4 GB on-device. The EMA
            step itself is done in fp32 and cast back, so a 1%-per-sync update
            is not lost to bf16 rounding.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ema_decay: float = 0.99,
        sync_every: int = 5,
        tau_range: tuple = (1.1, 1.5),
        topk: int = 512,
        H_pivot: float = H_PIVOT_DEFAULT,
        delta_max: float = DELTA_MAX_DEFAULT,
        gamma_e: float = 1.0,
        bisect_iters: int = 20,
        shadow_dtype: Optional[torch.dtype] = torch.bfloat16,
        chunk: int = 1024,
    ) -> None:
        self.ema_decay = float(ema_decay)
        self.sync_every = int(sync_every)
        self.tau_lo, self.tau_hi = float(tau_range[0]), float(tau_range[1])
        self.topk = int(topk)
        self.H_pivot = float(H_pivot)
        self.delta_max = float(delta_max)
        self.gamma_e = float(gamma_e)
        self.bisect_iters = int(bisect_iters)
        self.chunk = int(chunk)
        self.n_syncs = 0

        # Rule 1: initialize phi <- theta, as an EMA copy that is never trained.
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        if shadow_dtype is not None:
            self.shadow.to(dtype=shadow_dtype)
        self._model = model

    # -- EMA ---------------------------------------------------------------
    @torch.no_grad()
    def sync(self) -> None:
        """One EMA *update* — ``phi <- mu*phi + (1-mu)*theta``. Never a copy."""
        mu = self.ema_decay
        theta = dict(self._model.named_parameters())
        for name, p_phi in self.shadow.named_parameters():
            p_theta = theta[name]
            # fp32 arithmetic, then cast back: at mu=0.99 each sync moves phi by
            # 1%, which is only ~2.5x bf16's relative quantum. Doing the mul-add
            # in bf16 would round part of every update away.
            updated = p_phi.float().mul_(mu).add_(p_theta.detach().float(), alpha=1.0 - mu)
            p_phi.copy_(updated.to(p_phi.dtype))
        # Buffers (rotary caches etc.) are not averaged; keep them in step.
        buf = dict(self._model.named_buffers())
        for name, b_phi in self.shadow.named_buffers():
            if name in buf and b_phi.shape == buf[name].shape:
                b_phi.copy_(buf[name].to(b_phi.dtype))
        self.n_syncs += 1

    def maybe_sync(self, optimizer_step_idx: int) -> bool:
        """Sync iff this **optimizer** step is on the cadence (rules 1 and 2).

        Call once per optimizer step, never per micro-batch. Returns whether a
        sync happened, so a trainer can log the cadence and a test can count it.
        """
        if optimizer_step_idx > 0 and optimizer_step_idx % self.sync_every == 0:
            self.sync()
            return True
        return False

    # -- loss --------------------------------------------------------------
    @torch.no_grad()
    def _shadow_targets(self, input_ids, attention_mask, positions, labels):
        """Shadow logprobs at the data tokens, under the per-token ``tau_hat``.

        Returns ``(log_pi_phi_tau, H_t, tau_hat, delta_t)``, each ``(N,)`` over
        the requested ``positions``. Rule 3: everything here is the *shadow's*
        logits — the trainee's are never consulted.
        """
        # logits at position t-1 predict token t; `positions` are the token
        # positions, so the predicting rows are positions-1.
        rows = row_logits(self.shadow, input_ids, positions - 1, attention_mask)

        H_all, tau_all, delta_all, lp_all = [], [], [], []
        for s in range(0, rows.shape[0], self.chunk):
            blk = rows[s : s + self.chunk].float()
            k = min(self.topk, blk.shape[-1])
            topv = torch.topk(blk, k=k, dim=-1).values

            logp1 = F.log_softmax(topv, dim=-1)
            H = -(logp1.exp() * logp1).sum(-1)
            delta = self.delta_max * torch.sigmoid(self.gamma_e * (H - self.H_pivot))
            tau = solve_temperature(
                topv, H + delta,
                lo=self.tau_lo, hi=self.tau_hi, iters=self.bisect_iters,
            )
            # Exact full-vocab logprob of the data token at tau_hat.
            lp = F.log_softmax(blk / tau.unsqueeze(-1), dim=-1)
            y = labels[s : s + self.chunk].unsqueeze(-1)
            lp_all.append(lp.gather(-1, y).squeeze(-1))
            H_all.append(H)
            tau_all.append(tau)
            delta_all.append(delta)

        return (
            torch.cat(lp_all),
            torch.cat(H_all),
            torch.cat(tau_all),
            torch.cat(delta_all),
        )

    def loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        think_mask: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """K2 self-distillation loss on think tokens.

        Args:
            student_logits: ``(1, T, V)`` from the trainee's forward — carries
                grad; this is the only path gradient may flow through.
            input_ids: ``(1, T)``.
            think_mask: ``(T,)`` or ``(1, T)`` of 0/1 over **token** positions
                (from :func:`whetstone.segments.parse_segments`).
        Returns:
            Scalar loss, or ``(loss, stats)`` when ``return_stats``.
        """
        if student_logits.shape[0] != 1:
            raise ValueError("SED kernel is written for per-device batch 1")
        m = think_mask.reshape(-1)
        # Position 0 has nothing predicting it.
        m = m.clone()
        m[0] = 0
        positions = torch.nonzero(m, as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            zero = student_logits.sum() * 0.0
            return (zero, {"n_tokens": 0}) if return_stats else zero

        return self.loss_rows(
            student_logits[0, positions - 1, :],
            input_ids,
            positions,
            attention_mask=attention_mask,
            return_stats=return_stats,
        )

    def loss_rows(
        self,
        student_row_logits: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """K2 over pre-sliced rows — ``student_row_logits[i]`` predicts
        ``input_ids[0, positions[i]]``.

        The entry point for callers that already restricted the trainee's
        ``lm_head`` to the supervised rows (see :func:`row_logits`); ``loss``
        delegates here after deriving positions from the think mask.
        """
        if positions.numel() == 0:
            zero = student_row_logits.sum() * 0.0
            return (zero, {"n_tokens": 0}) if return_stats else zero

        labels = input_ids[0, positions]
        lp_phi, H, tau, delta = self._shadow_targets(
            input_ids, attention_mask, positions, labels
        )

        # Trainee side: exact full-vocab logprob at the same data tokens.
        # Chunked so a long sequence never materializes N x V in fp32 at once.
        rows = student_row_logits
        parts = []
        for s in range(0, rows.shape[0], self.chunk):
            blk = F.log_softmax(rows[s : s + self.chunk].float(), dim=-1)
            parts.append(blk.gather(-1, labels[s : s + self.chunk].unsqueeze(-1)).squeeze(-1))
        lp_theta = torch.cat(parts)

        k2 = 0.5 * (lp_theta - lp_phi) ** 2
        loss = k2.mean()

        if return_stats:
            return loss, {
                "n_tokens": int(positions.numel()),
                "H_shadow_mean": float(H.mean()),
                "delta_mean": float(delta.mean()),
                "tau_mean": float(tau.mean()),
                "tau_at_lo_frac": float((tau <= self.tau_lo + 1e-4).float().mean()),
                "tau_at_hi_frac": float((tau >= self.tau_hi - 1e-4).float().mean()),
                "k2_mean": float(loss.detach()),
            }
        return loss


__all__ = [
    "DELTA_MAX_DEFAULT",
    "H_PIVOT_DEFAULT",
    "SEDRegularizer",
    "row_logits",
    "solve_temperature",
    "topk_entropy",
]
