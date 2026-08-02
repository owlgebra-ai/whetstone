# P4 — Round 0: scorer inoculation + the F1 band-existence gate

STATUS: blocked (P3)
MACHINES: turing (training + eval forwards)
DEPENDS ON: P0–P3 (H_pivot, seed register splits, verbose control set)
BLOCKS: everything — F1 is the go/no-go for all teacher GPU-hours (design §8 Risk 1, §11)
DELIVERABLES: R token-set JSON, SED kernel (`whetstone/sed.py`) with unit tests, inoculation trainer, the four monitoring curves, the three meter unit tests, an **F1 verdict**.

## Objective

Calibrate the scorer so compact-register tokens read as a low "hum" while genuine reasoning leaps still spike (design §2). This is the single highest-risk item in v2: **if the calibration band does not exist, the whole design pivots** (fallback: prefix/LoRA scorer). It is deliberately run on the smallest tier before any Stage-A compute. Everything in this packet is instrumentation-first — the product is a *trustworthy measuring instrument*, not a capable model.

## Read first (mandatory, in order)

1. Design doc §2 (Round 0 spec — the loss, the three-state table, stopping rules, unit tests)
2. Design doc §12.3 (type aggregation), §12.4 (SED kernel — every bullet is a known bug you'd otherwise write)
3. Design doc §7 (Round-0 monitoring curves + overshoot signature)
4. Design doc §8 Risk 1 (what failure means and the fallback)

## Part 1 — R token-set builder (`scripts/build_register_tokenset.py`, new)

Over the seed register corpus **train split**, under the frozen start π_0 (HF forward, teacher-forced, think segments only):

1. Per token **type** (vocab id), collect surprisal `−log π_0(τ_t)` across all its occurrences.
2. `R = { types with mean surprisal > 75th pct across types AND across-occurrence std < median std }` — consistently-expensive, consistently-priced types are *style vocabulary* → train them. High-surprisal but type-inconsistent positions are *content* → mask them out.
3. Union with the **structural whitelist**: every token id in the tokenizations of `⇒ → ; ✓ ⚠ ?` and the card's step-marker strings (use P3's symbol-tokenization table; multi-token symbols contribute all their piece-ids).
4. Require a minimum occurrence count (≥ 10) before a type is eligible — rare types have garbage std estimates.
5. Dump `/data/whetstone/runs/round0/R_tokenset.json`: `{token_id: {surface, mean_surprisal, std, count, source: "stats"|"whitelist"}}`. Print the R-set size and the top-50 surfaces sorted by mean surprisal — **eyeball this list in the activity file**; if ordinary English words dominate, the 75th-pct threshold is wrong for this corpus.

## Part 2 — SED kernel (`whetstone/sed.py`, new — shared by Round 0 AND Stage B; write once, test hard)

API sketch: `SEDRegularizer(model, ema_decay=0.99, sync_every=5, tau_range=(1.1, 1.5), topk=512, H_pivot=<from P3>, delta_max=<0.5|0.7 per audit verdict>)` with `.maybe_sync(optimizer_step_idx)` and `.loss(student_logits, input_ids, think_mask)`.

Implementation rules — each one is a named bug from design §12.4:

1. **EMA update, never replacement:** `φ ← μ·φ + (1−μ)·θ`, μ=0.99, applied every **n=5 optimizer steps** (not micro-batches — with grad-accum 8, that's every 40 forward/backwards). Initialize φ ← θ. A hard copy every 5 steps silently destroys the stabilization (φ becomes a 5-step-lagged clone).
2. **Gate and temperature search on φ's logits:** one forward of φ per batch gives H_t (teacher entropy, top-512) → target `H_t + Δ_t`, `Δ_t = Δ_max · σ(γ_e (H_t − H_pivot))`.
3. **Bisection:** 20 iterations over τ̂ ∈ [1.1, 1.5] on φ's top-512 logits per token to hit the target entropy. Clamp at range ends without warning-spam.
4. **K2 loss at the data token:** `½ (log π_θ(y_t) − log π_φ,τ̂(y_t))²`, masked to think tokens, mean-reduced over unmasked positions.
5. At 1.7B keep φ on-GPU (bf16, ~3.4 GB). CPU-offload is the 8B fallback, not needed here.

**Unit tests (`tests/test_sed.py`, must exist before the trainer runs):**
- EMA horizon: after k syncs with constant θ, ‖φ−θ‖ decays as μ^k (analytic check on a tiny linear model).
- Bisection hits a random target entropy within 1e-2 nats on random logits; τ̂ clamps correctly at both ends.
- Optimizer-step counting: with grad-accum 8, exactly one sync per 40 micro-batches (mock the loop).
- K2 gradient flows to θ only (φ under `no_grad`).

## Part 3 — Inoculation trainer (`scripts/inoculate_scorer.py`, new)

**Loss (design §2):** `L = Σ_{t ∈ R, think} CE_t + 1.0 · L_SED` (+ optional light KL to π_0 on ¬R tokens — implement the flag, default off, turn on only if S2 keeps tripping).

**Config:** full-FT Qwen3-1.7B, bf16, sdpa, grad checkpointing ON, LR 1e-5 (calibration LR — this is not capability training), warmup 20 steps + cosine, ≤ 1 epoch over the seed-register train split, per-device batch 1, grad-accum 32, AdamW. Save a checkpoint at **every eval** (every 20 optimizer steps) — the stopping rule selects one *retroactively* (rollback is expected, not exceptional).

**Memory budget (32 GB):** θ bf16 3.4 + grads 3.4 + AdamW moments fp32 ~13.6 + φ (EMA) 3.4 + π_0 frozen eval copy 3.4 ≈ 27 GB + activations (short compact traces, bs 1, grad ckpt) — fits, but barely. If OOM: move π_0 evals to spark over HTTP (it's the same frozen model the reward server already serves — `prompt_logprobs` gives everything the metrics below need) before reaching for 8-bit optimizers.

**Metrics every 20 steps** (the four §7 curves — write to JSONL + a live-updating PNG):
1. **Held-out register p95 gap ↓** (stop criterion S1): teacher-force the `heldout_register` split through the *trainee*; per think-token `d_t = log p(top1) − log p(actual)`; report p95 over all tokens. S1 fires when p95 < τ_spike.
2. **Verbose-control KL ↑ slowly** (drift budget S2): mean per-token KL(π_θ ‖ π_0) on the verbose control set (top-512 approximation is fine — same top-512 convention as everywhere). S2 fires at κ_max.
3. **Entropy median ≈ flat** (floor S3): median per-token entropy on the control set; S3 fires if it drops > x% below the P2 audit baseline.
4. **Register-token mean surprisal ↓ to a hum plateau** (the intended effect happening).

**Stopping:** first of S1 / S2 / S3. τ_spike ≈ 4 nats, κ_max and x are *declared placeholders* (design §12.6) — this run's job is to measure and pin them; start with κ_max such that S2 wouldn't fire in the first 40 steps, and x = 10%. **Overshoot signature** (design §7): verbose-control likelihood falling while register p95 keeps dropping past τ_spike → roll back to the checkpoint at the S1 crossing.

## Part 4 — Meter unit tests (design §2 — all three, simultaneously, at one checkpoint)

Run against the selected checkpoint AND its two neighbors (the band may be narrow):

- **(a) Register hum:** heldout_register p95 gap < τ_spike. ✓ = the accent is installed.
- **(b) Verbose intact:** mean logprob of verbose control traces within ε of π_0's (report the delta; ε to be pinned — start at 0.2 nats/token). ✓ = no infection.
- **(c) Corrupted-trace probe — the decisive test:** take the `probe_pool` split (never trained on). For each trace create a corrupted twin by ONE of (implement both corruptions, ~25 traces each):
  1. **Chunk deletion:** remove one intermediate derivation chunk from the compact think, so the next step uses a result that now comes from nowhere.
  2. **Value substitution:** replace one intermediate numeric result with a plausible wrong value (e.g. ±1 on a digit), leaving the following steps (which used the *correct* value) as non-sequiturs.
  Score both twins with the checkpoint. **Pass:** p95 gap over the corrupted span (corruption point → +30 tokens) > τ_leap, while the uncorrupted twin's same span stays < τ_spike. Report the full (corrupted, clean) gap-histogram pair — this plot IS the band-existence evidence.

**F1 verdict:** some checkpoint passes (a)+(b)+(c) **simultaneously** → F1 PASS: record checkpoint path, measured (τ_spike, τ_leap, κ_max, x), and freeze that checkpoint as **the scorer** at `/data/whetstone/ckpt/scorer_v1/`. Load it into the spark reward server and re-verify the P0 scoring check against it.

No checkpoint passes all three → **F1 FAIL**: do not tweak endlessly — one (λ-orthogonal) retry is allowed if the failure is marginal and clearly attributable (e.g. R-set polluted by content tokens → rebuild R with a stricter std filter). Otherwise stop and write up the failure precisely (which test failed, by how much, at every checkpoint): the design's prescribed pivot is the **prefix/LoRA scorer arm** (design §8 Risk 1 — adapters can't reshape generation but likelihood calibration is a smaller ask), which becomes a new packet.

## Gotchas

1. **Do not touch the probe_pool split for anything before Part 4.** Leakage here fakes a PASS and the failure surfaces GPU-days later in Stage A as a verbose teacher.
2. **CE only on R-tokens, in think segments only** — masking mistakes that include answer tokens teach the scorer the register in the answer region, which Stage C explicitly protects from register change.
3. **The trainee is the SCORER, not the student.** Round 0's EMA copy belongs to Round 0 (design §12.4: never share or carry EMA copies across stages). Stage B's student starts from the *original* checkpoint, not from this one.
4. **d_t uses top1 − actual, not entropy.** Positions where actual = top1 give d_t = 0 by construction — assert this in code (it doubles as the P0 scoring-convention check).
5. Eval forwards (metrics 1–3) are `no_grad` + bf16; don't let them build graphs — the 27 GB budget has no slack for autograd during eval.
6. Log *wall-clock per 20-step cycle* in the activity file. Design expects minutes-to-an-hour scale for the whole run; if you're projecting days, something (probably eval breadth) is mis-sized — shrink the control sets, not the metrics.

## Definition of done

- [ ] R-set built, size + top-50 eyeball check logged.
- [ ] SED kernel + all four unit tests green, committed.
- [ ] Training run(s) complete with the four curves (PNGs in `activity/assets/NNN/`).
- [ ] All three meter tests reported with plots; **F1 verdict stated in bold** in activity file `NNN-round0-inoculation.md`, with measured values for every placeholder (τ_spike, τ_leap, κ_max, x, ε).
- [ ] On PASS: scorer frozen to `/data/whetstone/ckpt/scorer_v1/`, loaded on spark, scoring re-verified.
- [ ] CLAUDE.md + design-doc placeholder table annotated with the pinned values (small PR).
- [ ] Packet status flipped; ROADMAP updated (P5 unblocked on PASS, or the LoRA-arm packet drafted on FAIL).
