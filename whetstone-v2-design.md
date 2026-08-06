# WHETSTONE v2 ("HONE") — Consolidated Design Doc

**Pedagogically-compressed, entropy-preserved self-training for compact reasoning, starting from a structured thinking model.**

Replaces the v1 mechanism (prompt-compress → filter → surprisal-up-weighted SFT → KL-anchored DAPO) with: a *trained* compression teacher operating with the register in its context (Pedagogical RL), *learnability-gated + entropy-preserving* assimilation (Pedagogical RL × CurioSFT), and *collapse-targeted, segment-routed* RL with a dead-band rescue loop (TEA-RL × Pedagogical RL × SCA). Infra from v1 survives unchanged: deterministic verifier, calibration probes, resume invariants, curriculum-from-init, Pareto endpoint selection.

**Starting checkpoint: an existing thinking model** (long native `<think>` traces, stable boundaries). v1 §1.4 prohibited this because blind harvest depended on a pre-collapse high-entropy distribution; v2 dissolves that dependency — harvest is demoted to a small seed, and the training corpus comes from an RL-trained privileged teacher that does not rely on lucky high-entropy sampling.

---

## 0. Diagnosis of v1 (why the mechanism was unsound)

| # | v1 mechanism | Failure | Fixed in |
|---|---|---|---|
| 1 | Surprisal **up**-weighting w=1+0.5·S in SFT | Compressed traces contain unsupported jumps (elided steps → tokens with π_S≈e⁻⁸). Up-weighting concentrates gradient on the least-learnable tokens — a high-pass where a band-pass is needed | Stage B |
| 2 | Compressor prompted + sieved (Δlogp, audit), never trained | No optimization pressure toward better compressions; Δlogp measures answer-recoverability (reading), not generative learnability (writing) | Stage A |
| 3 | 2× pure-CE SFT + a register that deletes fork tokens | Entropy collapse before RL → DAPO step-50 rumination collapse (97% cap-hit); "compact" became "rigid" | Stage B (SED), Stage C (TEA) |
| 4 | Compactness-rigidity (v1 §12): group signal ∝ p² at low pass rate | RL stalls exactly where compression matters most | Stage C rescue + difficulty amplification |
| 5 | Char-count length penalty; uniform whole-sequence KL anchor | Penalty gamed (counter-restart); KL taxes exploration uniformly and anchors errors along with style | Stage A (budget in teacher reward), Stage C (routed anchoring) |

Underlying pattern: every v1 intervention is **uniform or monotone**. All source methods make the same move in different places: replace the uniform intervention with a **token-selective, signal-targeted** one — by learnability (Pedagogical RL), by uncertainty (CurioSFT), by collapse-attribution (TEA), by structural segment (SCA).

Unifying observation: privileged information is *native* to Whetstone — the gold answer and the verbose verified trace are exactly Pedagogical RL's context c. Compression is a pedagogy problem: produce the shortest trajectory that uses only doors the student can open.

---

## 1. Setting, roles, and preconditions

**Roles.** Two copies of the thinking checkpoint: the **teacher** π_T(τ | q, c), conditioned on privileged context c (gold answer; verbose trace when available; the register card — see below), trainable in Stage A; the **student/scorer** π_S(τ | q), no privileged context, frozen during Stage A and used as the learnability meter, trained in Stage B, RL'd in Stage C. The final deliverable is the student, prompt-free.

**Preconditions.**
1. **Entropy audit** of the starting checkpoint: per-token entropy histogram on ~200 pool problems, before anything else. Sets H_pivot for Stage B and decides whether SED runs in preservation or *restoration* mode (post-RL thinking models often arrive entropy-collapsed; raise Δ_max if so). The 80/20 fork structure is native to this model class, so expect a usable bimodal histogram.
2. **Register card:** a one-page human-written spec of the target notation plus 5–10 exemplars. The single point where design intent enters the pipeline. The register is *specified, not discovered* — no downstream component is asked to invent it.
3. **Seed harvest:** small blind harvest (K=2, 10–20% of pool, standard budget), verifier-filtered. Native think blocks; run the v1 Step-2.1 calibration probe unchanged.
4. **Seed register corpus:** the card applied once via the v1 chunkwise prompted-compression machinery to the seed harvest (~300–1,000 traces, temperature 0.3–0.5 for mild register-internal variance, Δlogp gate retained — its only remaining use).
5. Verifier, pool-with-golds, persistent storage, cross-family audit API: as v1.

---

## 2. Round 0 — Scorer Register Inoculation

The scorer is a verbose-CoT native: compact-register tokens read as improbable for *style* reasons, and untreated this drives the Stage-A reward toward verbosity. Round 0 calibrates the meter — the smallest dose that creates recognition without infection.

**The objective is a threshold, not a minimum.** The meter has three states:

| State | Meter behavior | Downstream cascade |
|---|---|---|
| Undertrained | register tokens spike | pedagogy reward drives teacher verbose; compression dies |
| **Band (target)** | register = hum (elevated mean, no spikes); verbose ≈ baseline; genuine leaps still spike | reward separates style from leaps as intended |
| Overtrained | register becomes argmax; residual prose reads spiky | teacher driven to caveman degeneration (v1 audit failure class) |

**Loss.**

```
L_inoc = Σ_{t ∈ R} CE_t  +  α_sed · L_SED   ( + optional KL(π‖π_0) on ¬R )
```

- **R (register-token set) by type aggregation:** over the seed corpus under the frozen start π_0, compute per-token-type mean and variance of surprisal. Types with consistently elevated surprisal (high mean, low across-occurrence variance) are style vocabulary → train. Positions with idiosyncratic spikes (high surprisal, type-inconsistent) are content → mask. Light-IF's selection logic with inverted purpose: select what to install, not what to protect.
- **L_SED:** CurioSFT self-distillation exactly as in Stage B (EMA teacher, entropy-gated τ̂, K2) — the entropy-stability mechanism during this SFT.
- LR 1e-5 (calibration, not capability), ≤1 epoch, warmup + cosine, eval every ~20 steps.

**Stopping — first of:**
1. **S1 calibration reached:** held-out register p95 surprise gap < τ_spike.
2. **S2 drift budget exceeded:** mean KL(π_θ‖π_0) on a verbose control set > κ_max (minimal-KL update discipline).
3. **S3 entropy floor breached:** median per-token entropy drop > x% vs π_0.

**Meter unit tests (mandatory before trusting the scorer):** (a) held-out register traces → hum; (b) verbose traces → near-baseline likelihood; (c) **corrupted-trace probe:** traces with one deliberately inserted unsupported leap must still spike (p95 gap over the corrupted span > τ_leap). Failing (c) invalidates the instrument regardless of (a)/(b) — roll back.

**Cost:** a few hundred steps, minutes to ~an hour. The risk it retires — a silently inverted reward meter steering Stage A the wrong way for GPU-days — is the largest in the design.

---

## 3. Stage A — Compression-Teacher RL

### 3.1 Register lives in the teacher's context, not its weights

Every teacher rollout is conditioned on the register card + exemplars (+ gold, + verbose trace when available). In-context conditioning puts terse output inside the sampling support from step 0; GRPO's job is execution and refinement, not discovery (a group-relative reward can only rank sampled lengths — it trims within the current mode and cannot jump to a register it never samples). The teacher receives **no register SFT**: no priming-induced entropy collapse before its RL, no dependence on spontaneous terseness. The student's `<think>`-format prompting is unprivileged and identical to eval.

### 3.2 Reward (product form — non-negotiable)

```
r_ped(q, c, τ) = R_acc(τ, a*) · G_spike(τ | q; π_S) · G_budget(τ_think)
```

- **R_acc** — deterministic verifier, v4.6.1 rules (post-`</think>` extraction only), three-tier structure retained.
- **G_spike** — spike-aware learnability under the inoculated scorer:

  ```
  d_t     = log π_S(a_max | q, τ_<t) − log π_S(τ_t | q, τ_<t)
  G_spike = exp[ −(λ/β) · log( (1/T) Σ_t exp(β·d_t) ) ]
  ```

  **(λ, β) split for the thinking-model scorer:** residual register novelty appears as uniformly elevated gaps and costs λ·mean regardless of β; a reasoning leap is one catastrophic gap dominating at high β. Set λ modest (tolerate the accent), β high (never tolerate the unfollowable step); sweep β∈{5, 10}. Re-tighten λ as inoculation rounds shrink the baseline register tax. Subsumes the Δlogp gate and most of the audit's faithfulness function: a hallucinated or unsupported compact step is, by construction, a spike.
- **G_budget** — on think tokens only: `exp[−μ·max(0, T_think − B)/B]`, μ=1, soft tail (v1 cliff lesson). **Annealed with a freeze rule:** B starts at the current median prompted-compressed length, tightens toward B_target (~600); tightening pauses whenever within-group think-length std < s_min. A reward must never demand lengths outside the realized group spread — that yields zero signal and pure drift.

### 3.3 Training and monitoring

GRPO, group size 8, rollout T≈0.9, **TEA regularization on the teacher's own GRPO** (its exploration matters too). Dashboards: symbol density (fraction of tokens in the register vocabulary), think-length bimodality index (healthy: mass migrating verbose→terse; sick: one mode narrowing in place), and **mean gap vs max gap as separate curves** — healthy is mean drifting down as scorer and teacher co-adapt, max pinned near zero.

Cross-family audit (Claude Sonnet), demoted to a **spot-check**: 100-sample stratified per teacher checkpoint, prompted specifically for reward-hacking signatures (fluent filler, restated-problem no-ops, caveman degeneration); gate = ≥90% pass.

**Output.** Trained teacher π̃_T + a corpus sampled from it on the full pool (K=4, T=0.8, verifier-filtered).

---

## 4. Stage B — Learnability-Gated, Entropy-Preserving Assimilation SFT

Student trains on teacher trajectories (no register card in the student's prompt — the register enters weights here, and only here). Two modifications to plain CE:

### 4.1 ZPD band-pass token weight (fixes Diagnosis #1)

```
w_t = σ( κ · (log π_S(τ_t | q, τ_<t) − γ) ) · ( 1 + α_nov · min(S_t, s_cap) )
```

Gate ≈1 where the student already assigns reasonable probability, ≈0 on residual spikes; the capped novelty factor (s_cap≈4 nats, α_nov=0.5) up-weights surprisal only inside the reachable zone. Start κ=1; **set γ from the measured student log-prob histogram on teacher outputs** (a verbose-native model shifts it). Normalize per-sequence by Σw; assistant-only masking. Net shape vs v1: boilerplate ≈ baseline, learnable novelty boosted, spikes suppressed instead of amplified.

### 4.2 SED term (fixes Diagnosis #3; restoration mode)

```
L = L_assim + α_sed · L_SED ,  α_sed = 1
```

CurioSFT exactly: K2 to the EMA copy (sync 5 steps, decay 0.99) at per-token τ̂_t ∈ [1.1, 1.5] hitting H_t + Δ_t, Δ_t = Δ_max·σ(γ_e(H_t − H_pivot)), top-512 entropy. **H_pivot from the precondition entropy audit** (~80th percentile of the compact-register histogram); if the audit showed the start already collapsed, SED runs in restoration mode — raise Δ_max. Rationale: rigidity *is* entropy collapse expressed in the compact register; the ⚠ token can be one character on the page while the distribution behind it stays open, giving Stage C live forks at 600 tokens.

### 4.3 Iteration and gate

Two rounds: train → **recompute all gates under the updated student** (one forward pass) → fresh teacher batch → train. Optional short teacher re-RL against the updated student in run 2. Sanity gate: eval accuracy as v1, plus median per-token entropy **not below the audit baseline** at comparable accuracy (above it, if restoration was needed) — otherwise the SED calibration is wrong; fix before RL.

---

## 5. Stage C — Segment-Routed RL with TEA and Dead-Band Rescue

### 5.1 Objective and routing

DAPO backbone (clip 0.2/0.28, group 8) with SCA-style segment parsing (`<think>` boundaries → think/answer masks, advantages normalized and routed per segment) and a three-way anchor routing that **replaces the uniform whole-sequence KL entirely**:

- **Think tokens:** compression pressure (soft length tail, v1 form) + **TEA** — per batch, Cov_t = centered(log p_t)·centered(A_t); weight_t = min(softmax(Cov_t/τ_c), c/|T_r|); L_TEA = |T_r|·Σ weight_t·H_t, subtracted with λ_TEA. Start (τ_c, λ_TEA, c) = (1.0, 0.05, 100). Targets entropy protection at exactly the tokens the current update is sharpening — the v1 step-50 collapse mechanism. No style anchor on think tokens: changing that register is the point.
- **Answer tokens:** forward KL to the **original thinking checkpoint** (the best answer-writer in the pipeline) + SCA length band with tolerance f (anchor f=32). Answer drift is live from day one at this starting point — thinking models carry multi-thousand-token answer segments.

### 5.2 Curriculum

Per phase: fresh K=8 from the current init (v1 rule); bucket 0/8, 1–7/8, 8/8.

- **1–7/8 → DAPO**, with SCA's bounded difficulty amplification W(x) = 1 + α(1 − p̂_succ(x)), α=0.5, applied to *positive think advantages only* — the cheap tier against signal dilution at 1–2/8.
- **0/8 → pedagogy rescue:** teacher (gold in hand) generates M=4 candidates per dead problem; verifier + G_spike threshold filter; assimilate survivors with the Stage-B loss (small LR, brief); problems re-enter the next K=8 refresh. The expensive tier, for where group signal is dead (∝ p²). With a capable starting checkpoint the dead band is small early and this is a late-phase instrument.
- Endpoint per phase from the accuracy × tokens-per-correct Pareto frontier; expect over-training past ~80% of steps to remain the failure mode — TEA pushing that boundary later is itself a measurable claim.

Per-checkpoint rollout investigation, stop rules, merge/eval cadence: v1 §7.6–7.7 unchanged.

---

## 6. Metrics, reporting, and baselines

- **Segment-level reporting, always:** think length and answer length as separate numbers; answer quality judged from the answer segment only. One combined length number is how drift hides.
- **Headline decomposition:** total token reduction = (trim within the verbose register — SCA-comparable, expect ~2×) + (register change — the v2-specific contribution). Report the factors separately. The v1 "~21× vs verbose base" framing is retired; it does not survive the change of starting point.
- **Entropy trajectory** across the pipeline (audit → post-B → per-RL-checkpoint) and **spike rate** (mean and p95/max gap as separate curves) as first-class dashboards.
- **Mandatory baselines from the same starting checkpoint:** SCA and DeepCompress (does teacher-mediated register installation beat segment-routed length-RL alone?), plus the prompted-compressor-only arm. Main-result models are chosen to match SCA's published setup (§11), so their reported numbers double as external references.
- Accuracy suites, stratified by base 8-sample pass rate as v1 §12 — the success criterion is specifically improving the low-pass-rate rows.

---

## 7. Training-dynamics playbook

**Round 0:** log four curves — held-out register p95 gap ↓ (stop criterion); verbose-control KL ↑ slowly (drift gauge); entropy median ≈ flat (SED health); register-token mean surprisal ↓ to a hum plateau. Overshoot signature: verbose-control likelihood falling while register p95 continues past τ_spike — roll back to the S1 crossing.

**Stage A:** mean gap ↓ with max gap pinned ≈ 0; bimodality index rising then resolving into the terse mode; within-group length std tracked against the B schedule (std → 0 while B tightens ⇒ freeze fires).

**Stage C:** entropy per checkpoint (TEA health), think/answer lengths per checkpoint (routing health), rollout-investigation flags (v1 rot patterns).

**Failure-cascade map:** undertrained scorer → verbose teacher → Stage B no-op (adherence dashboard). Overtrained scorer → caveman teacher (audit spot-check). Teacher spread collapse → freeze rule fires (check TEA weight). Scorer drift across rounds → re-inoculate between teacher rounds rather than lowering λ (too-low λ stops penalizing genuine mean-level sloppiness).

---

## 8. Risks

1. **The calibration band may not exist** — register-hum and leap-spike could be inseparable (calibrating away the style tax dulls the leap detector). **De-risk first, before any teacher GPU-hours:** run round 0 alone on the smallest tier and check whether a stopping point passes all three meter unit tests simultaneously. Fallback: the prefix/LoRA scorer arm, which never moves base weights (SCA shows adapters can't reshape *generation*, but scoring needs only *likelihood* calibration — a smaller ask).
2. **G_spike reward hacking** (degenerate fluent filler, no-op restatements). Mitigations: product with R_acc and G_budget; audit spot-check prompted for hacking signatures; escalate β toward the max-gap regime.
3. **No capability injection — by design and therefore by limit.** All components are self-referential; frontier-hard regimes improve only via better sampling (rescue loop), not new knowledge. Expect the v1 §12 shape to soften, not vanish.
4. **Hyperparameter surface.** New knobs all carry a published anchor or a measured-histogram procedure; sweep only β, H_pivot, λ_TEA in run 1.
5. **Compute.** Added: teacher GRPO (thousands of short rollouts), SED forward (~25% SFT slowdown), round 0 (negligible). Removed: full blind harvest and bulk audit. Net expectation: cheaper end-to-end; verify on the seed run.

---

## 9. Ablation grid

| ID | Question | Arms |
|---|---|---|
| A1 | Weighting direction | v1 up-weight vs gate-only vs ZPD band-pass |
| A2 | Entropy preservation in SFT | ±L_SED; hard top-k mask vs sigmoid gate |
| A3 | RL anchoring | routed (answer-KL + TEA) vs uniform whole-seq KL (v1) vs TEA-only vs plain entropy bonus |
| A4 | Teacher | prompted-only compressor vs RL-trained register-in-context teacher; external: SCA, DeepCompress from same checkpoint |
| A5 | Reward form | product vs additive (Teacher-RL ablation, compression setting) |
| A6 | Hard-prompt signal | ±dead-band rescue; ±difficulty amplification; both |
| A7 | Inoculation mechanism | masked-CE+SED full-FT vs prefix/LoRA vs plain-CE control; (λ,β) grid for style/leap separation |

A1 and A4 test the "v1 was unsound" claims directly; the round-0 band-existence check (Risk 1) and then A1/A4 run first, on the Qwen3-1.7B feasibility tier (§11), gated by F1–F4 before any 4B/8B compute.

---

## 10. v1 → v2 map

| v1 | v2 | Status |
|---|---|---|
| Stage 1 blind harvest (GPU-days) | Seed harvest (K=2, 10–20% pool, native think blocks) | shrunk |
| Stage 2 chunkwise prompted compression | One prompted pass → seed register corpus only (register defined by the card) | repurposed |
| Stage 2.5 Δlogp gate + compression audit | G_spike in the teacher reward; Δlogp on seeds only; audit → spot-check | absorbed |
| — | Round 0 scorer inoculation (threshold-stopped, unit-tested) | new |
| Stage 3 surprisal-up-weighted SFT | Stage B: ZPD band-pass + SED | replaced |
| Stage 4 regen + audit + second SFT | Stage B round 2 (recomputed gates) | simplified |
| Stage 5 DAPO + uniform KL + length penalty | Stage C: segment-routed anchors, TEA, difficulty amplification, dead-band rescue | extended |
| Verifier, probes, resume infra, curriculum-from-init, Pareto endpoint | unchanged | kept |

---

## 11. Model and Run Plan

**Main results:** Qwen3-4B-Thinking-2507 and Qwen3-8B (thinking mode). This deliberately matches SCA's exact evaluation models: their published per-benchmark accuracy, think-length, and answer-length numbers become direct external reference points alongside our own reproduced SCA/DeepCompress baselines, and any reviewer can line the tables up one-to-one.

**Feasibility tier (run first):** the *complete* pipeline end-to-end on **Qwen3-1.7B** (hybrid thinking; `enable_thinking=True` throughout). Purpose is to validate mechanism, not headline numbers — 1.7B's weaker priors mean smaller frontier-tier gains are expected (the CurioSFT Llama precedent). Go/no-go gates before any 4B/8B compute:

- **F1 — the band exists:** Round 0 reaches a stopping point where all three meter unit tests pass simultaneously (register hum, verbose intact, corrupted-trace spike).
- **F2 — teacher converges to the register:** symbol density plateaus, the think-length bimodality resolves into the terse mode, teacher R_acc stays within 3 pts of the prompted-compression baseline, audit spot-check ≥ 90%.
- **F3 — assimilation holds:** Stage B student within 1 pt of the starting checkpoint's accuracy at ≤ 50% of its median think tokens, and median entropy not below the audit baseline.
- **F4 — RL is stable:** 50 DAPO steps with no critical rollout-investigation flag and at least one checkpoint Pareto-dominating the start on the easy suite.

Failing F1 → the prefix/LoRA scorer arm before anything else. Failing F2 with F1 passed → (λ, β) grid and budget schedule before touching Stage B.

**Model-specific notes.** Qwen3-1.7B and Qwen3-8B are hybrid-mode: set `enable_thinking=True` in the chat template for every rollout, scoring pass, and eval. Qwen3-4B-Thinking-2507 is a dedicated thinking model (no flag; always emits `<think>`), with long native context — no template switch exists, so verify the segment parser against its exact template before anything else. The entropy audit matters most on the 2507-Thinking variant (already heavily RL-trained; most likely to arrive entropy-collapsed and need SED in restoration mode).

---

## 12. Implementation Specifics

### 12.1 Templates and segment parsing
Segment masks from the `<think>`/`</think>` boundaries; answer segment = post-`</think>` to EOS/`<|im_end|>`. Malformed outputs (missing/duplicated boundaries, empty answer) → quality gate g = 0, excluded from all structural rewards and alignment losses (SCA's gate rule). Reuse the v1 parser; add the two binary masks to every rollout record.

### 12.2 Scoring passes (G_spike and ZPD gates)
All scorer quantities come from **one teacher-forced vLLM prefill pass** per batch: score the concatenated (q, τ) with `prompt_logprobs ≥ 2`, which returns per-position the actual-token logprob and the top-1 logprob — d_t = top1 − actual, done. No sampling, no second model class.
- The scorer runs as a **dedicated frozen vLLM instance** ("reward server"): tp=1 on one GPU at 1.7B; tp=2 at 8B.
- **Stage B ZPD gates are precomputed offline** per corpus refresh (the scorer is frozen within a round): one scoring pass over the teacher corpus writes per-token w_t into the training JSONL. SED cannot be precomputed (the EMA teacher moves) and runs online.
- Recompute gates after every assimilation round (one pass; cheap) — stale gates are the known drift failure.

### 12.3 Offline analysis scripts (before any training)
- **Entropy audit:** HF forward pass on ~200 stratified pool problems, top-512 entropy per token; output the histogram, H_pivot (≈80th percentile of the compact-register histogram), and the preservation-vs-restoration verdict.
- **Type aggregation for R:** tokenize the seed register corpus; per token-id collect surprisal under π_0 across occurrences; R = { types: mean surprisal > 75th percentile AND across-occurrence std < median } ∪ structural whitelist (⇒, →, ;, ✓, ⚠, ?, numbered-step markers). Dump R as a token-id JSON consumed by the inoculation loss mask.

### 12.4 SED kernel (and the EMA gotchas)
- Teacher maintenance is an **EMA update, not a replacement**: φ ← μ·φ + (1−μ)·θ with μ = 0.99, applied every n = 5 **optimizer** steps; initialize φ ← θ. A hard copy every 5 steps is a bug — the teacher becomes a 5-step-lagged clone of the student (fast-moving target; the stabilization vanishes). At μ = 0.99 and 5-step cadence the effective averaging horizon is ~1/(1−μ) ≈ 100 syncs ≈ 500 optimizer steps: the teacher is a slow shadow of the student, which is the point.
- **Count optimizer steps, not micro-batches.** With grad-accum 8, "every 5 steps" means every 40 forward/backward passes. Syncing per micro-batch moves the teacher 8× too fast.
- **The gate and the temperature search run on the teacher's logits:** H_t (for Δ_t and the bisection target) is computed from π_φ, not the trainee — one forward pass of φ per batch supplies both the entropy gate and the distillation target.
- Bisection: 20 iterations over τ̂ ∈ [1.1, 1.5] on top-512 logits per token to hit H_t + Δ_t; K2 = ½·(log π_θ(y_t) − log π_φ(y_t))² at the data token (θ = the model being trained in that stage).
- Round 0 and Stage B each maintain **their own** EMA copy of their own trainee (scorer in Round 0, student in Stage B). Never share one across stages or carry it over.
- Memory: at 8B hold φ shards on CPU between syncs and gather at sync; at 1.7B keep on-device. Compute-bound fallback (CurioSFT's ablation): drop the separate EMA copy and distill against the live self — costs ~1 pt, saves one model copy.

### 12.5 Topology and memory (8×80GB reference; v1 scaling rules apply)
- **1.7B:** trainer FSDP2 on 4 GPUs (single-GPU works with grad checkpointing), rollout vLLM 2–3 GPUs tp=1, scorer vLLM 1 GPU. Weights 3.4 GB BF16; student + EMA + scorer all fit comfortably — Round 0 and Stage B iterate on a 2-GPU workstation.
- **4B/8B:** 6 trainer GPUs + 2 inference GPUs, with rollout engine (gpu_mem ≈ 0.4, v1-proven) and scorer instance (gpu_mem ≈ 0.3) sharing the inference pair; time-multiplex if allocation races appear (v1 §11 vLLM notes apply verbatim).
- SFT: per-device batch 1, grad-accum to effective 64, grad checkpointing on, LR/masking as stage specs. DAPO: v1 §7.3 config with this doc's Stage-C deltas.

### 12.6 Consolidated hyperparameter table

| Knob | Start | Anchor |
|---|---|---|
| G_spike λ / β | 1 / {5, 10} | Pedagogical RL; (λ,β) split §3.2 |
| G_budget μ / B_0 / B_target / s_min | 1 / **n/a** / **600 — PINNED by activity 008** / **n/a** | v1 552-token sweet spot; freeze rule §3.2. **B_0 and s_min are moot under a frozen teacher** (activity 006): with generate-and-select there is no annealing schedule for the freeze rule to outrun, so B is fixed at B_target. Selected-corpus think length came out median 182, p95 ~710 — the whole distribution sits under B, so G_budget is inactive for most drafts and acts only on the tail |
| Teacher sampling: **K / T / top-p** (was GRPO group / T) | **8 / 0.8 / 0.95 — PINNED by activity 008** | v1 / Pedagogical RL. **Stage A is generate-and-select, not GRPO** (activity 006): a 32B cannot be trained on one 32 GB card. Two further Stage-A parameters are pinned by activity 008 and are *not* optional: the think block must be **prefilled** (`<think>\ngoal:`) or the register does not land at all (0.11 vs 2.10 markers/100 char), and generation must be **two-phase** with the `</think>` boundary imposed rather than sampled (48% of prefilled drafts otherwise fail on that transition alone). **K is not a lever for structural coverage** — retention is clustered by problem, so measured P(≥1 branch-keeping draft) is 35.5% at K=8 where independence predicts 75.3% |
| ZPD κ / γ / α_nov / s_cap | 1 / from histogram (init log 1e-4) / 0.5 / 4 nats | Pedagogical RL assimilation; §4.1. **The histogram now exists** — activity 008 stores a per-draft think-surprisal histogram on every Stage-A score record (`think_surprisal_hist` + `surprisal_bin_edges`), so the masked fraction is computable for any γ without re-scoring: 72.5% of teacher think tokens sit below 0.5 nats, γ=1 masks 20.8%, γ=4 masks 6.4%. **⚠ Measured under `scorer_v1`, which has had 91% of the register style tax removed — Stage B's gate runs under the *original* checkpoint, so these are a LOWER BOUND on the masked fraction and P6 must re-measure before pinning γ** |
| SED α_sed / Δ_max / H_pivot / EMA / τ̂ range / top-k | 1 / **0.7 — restoration mode, PINNED by activity 003** / **0.6707 — PINNED by activity 005** / (5, 0.99) / [1.1, 1.5] / 512 | CurioSFT; kernel + 6 unit tests shipped in `whetstone/sed.py` by activity 007, shared verbatim with Stage B (**new EMA copy per stage**) |
| TEA (τ_c, λ_TEA, c) | **τ_c = 3.0 / λ_TEA = 0 for the next arm / c = 100 — PINNED by activity 010** | Light-IF |
| | ⚠ **Three corrections from the F4 run, all measured.** (i) **τ_c has a narrow usable window and 1.0 is below it.** Effective think tokens under protection, measured offline on 894,860 pilot tokens: **44.6 at τ_c 0.7, 91.1 at 1.0, 1,251 at 3.0, 587,943 at 10.0** (uniform, i.e. the term is inert). The `c = 100` cap only binds — and therefore only means anything — near **τ_c ≈ 3.0**. Activity 003's note is right that 1.0 is wrong, but wrong about the direction: τ_c is a softmax temperature over *covariance*, not an entropy threshold, so it must go **up**, not down. (ii) **`Cov` must be centered over the whole BATCH.** Advantages are constant within a rollout, so a micro-batch of one rollout gives `Cov ≡ 0`, a uniform softmax, and a term that adds mean entropy while selecting nothing — measured as `L_TEA` equalling mean think entropy to eight decimal places. (iii) **`L_TEA = |T_r|·Σ w·H` is rescaled to a weighted MEAN** (attested deviation): the literal form scales with think-token count against a per-token-mean policy loss, giving λ_TEA·L_TEA ≈ 155 at \|T\| ≈ 5,000 versus a policy loss of ~0.1. (iv) **λ_TEA = 0 is the recommended next arm** — not because TEA is harmful, but because F4 failed on *excess* entropy (finding 23) and TEA is one of two terms raising it. | |
| DAPO clip / LR / group | **0.2, 0.2 (SYMMETRIC) / 1e-6 / 8 — REVISED by activity 010** | v1 §7.3 |
| | ⚠ **Clip-higher is the prime suspect for F4's failure on this checkpoint.** ε_high 0.28 > ε_low 0.20 is the design's own "entropy-preserving half of the algorithm", and Stage C has **no ceiling on think entropy anywhere** — the answer-KL bounds only the answer, the length tail only length. That is correct for v1's *collapsed* checkpoint and wrong for one whose entropy Stage B already restored to **10.1× the audit baseline in median** (F3c). 60 pilot steps took think entropy **1.05 → 3.18 nats** and the first casualty was the lowest-entropy decision in the sequence — emitting `</think>` (malformation 5.6% → 35.6% at the training sampler). **Symmetric clipping is the cheapest test**: it removes the dominant driver and changes nothing else. | |
| Answer band f / λ_align | **32 / 0.1 — PINNED by activity 010** (target = the baseline card's answer median **288**, not the corpus's 189: the band and the π_0 anchor must agree on what an answer is). Held: `corr(step, answer_median) = −0.008` over 60 steps, window means 379 → 354, answer-KL stable 0.175 → 0.213, against 009 round 2's uncontrolled 288 → 19 collapse. **The one Stage-C component that demonstrably did its job.** | SCA (f) |
| Difficulty amplification α | 0.5 | SCA |
| Rescue M / rescue LR | 4 / 5e-6, ≤1 epoch* | §5.2; *still a placeholder — rescue was built (`scripts/stagec_rescue.py`) but not run, since F4 blocked the phase boundary it belongs to. Two filter deviations are pinned by activity 010 regardless: **no G_spike threshold** (008 f10b — AUC 0.541 at level 9) and **gold+trace conditioning wherever a trace exists** (008 f13 — gold-only confabulates 73.7% in the hard band) |
| Inoculation LR / τ_spike / τ_leap / κ_max / entropy floor x / γ_e | 1e-5 / **2.25** / **3.175** / **0.3174** / 10% (never fires — see below) / **1.0** | §2; **all PINNED by activity 007** (F1 run, Qwen3-1.7B). The 4-nat τ_spike placeholder and the packet's 1.2 replacement are both retired: this corpus's step-0 register p95 is 6.375, not the 2.375 activity 004 measured. τ_spike = clean-span median under scorer_v1; τ_leap = Youden-optimal on 110 paired corrupted/clean probes (AUC 0.810). **x cannot fire in restoration mode** — entropy *rose* 29.8%; the guard against inflation is meter test (c), not S3. |

Asterisked values are declared placeholders: the 1.7B feasibility run exists to pin them before they are treated as defaults.

**Stage-C sampling and batching, pinned by activity 010:** RL rollouts and every
curriculum re-bucket run at **T = 1.0 / top-p 1.0** (policy-gradient correctness;
measured to buy 63% → 84% mixed groups on the 200-screen at a cost of 16 Pass@1
points *at the training temperature only*). Rollout cap **12,288** — the packet's
8,192 was a GSM8K-derived number that truncates **5.58%** of well-formed
generations on a pool that is half DeepMath. Batch **8 problems/step with
`--prefetch`**: at 4 problems/step the per-step length medians carry a CV of
**0.82** and accuracy tracks batch difficulty at `corr = +0.770`, so training-log
curves cannot support trend claims — only the fixed screen can. Weight sync
**N = 8** optimizer steps (measured swap 52–55 s, staleness 0 versions).
**Gate/screen evals stay at T = 0.7 / top-p 0.95, always as the K-draw mean**,
with the baseline re-screened through the same harness — single-draw vs K-draw
Pass@1 differ by 4 points on this suite and that gap once manufactured a
+4.75-point "gain" that was really −2.19.

### 12.7 Data pool and eval protocol
- **Training pool:** DeepMath-103K as the main pool (verified golds + difficulty labels → level-stratified probes, curriculum bands, and the v1 §12-style pass-rate stratification), with GSM8K supplying the easy tier. Record schema `_uid / prompt / ground_truth / level` as v1. The **SCA-comparison arm** additionally mirrors SCA's exact 5k three-stage curriculum (2,000 GSM8K → 1,400 GSM8K + 600 DeepMath ≤ diff-4 → 1,000 GSM8K + 500 low + 500 high) so that arm is apples-to-apples with their training recipe, not just their eval.
- **Eval:** SCA-matched protocol for all main tables — N = 8 per problem, T = 0.7, top-p 0.95, 32k max new tokens — on MATH-500, AMC23, MinervaMath, AIME24/25 (math) and HumanEval, GPQA-Diamond (cross-domain), reporting Pass@1 ± seed std, think length, answer length, and answer-segment-only quality. v1's `standard_eval_300` retained as the internal continuity dashboard (per-checkpoint auto-eval cadence). The Qwen model cards recommend slightly different sampling (T = 0.6, top-k 20) — report that setting once in an appendix; comparability with SCA wins for the main tables.
