# P7 — Stage C: segment-routed DAPO recovery RL (two phases + pilot) and the F4 gate

STATUS: blocked (F4 FAILED — activity 010; Parts 0–2 done, Parts 3–5 blocked on the entropy-ceiling fix) → **continuation packet: [P7b-stage-c-entropy-fix.md](P7b-stage-c-entropy-fix.md)** (pilot 2: symmetric clipping first, thermostat/sampler arms behind triggers; Phase 1 resumes on its PASS using this packet's Parts 3–5 unchanged)
MACHINES: **spark = trainer** (fp32 AdamW, unified memory); **turing = rollout server + π_0 anchor server** (user topology decision 2026-08-05, validated by the Part-2 pilot); GLM API (memorization spot-checks + rescue filter)
DEPENDS ON: P6/activity 009 (round-1 checkpoint, pass@8 evidence); P5 machinery (32B rescue generation); P4 (`whetstone/segments.py`, reward infra)
BLOCKS: P8 final comparisons
DELIVERABLES: pre-RL entropy card (F3c debt); strict-grading reward wrapper; the DAPO loop + segment routing + TEA; **50–100-step pilot with topology verdict and F4 gate**; Phase-1 recovery run; Phase-2 boost run on fresh problems; per-phase Pareto endpoints; journal `activity/010-stagec-rl.md`.

---

## 1. Objective — recovery first, boost second

Convert pass@k into pass@1. The round-1 student reaches the right answer on **90.5% of problems within 8 samples (89.5% strict) but only 66.5% on the first try** — reliability, not capability, is what Stage B lost, and reinforcement of self-sampled correct rollouts is the mechanism that restores reliability. The student is an unusually good RL substrate: **63% mixed groups (7.9× the original checkpoint's usable DAPO signal), 2× baseline entropy, register in-weights**, and — finding 17 — **correct rollouts are SHORTER than incorrect ones**, so compression pressure and accuracy pressure point the same way.

Two phases (user decision 2026-08-05): **Phase 1 (recovery)** on the 4,000-problem Stage-A set — 2,414 SFT-seen + 1,586 same-distribution unseen, tagged and tracked separately as a built-in memorization control. **Phase 2 (boost)** on a fresh 4–6k draw from the untouched ~26k pool, tilted toward wherever Phase 1's endpoint still shows mixed groups. A **50–100-step pilot** precedes any long run.

## 2. Read first

1. Design §5 (Stage C), §12.6 (knobs); v1 §7.3 (DAPO config), §7.6–7.7 (rollout investigation + stop rules — **kept verbatim**)
2. Activity [009](../009-stageb-assimilation.md) findings 14–17 (loops, verifier leniency, pass@8, length-correctness) — every one shapes this packet
3. Activity [008](../008-stagea-teacher-corpus.md) findings 10b/10c (**no G_spike-like training signal in the hard band — binding**)
4. Activity [003](../003-preconditions.md) finding 3 (the entropy second mode at ~0.7 nats → τ_c sweep note)
5. ROADMAP facts blocks; `whetstone/reward/` package docs (three-tier R_acc survives; uniform KL + char-count length die)

## 3. Inputs

| asset | value |
|---|---|
| Init checkpoint | `/data/whetstone/ckpt/stageb/golden/round1/final` — **round 2 is dropped** (worse on every axis) |
| Phase-1 pool | Stage-A `subset_stagea_uids.json` (4,000) + `golden_faithfulness.jsonl` uid list → per-problem `seen` (2,414) / `unseen` (1,586) tags |
| Phase-2 pool | drawn in Part 6 from `train_30k.jsonl` minus the 4,000 |
| Anchor model | `Qwen/Qwen3-1.7B` original — answer-segment forward-KL target |
| Rescue teacher | `nvidia/Qwen3-32B-NVFP4` + P5's `teacher_generate.py` (off until Part 7) |
| Baselines | baseline card (`/data/whetstone/eval/baselines/qwen3-1.7b-original/`); 200-problem screen + its round-1 numbers (66.5/90.5 as-scored, 64.25/89.5 strict); entropy audit npz |
| Rollout cap | ~~8,192 tokens~~ → **12,288 tokens** (CORRECTED during execution, activity 010 finding 4: the 8,192 figure and its "truncates 0/377" justification came from 009 Run 12, which was **GSM8K only**. This pool is 50% DeepMath, where 8,192 truncates **5.58%** of well-formed generations and 12,288 truncates **0.00%**. A truncated-but-legitimate generation scores `g=0 → R_acc=0`, so the old cap would have taught the model that hard problems are unsolvable when they are merely long) |
| Eval discipline | gsm8k_test = validation; `--limit 300 K=1 T=0` continuity; primary suites only at F4-pass/final; **strict + as-scored reported everywhere** (finding 15) |

## 4. Topology (user decision, with the required refinements)

```
turing (RTX 5090)                          spark (GB10, 128 GB unified)
┌──────────────────────────────┐           ┌─────────────────────────────┐
│ :8000 student rollout vLLM   │  rollouts │ trainer (fp32 AdamW — NOT   │
│   (current policy, reloaded  │──────────►│  8-bit; memory is abundant  │
│    every sync)               │   /data   │  and bitsandbytes-on-ARM is │
│ :8002 π_0 anchor vLLM        │◄──────────│  an unforced risk)          │
│   (original ckpt, frozen —   │  weights  │ CPU: verify (strict), parse,│
│    prompt_logprobs for KL)   │   sync    │  advantages, bucketing      │
└──────────────────────────────┘           │ :8100 scorer_v1 — IDLE, do  │
                                           │  not touch, not used here   │
                                           │ :8101 π-round1 — STOP IT    │
                                           │  (Stage B done; free mem)   │
                                           └─────────────────────────────┘
```

- Both 1.7B servers fit on turing together (3.4 GB weights each; gpu-mem 0.45 / 0.30). π_0 anchor scoring rides turing because turing idles while spark trains — free overlap.
- **Weight sync:** trainer writes bf16 export to `/data/whetstone/ckpt/stagec/live/` every **N optimizer steps** (start N=8); turing swaps the rollout server to it. Pick the swap mechanism during the pilot (vLLM sleep/wake reload vs full restart ~75 s); DAPO's clipping tolerates the ~1-step staleness this creates — **record the measured staleness, don't hide it**.
- **The pilot's first deliverable is the pipeline balance:** wall-clock per step split into (rollout gen | scoring | sync | trainer step). If spark's trainer step exceeds ~2× turing's batch generation time, the topology inverts and turing idles — the documented fallback is full time-multiplex on turing (sleep/wake), which the design always assumed. Do not push a losing topology uphill for aesthetics.

## 5. Part 0 — Prerequisites (before any RL step)

1. **Entropy card / F3c debt:** `entropy_audit.py` generate-mode on the round-1 student (exact 009-pinned protocol: val_2k, n=200, seed 0, T=0.9, 16,384 cap). This is both the owed F3c measurement and **TEA's calibration baseline** — record median/mean/p80 and the mode structure (003's second mode sat at ~0.7 nats; see the τ_c note in Part 1).
2. **Strict-grading reward wrapper** in `whetstone/reward/` (NOT in `verify.py` — CLAUDE.md invariant): exact + numeric equivalence only; **no suffix fallback; no extraction at all when `</think>` never closed**. Finding 15 measured the leniency rewarding degenerate models 14× more than the baseline — RL against lenient grading would optimize straight into those holes. Unit-test the wrapper on finding 15's cases (gold 200/pred 0 must be WRONG; unfinished think must be WRONG).
3. **Stop spark:8101** (π-round1 server — Stage B is over). Verify :8100 untouched.
4. **Phase-1 bucketing:** K=8, ~~T=0.7, 8,192 cap~~ → **T=1.0, top-p 1.0, 12,288 cap** on all 4,000 problems under the init checkpoint (CORRECTED during execution, activity 010: the T is resolved against the Part-1 sampling table, which explicitly scopes "Parts 0.4, re-buckets" to T=1.0 and gives the reason — buckets must match the rollout sampler or they mis-predict group composition; the cap per finding 4 above. Measured cost is **~6 h**, not ~1 h — the ~1 h estimate extrapolated GSM8K screen throughput to a pool that is half DeepMath). Buckets: 0/8, 1–7/8, 8/8, **split by seen/unseen** and by level. This table is Phase 1's curriculum AND the memorization check's baseline — if seen problems bucket dramatically easier than unseen at the same level, say so loudly before training on them.

## 6. Part 1 — The loop (`scripts/stagec_train.py` + `whetstone/dapo.py`, new)

**DAPO backbone (v1 §7.3 + design §5.1):** clip 0.2/0.28 (clip-higher), group 8, LR 1e-6, token-level policy loss, **dynamic sampling** — oversample problems, drop all-correct and all-wrong groups from the gradient (they carry zero within-group advantage; log the drop rate — it is the phase-exhaustion signal).

**Sampling parameters (pinned, with the reasoning on record):**

| context | T | top-p | why |
|---|---|---|---|
| **RL rollouts** | **1.0** | **1.0** | policy-gradient correctness — the gradient assumes samples from π itself; T<1 sharpens, top-p truncates gradient support, T>1 flattens off-policy *and* feeds the loop tail (derailments are stochastic and near-absorbing, 009 f14; heat measurably increases artifact rates, 008 f11). Exploration is not the scarce resource — 63% mixed groups and 2× baseline entropy already; TEA guards entropy during training, not the sampler |
| curriculum bucketing (Parts 0.4, re-buckets) | 1.0 | 1.0 | **must match rollout sampling** or buckets mis-predict group composition |
| continuity evals | 0.0 | — | greedy trend line, as spec'd |
| gate/screen/final evals | 0.7 | 0.95 | the SCA-matched protocol — comparability wins; never train against it |
| rescue generation (32B) | 0.8 | 0.95 | P5's pinned regime |

**Pilot micro-check:** one 200-problem K=8 pass at T=1.0 vs the existing T=0.7 numbers — confirm the loop/cap-hit rate stays acceptable (< ~5%) and mixed-group fraction holds. If T=1.0's loop rate explodes, fall back to T=0.9 and record the off-policy compromise explicitly rather than silently.

**Reward per rollout:** three-tier `R_acc` from `whetstone/reward` **wrapped strict** (Part 0.2). Malformed (g=0, incl. cap-hits) → excluded from all structural rewards, R_acc=0 — the loops (2.7%) die by construction, no special mechanism.

**Segment routing (the design's replacement for uniform KL):**
- **Think tokens:** soft length tail (v1 form, budget from the realized group spread — a reward must never demand lengths outside it) + **TEA**: per batch `Cov_t = centered(log p_t)·centered(A_t)`; `weight_t = min(softmax(Cov_t/τ_c), c/|T_r|)`; `L_TEA = |T_r|·Σ weight_t·H_t` subtracted with λ_TEA. Start (τ_c, λ_TEA, c) = (1.0, 0.05, 100) **and sweep τ_c ∈ {0.7, 1.0} in the pilot** — 003 measured this checkpoint's real entropy mode at ~0.7 nats, below the design's τ_c=1.0 (the flagged concern; the Part-0 entropy card refreshes the number for *this* student). **No style anchor on think tokens** — changing that register is the point.
- **Answer tokens:** forward KL to π_0 via the per-token logprob-diff estimator (π_0 logprob of the sampled token from turing:8002 `prompt_logprobs` prefill — one pass per rollout batch), λ_align 0.1 (pin in pilot), + SCA length band f=32 around the answer-length target. Round 2 proved answers collapse (288→19) without this anchor; watch answer medians per checkpoint.
- **Difficulty amplification:** W(x) = 1 + 0.5·(1 − p̂_succ(x)) applied to **positive think advantages only** (design §5.2).
- **NO G_spike-like terms anywhere** (008: inverts at the hard tier). TEA's diagnostics (weights, H_t at selected tokens) logged from step 1 — the 009 gap of unlogged SED internals does not get repeated here.

**Per-checkpoint rollout investigation (v1 §7.6–7.7 verbatim):** sample and *read* rollouts every eval; the named rot patterns plus this project's own — announce-never-execute loops, runaway `chk:` chains, `case N:` enumeration, answer-segment register leakage, think/answer contradiction. **Finding 11 is the standing law: only generative inspection catches death; losses will look fine.**

## 6b. Part 1b — Reward specification (craft it like an instrument, then unit-test it like one)

Read `trashed/WHETSTONE_STAGE5_REWARD_DESIGN.md` first — Stage C's reward is an *adaptation* of that catalogue, not a fresh invention. The separation below is the first thing to get right:

| enters the **scalar reward** (→ group advantage) | enters the **loss directly** (token-level) |
|---|---|
| r_acc (strict), r_fmt, think-length shaping, answer band, penalties below | TEA (think), forward-KL to π_0 (answer), clip objective |

Mixing these up (e.g., putting KL into the scalar reward) recreates v1's uniform-anchor mistake by the back door.

**Composition (additive, v1 §2.3 magnitude-budget style):**

- **r_acc:** strict-correct **1.0**, wrong **0**. The v1 *lenient* tier is **retired for RL** — finding 15 showed lenient grading is exactly where degenerate policies farm reward. (Keep lenient as a *logged diagnostic*, never a reward.)
- **r_fmt ≈ 0.10** for g=1 with both boundaries and a non-empty answer (v1 invariant I1) — so well-formed-but-wrong beats malformed, giving the 4% loop tail a gradient toward form even at R_acc=0.
- **Think-length shaping is a tail, NEVER a monotone bonus — with an empty-think guard.** ⚠ **The empty-think attractor is live and measured**: this model already knows how to emit `<think>\n</think>` + a correct answer (009, no-floor run, step 100), the parser scores empty think as g=1, and "correct + shorter is better" makes empty think the global optimum on every easy problem. Guards, all three: (i) length reward is `exp(−max(0, T_think − B)/B)` — **zero benefit below B**, only cost above it; (ii) **T_think < 16 tokens ⇒ r_fmt = 0 and no structural bonuses** (treated as a format violation, logged as `empty_think`); (iii) the `empty_think` rate is a first-class dashboard curve — if it climbs, stop and read rollouts. B from the realized group spread per the freeze rule.
- **Answer band:** SCA-style band (f=32) around the baseline answer median (288 — the π_0-anchored target, not the corpus's 189: the anchor and the band must agree on what an answer looks like). Soft penalty outside; this plus the KL term is the round-2-collapse (288→19) protection.
- **Penalty catalogue triage (from v1 §4, each with its v2 verdict):**
  - **KEEP — think/answer contradiction (v1 §4.10):** the think concludes X, the answer states Y. Observed live in this project (005 hand-inspection: think 6200, answer 6600). Detector: last `⇒` value vs boxed answer, **register-aware normalization** (Unicode↔LaTeX, 005 finding 14 — a naive comparison misgrades the register's own notation; if the normalizer isn't ready, log-don't-penalize until it is).
  - **KEEP — post-think register leakage (v1 §4.7):** register-specific detector (line-initial `goal:`/`chk:`, `⇒`/`✗`), not bare substrings — the English word `case` appears in 10% of honest answers (009 finding 1).
  - **KEEP — answer repetition (v1 §4.6)** and **n-gram loop penalty (v1 §4.3)** at reduced weight: g=0 catches *terminal* loops; these catch partial loops inside completed rollouts.
  - **DROP — char-count length (§4.1), chunk-structure penalties (§4.2, §4.5):** replaced by token-level segment shaping; v1's chunk formalism doesn't exist in this register. Keep counter-restart *detection* as a logged diagnostic only.
- **Invariants, asserted in code (adapted from v1 §5):** (I1) r_fmt ≥ 0.10 whenever both boundaries present and think ≥ 16 tokens; (I2) the **worst-scoring correct** rollout (verbose, penalties applied) outranks the **best-scoring wrong** one by ≥ 0.30 — accuracy must dominate style, always; (I3) every structural bonus and the length reward **gate on strict-correct** — style reward on wrong answers is how registers get farmed.

**The reward test battery (`tests/test_stagec_reward.py`) — mandatory green before the pilot's first step.** Synthetic rollouts asserting the full ordering, the same discipline as Round 0's meter tests:

```
correct + compact register + clean        →  highest
correct + verbose think                   →  lower (tail), still ≫ any wrong
correct + EMPTY think                     →  below correct-verbose (guard works)
correct + think contradicting the answer  →  penalized below correct-verbose
correct + register leaked into answer     →  penalized
wrong + well-formed                       →  ≈ r_fmt only; ≥0.30 below worst correct
finding-15 leniency cases (gold 200/pred 0; unfinished think with boxed gold inside) → graded WRONG
loop / cap-hit (g=0)                      →  floor
```

Plus property checks: I1–I3 hold on 200 randomly-perturbed synthetic cases; every component logged per rollout via the v1 `RewardBreakdown` machinery (it survives for exactly this). **A reward change of any kind mid-run re-runs the battery first** — the battery is cheap and Goodhart is not.

## 7. Part 2 — THE PILOT: 50–100 steps, then a hard stop and a verdict (user-mandated)

Run Phase-1 config for 50–100 optimizer steps. Deliverables, all go/no-go:

1. **Topology verdict:** step-time breakdown; pipeline balance; sync staleness; spark trainer throughput. Keep the split or fall back to turing-multiplex — decided by numbers, in the journal.
2. **F4 gate (design §11), evaluated on this window:** 50 DAPO steps with **no critical rollout-investigation flag**, and ≥1 checkpoint **Pareto-dominating the init** on the easy suite (200-screen: strict Pass@1 up at equal-or-less think median). F4 PASS/FAIL stated in bold.
3. **Reward-integrity check:** loop rate trending to zero; no new degenerate mode in read rollouts; strict-vs-as-scored gap not widening (a widening gap = the policy found a grading hole).
4. **Entropy trajectory** (TEA health): no collapse below the Part-0 card; τ_c sweep read-out.
5. **Memorization watch:** seen-vs-unseen Pass@1 delta at pilot end + GLM spot-check (~30 judgments) on seen-problem *correct* rollouts — does the think actually derive? A big seen-side jump with confabulated derivations = recall-reward, and the response is dropping seen 8/8-adjacent problems, not celebrating the curve.
6. Continuity: gsm8k_test `--limit 300 K=1 T=0` every 10 steps.

**Only on a clean pilot** does Phase 1 continue past step 100.

## 8. Part 3 — Phase 1: recovery (the 4,000 set)

**Batch curriculum (user direction 2026-08-05, refined): success-rate-ordered, saturation-paced.** Batches sample from the mixed-group (1–7/8) pool with weights tilted by measured pass rate, NOT by level label and NOT on a step schedule:

- **Early: ~75% from the p̂ ≥ 5/8 bucket** (empirically this is mostly GSM8K/level-1 — the user's easy-first intent, delivered by measurement), **~25% from lower buckets** — never 0%, because the loop tail lives on hard problems and the g=0-vs-r_fmt within-group contrast is the gradient that extinguishes it; a pure-easy diet postpones the cure.
- **Progression is saturation-driven:** problems that reach 8/8 auto-drop (dynamic sampling); re-tilt the weights at every 25-step eval from the live per-bucket mixed fractions — when the easy tier's mixed share falls below ~⅓, shift the tilt one bucket down. No fixed switch step: the buckets know whether "step 15" is too early or too late.
- 0/8 problems stay out of batches entirely (rescue's clientele, Part 5); difficulty amplification (W(x), positive think advantages) is the late-curriculum instrument on the low-p̂ buckets.
- Log per-bucket: batch share, advantage variance, loop rate — if the pilot shows low-p̂ groups injecting outsized variance early, tighten the early tilt and record it.

Continue to the phase endpoint. Cadence: cheap continuity every 25 steps; 200-screen (K=8, strict+as-scored) every 100; checkpoints every 25 (~3.4 GB each — prune losers after endpoint selection). **Endpoint = accuracy × tokens-per-correct Pareto frontier** (v1 §7.9), with v1's expectation that over-training past ~80% of steps is the failure mode — TEA moving that boundary later is itself a measurable claim; measure it.

Provisional success target (pin honestly at pilot, not after): **strict Pass@1 ≥ 80% on the 200-screen at think median ≤ 500** — i.e., convert ≥60% of the 25-pt strict headroom while staying ≥3× compressed. Re-bucket all 4,000 at endpoint: the mixed-group fraction remaining, per level and seen/unseen, is Phase 2's input.

## 9. Part 4 — Phase 2: boost (fresh problems)

Draw 4–6k from the untouched ~26k (`poolutil`, stratified, **tilted toward the levels where the endpoint re-bucket still shows mixed groups** — expected 5–8, but let the re-bucket decide). Fresh K=8 bucketing under the Phase-1 endpoint checkpoint (design rule: per phase, from the current init). Same loop, same gates, same cadence. Phase-2 endpoint: same Pareto rule; if mixed groups are still plentiful at endpoint, a Phase 3 is legitimate — **the re-bucket, not the calendar, decides when RL is done.**

## 10. Part 5 — Pedagogy rescue (0/8 problems, both phases)

Per design §5.2, adapted to the frozen teacher and the 008/009 evidence: 32B generates M=4 candidates (card-conditioned, `gold+trace` where a trace exists — gold-only confabulates at 73.7% on hard problems, 008 f13); filter on **strict verify + g=1 + in-register + GLM faithfulness** (NOT G_spike — 008; this replaces the design's G_spike-threshold filter, deviation attested); assimilate survivors with the Stage-B loss (whitelist floor active, LR 5e-6, ≤1 epoch, **fresh EMA**); rescued problems re-enter the next K=8 refresh. Run at phase boundaries, not continuously. Track: rescue yield per level, and whether rescued problems actually move buckets next refresh.

## 11. Gotchas

- **Strict reward or nothing.** RL against `verify.py`'s lenient tail is optimizing into measured holes (14× differential, finding 15). The wrapper lives in `whetstone/reward/`; `verify.py` stays untouched.
- **The reward test battery (Part 1b) gates the pilot** — no RL step until it is green, and it re-runs before any mid-run reward change. The empty-think guard is its most important case: without it, "correct + short" makes `<think>\n</think>` the global optimum and RL finds it in tens of steps.
- **spark trains, turing generates — never the reverse.** Spark decode measured 10× slower (009 ops note). If the pilot inverts the topology, fall back to turing-multiplex; don't move rollouts to spark.
- **No bitsandbytes on spark** — fp32 AdamW everywhere on the trainer; memory is abundant. (`theta_drift_rel` logging stays — it has caught two silent no-ops already.)
- **Two turing servers:** check `--served-model-name` before any kill; the anchor server (:8002) is frozen π_0 — reloading IT with student weights corrupts the KL anchor silently. Put the model-sha assert in the scoring client.
- Weight-sync races: rollout server must never serve a half-written export — write to a temp dir, atomic rename, then swap. A torn checkpoint generates garbage that *parses* (g=1) and poisons a whole batch.
- Sampling params on rollouts: T=0.7 top-p 0.95 seed-per-rollout (`sha1(uid:k:step:seed)`) — byte-identical group members make within-group advantage zero and DAPO silently learns nothing from them.
- **Advantage routing is token-masked by `parse_segments`** — never by string offsets; assert think/answer masks sum to completion length per rollout.
- The length tail obeys the freeze rule: budget from realized group spread, pause tightening when within-group think std < 40 tokens.
- 0/8 ≠ hopeless; 8/8 ≠ done: dynamic sampling drops both from gradients, but **log both rates per eval** — rising 8/8 is progress, rising 0/8 is rot (v1's named pattern).
- Cross-machine canon: venv activation; `VLLM_USE_FLASHINFER_SAMPLER=0` on spark vLLM (scorer only — the *trainer* doesn't sample); EngineCore orphans killed by PID; never pipe vLLM into `head`; sync = its own verified command (`git rev-parse` both ends); matplotlib is turing-only.
- **Entropy readings during RL come from the trainer's own logprobs** (top-512 on rollout tokens) — cheap, but distribution-of-sampled-tokens ≠ full-dist entropy; the binding measurements are the Part-0-protocol audit re-runs at phase endpoints.
- The GLM quota is finite: budget ~30 judgments for the pilot memorization check, ~100 per rescue round, and leave headroom for P8.

## 12. Definition of done

- [ ] Part 0 complete: entropy card (F3c settled + TEA baseline), strict wrapper unit-tested on finding-15 cases, :8101 stopped, Phase-1 buckets with seen/unseen split journaled.
- [ ] **Reward test battery green** (full ordering + I1–I3 property checks + per-component `RewardBreakdown` logging verified) before pilot step 1.
- [ ] **Pilot report: topology verdict, F4 verdict in bold, τ_c choice, memorization read** — before step 101.
- [ ] Phase 1 to Pareto endpoint; endpoint checkpoint named; re-bucket table.
- [ ] Phase 2 drawn, bucketed, run to endpoint; Phase-3 decision from its re-bucket.
- [ ] Rescue rounds logged with yield; strict+as-scored reported side-by-side everywhere.
- [ ] Final: full-protocol gsm8k_test on the endpoint checkpoint (loops should be extinct, making it affordable) + primary suites once; numbers beside the baseline card.
- [ ] Journal `activity/010-stagec-rl.md`; ROADMAP facts block; §12.6 updated (τ_c, λ_align, N_sync as pinned); packet flipped; P8 unblocked.
