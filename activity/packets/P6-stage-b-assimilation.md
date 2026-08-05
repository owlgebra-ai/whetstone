# P6 — Stage B: assimilation SFT (ZPD band-pass + SED) and the F3 gate

STATUS: in-progress (activity 009)
MACHINES: turing (baseline evals, training, checkpoint evals); spark (ZPD gate scoring on a π-of-round server); no GLM needed
DEPENDS ON: P5 done (activity 008, F2 PASS — certified corpus exists); P4 (`whetstone/sed.py` shared verbatim)
BLOCKS: P7 (Stage C RLs the Stage-B student); P8 comparisons use this packet's baseline evals
DELIVERABLES: `gsm8k_test` eval suite (finally built); **baseline eval card for the original checkpoint** (needed by every later packet); trained student (2 rounds, golden arm) + control-arm student (unfiltered corpus); entropy + length + register dashboards; **F3 verdict**; golden-vs-unfiltered comparison.

---

## 1. Objective — what Stage B is

The student — a fresh copy of the **original** Qwen3-1.7B — trains on the certified teacher corpus with a plain, unprivileged prompt. No register card, no gold, no verbose trace: **the register enters the weights here, and only here** (design §4). Two modifications to plain cross-entropy, both fixes to named v1 failures:

1. **ZPD band-pass token weighting** (fixes Diagnosis #1 — v1's surprisal *up*-weighting concentrated gradient on the least-learnable tokens): `w_t = σ(κ·(log π_S(τ_t) − γ)) · (1 + α_nov·min(S_t, s_cap))` — gate ≈1 where the student already assigns reasonable probability, ≈0 on residual spikes, novelty boosted only *inside* the reachable zone. κ=1, α_nov=0.5, s_cap=4 nats; γ from a measured histogram (Part 3).
2. **SED self-distillation** (fixes Diagnosis #3 — entropy collapse before RL): the CurioSFT term via `whetstone/sed.py`, **restoration mode** (H_pivot=0.6707, Δ_max=0.7, γ_e=1.0), α_sed=1. A **NEW EMA copy of the student** — never Round 0's, never shared (design §12.4).

Success is **F3**: accuracy within 1 pt of the starting checkpoint at **≤50% of its median think tokens**, with median per-token entropy **above** the audit baseline (restoration mode's stronger requirement). Expectation to hold in mind: the corpus thinks in ~189–390 tokens where the student natively thinks in ~6,099 — if assimilation works at all, the length collapse will be dramatic; the fight is keeping accuracy and entropy while it happens.

## 2. Read first

1. Design §4 (Stage B), §12.4 (SED/EMA — every bullet is a bug), §12.6 (knob table)
2. `/data/whetstone/corpora/stagea_golden/GOLDEN_HANDOFF.md` and `STAGE_B_HANDOFF.md` — **before touching the corpus**
3. Activity [008](../008-stagea-teacher-corpus.md) "What the next packet must know" + findings 6 (ZPD lower-bound caveat), 10b/10c (G_spike limits — context for why Stage B has no G_spike term at all), 13 (level-9 weakness)
4. Activity [007](../007-round0-inoculation.md) deviations 2–5 (the training-loop craft this packet inherits)
5. ROADMAP facts blocks 001–008

## 3. Inputs

| asset | value |
|---|---|
| **Training corpus (primary arm)** | `/data/whetstone/corpora/stagea_golden/golden_faithfulness.jsonl` — 2,414 problems × 1 certified trace, 750,087 think tokens. Level mix by problems: L1 35%; **by think tokens: L1 15.2%, L≥6 56.3%** |
| **Control arm** | `/data/whetstone/corpora/stagea_selected/selected.jsonl` — 11,954 traces / 3,994 problems, verified-only. **Weight per problem (1/n_kept or sample-one-per-epoch)** — n_kept ∈ {1,2,3} is sampling luck |
| Student init | **`Qwen/Qwen3-1.7B` — the ORIGINAL checkpoint.** Never `scorer_v1`, never any Round-0 artifact |
| SED kernel | `whetstone/sed.py` (007: 6 unit tests green) — H_pivot 0.6707, Δ_max 0.7, sync 5 optimizer steps, decay 0.99, τ̂∈[1.1,1.5], top-512 |
| Sequence construction | `whetstone/round0.py` — the shared build; **assert g=1 per record** |
| Entropy baseline | `/data/whetstone/runs/entropy_audit/per_token_entropy.npz` (native medians 0.0278/mean 0.3176/p80 0.6923) |
| Eval suites | `/data/whetstone/eval/` — `standard_eval_300` (frozen), primary 5; **`gsm8k_test` does NOT exist yet — Part 0 builds it** |
| Stack / ops | vllm 0.26.0 / torch 2.11.0+cu130 / transformers 5.14.1; venv activation; spark flags; all ROADMAP standing rules |

## 4. Part 0 — Missing prerequisites (half a day, mostly GPU-light)

**0a. Build `gsm8k_test.jsonl`** (the ROADMAP TODO that has slipped since the eval plan was ratified): add the suite to `build_eval_sets.py` — `openai/gsm8k` config `main`, split `test` (1,319 rows), same `_uid/prompt/ground_truth/level` schema, pin the revision, sidecar meta. Contamination pre-cleared (activity 002 Run 5: 0 hits vs the train pool). Gold extraction: text after `#### `, normalized (poolutil / the P1 recipe).

**0b. Baseline eval card for the original checkpoint** — F3 is "within 1 pt of the starting checkpoint", and those numbers **do not exist yet** (the audit's 6,099-token median was measured at T=0.9 sampling, not the eval protocol — protocols must match or the comparison is fiction). Run `run_eval.py` (P2 defaults: N=8, T=0.7, top-p 0.95, 32k, `enable_thinking=True`, no system prompt) for `Qwen/Qwen3-1.7B` on:
- `standard_eval_300` (full protocol),
- `gsm8k_test` (full protocol),
- the 5 primary suites (full protocol) — expensive (~a few hours) but this baseline card is consumed by F3, F4, P8, and the final tables; run it once, store it forever at `/data/whetstone/eval/baselines/qwen3-1.7b-original/`.

Record per suite: Pass@1 ± seed std, **think and answer token medians separately**, cap-hit rate, g-rate. These four numbers per suite are the fixed goalposts for the rest of the project.

## 5. Part 1 — Training data assembly

For each corpus record: `prompt_text = apply_chat_template([{user: prompt}], add_generation_prompt=True, enable_thinking=True)` (NO system message), `completion_text = "<think>\n" + compact_think + "\n</think>\n\n" + answer` — via `round0.build_completion_text`. Assert g=1; assert zero literal boundary-token strings in any text you add; assert `verify_response` passes (it must — the corpus is certified; a failure means assembly is wrong, not the corpus).

- **Loss masking:** assistant-only (completion tokens). CE with ZPD weights over **all completion tokens** (think and answer both — the student must learn to write the answer segment too); **SED applies to think tokens only** (its purpose is reasoning-entropy restoration; the kernel's API takes `think_mask`). Record this split in the journal as a pinned decision.
- **Per-sequence normalization by Σw** (design §4.1) — not by token count. A sequence whose tokens are mostly gated off must not have its few surviving tokens amplified by a small denominator against other sequences; cap the normalizer at `max(Σw, 0.25·n_completion_tokens)` and log how often the cap binds (if often, γ is wrong).
- Shuffle over problems each epoch, seeded. No curriculum ordering in round 1 (the token mix is already hard-weighted); level is logged per batch so per-level loss curves come free.

## 6. Part 2 — ZPD gate precompute (offline, per round — spark)

The w_t come from **one teacher-forced scoring pass under π_S-at-round-start** (design §12.2: precompute offline; the scorer within a round is frozen; SED cannot be precomputed and runs online).

- **Round 1: π_S = the ORIGINAL checkpoint.** Launch a second vLLM on **spark:8101** serving `Qwen/Qwen3-1.7B` (`VLLM_USE_FLASHINFER_SAMPLER=0`, `--gpu-memory-utilization 0.35`, `--port 8101`) — **spark:8100 still holds scorer_v1; do not touch it, do not use it for anything in this packet.** GB10's unified memory holds both comfortably.
- Score every corpus sequence (`prompt_logprobs=2` gives the actual-token logprob = −S_t per position; the `round0.py` path). Write per-token `logp` arrays into the training JSONL (or a parallel `.npz` keyed by `_uid` — recorded either way).
- **Round 2: π_S = the round-1 student.** Serve the round-1 checkpoint on spark:8101 (swap the model), re-score the **same corpus**, rewrite the gates. **Stale gates are the named drift failure** (CLAUDE.md invariant; design §4.3) — round 2 with round-1 gates is invalid, full stop.

## 7. Part 3 — Pin γ (and re-answer 006's masking question honestly)

Activity 008's ZPD histogram was measured under `scorer_v1` — a model with 91% of the register tax removed — and is therefore a **lower bound** on masking. From the Part-2 round-1 pass (original checkpoint), plot the per-token log-prob histogram over think tokens and:

1. Start γ = ln(1e-4) ≈ **−9.21** (design init).
2. Report the **masked fraction** (tokens with gate < 0.1) and the **boosted fraction** (novelty factor > 1.2), overall and **per level**. Sanity band: masked 5–30% overall. Masked > 40% in the hard band = the 006 fear (teacher outside the student's reach) materializing — report it prominently; the response is γ adjustment within reason, never pretending the tokens are learnable.
3. Pin γ in the journal + §12.6 table. Gate-shape plot (γ, κ against the histogram) into `activity/assets/009/`.

Also report the register-marker tokens' position in the histogram (`goal`, `⇒`, `chk` will sit in the far tail under the *original* checkpoint — ~40 nats for `goal`, activity 007 finding 1). **This is correct and expected**: the novelty cap (s_cap = 4) and the gate mean these tokens arrive with weight ≈ σ(κ(−40+9.2)) ≈ 0 — the register enters the weights gradually as π_S catches up across epochs and rounds, not by brute-force up-weighting in epoch 1. If the register never starts flowing (marker-token mean w_t stays ≈0 through round 1), that is an F3-relevant finding, not a reason to hack γ mid-run.

## 8. Part 4 — Trainer (`scripts/stageb_train.py`, new; inherits every 007 lesson)

**Config:** full-FT from the original checkpoint; **fp32 master weights + bf16 autocast + AdamW8bit** (the bf16-underflow trap at LR ~1e-5 is identical to Round 0 — 007 deviation 3; log `theta_drift_rel` every eval and treat 0 as a bug); LR 2e-5, warmup 30, cosine (remember: step 1 runs at LR 0 — harmless over ~900 steps); per-device batch 1, grad-accum 8; grad checkpointing; `row_logits` on completion rows only (007 deviation 4); SED think positions capped at 1,024/record; eval forwards under bf16 autocast (fp32 SDPA falls to the math backend and OOMs on long sequences — 007 deviation 5), strictly `no_grad`.

**Rounds and epochs:** Round 1 = 2 epochs over the golden corpus (~604 optimizer steps at effective batch 8). Then Part-2 gate recompute under the round-1 student, then **Round 2 = 1 further epoch with the fresh gates** (continuing from round 1's weights; **fresh EMA copy initialized from the round-1 student** — φ never carries over). *Deviation from design §4.3, attested:* the design's "fresh teacher batch" between rounds priced in a trainable teacher; with a frozen teacher a fresh batch costs a regeneration + judge quota. Round 2 runs on the same certified corpus with recomputed gates (the load-bearing half of the prescription); a fresh batch is warranted only if F3 fails in a way that implicates corpus staleness — journal the reasoning either way.

**Memory budget (32 GB):** θ fp32 6.4 + grads 6.4 + Adam8bit ~3.2 + φ bf16 3.4 ≈ 19.4 GB + activations (short sequences, grad-ckpt) — Round 0 peaked at 26.0 GB with the same recipe; expect similar. If OOM: φ to CPU with gather-at-sync (design §12.4's 8B fallback) before anything else.

**Per-eval metrics (every 25 optimizer steps; JSONL + PNGs):**
1. Loss components separately: weighted CE, K2 (SED). Healthy = they trade off (007 finding 8), not one collapsing.
2. w_t health: masked fraction, Σw-cap hit rate, register-marker mean w_t (the "is the register flowing yet" curve).
3. **Entropy trajectory** on a fixed control slice (teacher-forced, top-512): median AND mean AND p80 vs the audit baseline — mean is the sensitive one (007 finding 8: the median has no resolving power at 0.0278).
4. `theta_drift_rel`; EMA-sync counter (must be optimizer steps ÷ 5, exactly).
5. **Generative spot-check, 20 fixed val problems** (vLLM not needed — HF `generate`, greedy, 2k cap): median think tokens, marker density, g-rate, and **answer-segment marker leakage** (must stay ≈0 — the register belongs in think only). This tiny panel is the earliest visible signal that assimilation is happening: think length should start collapsing from 6k toward corpus scale within the first epoch.

**Checkpoints:** every 50 steps + end of each epoch, to `/data/whetstone/ckpt/stageb/<arm>/`. Keep all until F3 verdict (~20 saves × 3.4 GB bf16 export ≈ 70 GB — fine on /data).

## 9. Part 5 — Evaluation and the F3 gate

Per-checkpoint continuity: `standard_eval_300` **cheap mode** (K=1, T=0, 12k cap — flags exist from P2) — trend line only, never the verdict.

**F3 verdict (on the best round-2 checkpoint, chosen on cheap-mode continuity + entropy health):** full protocol (N=8, T=0.7, top-p 0.95, 32k, `enable_thinking=True`) on `gsm8k_test` + `standard_eval_300`, against the Part-0 baseline card:

- **F3a — accuracy:** within **1 pt** of baseline on both suites (Pass@1, ± std reported).
- **F3b — length:** median think tokens ≤ **50%** of baseline **under the same protocol** (expect far better — corpus median is ~189–390 vs baseline ~6k; report answer length separately and confirm it stays in the baseline's band — answers must NOT compress).
- **F3c — entropy:** median per-token entropy of the student on its own fresh rollouts (re-run `entropy_audit.py` generate-mode with the student checkpoint, same 200-problem protocol) **above** the audit baseline (restoration mode), with mean and p80 reported alongside.
- **F3d — form:** g-rate ≥ 99% on eval generations; answer-segment register leakage ≈ 0; verifier extraction rate at baseline level.

**On PASS:** run the 5 primary suites (full protocol) once for the record; freeze the student as `/data/whetstone/ckpt/student_b2/`; primary numbers go next to the Part-0 baseline card. **On FAIL:** the diagnosis tree is short — accuracy floor broken → over-assimilation (check entropy + w_t curves; consider α_sed↑ or 1 epoch less); length target missed → assimilation didn't take (check register-flow curve; consider the unfiltered arm's extra data); entropy floor broken → SED miscalibration (check τ̂ saturation at 1.5 = Δ target unreachable; H_pivot sanity). One attributable retry; otherwise journal and stop for design review.

## 10. Part 6 — Control arm (golden vs unfiltered, the 008 deviation's promised measurement)

Repeat Parts 2–5 with the **unfiltered** corpus (11,954 traces, per-problem weighting), same config, same rounds, same eval battery, checkpoints under `/ckpt/stageb/control/`. Report side by side: F3a–d, plus per-level accuracy deltas. This answers two questions at once: *does judge-filtering earn its quota* (the deviation's justification) and *does trace diversity matter* (the user's 2–3-keeps design lives only in this arm). ~2–4 h of GPU; do not skip it — it is the only planned measurement of either question.

## 11. Gotchas (each has already bitten someone)

- **fp32 master weights or the run is a silent no-op** at these LRs (bf16 quantum > update size). `theta_drift_rel` = 0 after step 2 ⇒ stop.
- **EMA:** update-not-copy; **optimizer** steps not micro-batches (accum 8 ⇒ sync every 40 forwards); φ initialized from the round's starting student; round 2 gets a **fresh** φ. Round 0's φ is dead — never load it.
- **Stale gates:** round 2 must re-score under the round-1 student. If the gate files' sidecar model-sha doesn't match the round's π_S, refuse to train (put the assert in the trainer, not in a checklist).
- **The two spark servers:** 8100 = scorer_v1 (leave alone), 8101 = π-of-round for gates. Killing the wrong one wastes an afternoon; check `--served-model-name` before kill.
- **Per-problem weighting on the control arm** — forgetting it triples the weight of 3-keep problems and silently reshapes the curriculum.
- Eval forwards: bf16 autocast + `no_grad`, or the math-backend attention OOMs on a long trace.
- `apply_chat_template(tokenize=True)` returns `BatchEncoding` — `list(enc["input_ids"])`.
- **No G_spike anywhere in this packet.** Stage B has no reward; and 008 (10b/10c) showed G_spike inverts in the hard band — anyone "improving" selection or weighting with it here is importing a known-broken signal.
- Level 9 (82 problems) is a known-weak slice of the corpus (23.6% faithful winners pre-filter): keep it, but track its per-level loss and eval separately; do not let it dominate any average you act on.
- Orphaned `VLLM::EngineCore` / never pipe vLLM into `head` / `source .venv/bin/activate` / remote-sync = its own verified command (`git rev-parse` both ends) — the full 001–008 operational canon applies.
- Baseline evals (Part 0) and training (Part 4) both want turing's GPU — run Part 0 **first**, not concurrently; the baseline card must exist before anyone is tempted to eyeball F3 against a made-up baseline.

## 12. Definition of done

- [ ] `gsm8k_test` built, revision pinned, schema-validated.
- [ ] Baseline eval card for the original checkpoint: all 7 suites, full protocol, think/answer medians separate, stored under `/data/whetstone/eval/baselines/`.
- [ ] γ pinned from the original-checkpoint histogram; masked/boosted fractions per level journaled; 006's masking question answered under the right model.
- [ ] Round 1 + gate recompute + round 2 complete on the golden arm; all curves in `activity/assets/009/`.
- [ ] **F3 verdict in bold** with all four sub-gates, against the Part-0 baselines, in `activity/009-stageb-assimilation.md`.
- [ ] On PASS: primary suites run once; `student_b2` frozen; non-winner checkpoints pruned.
- [ ] Control arm trained + evaluated; golden-vs-unfiltered table in the journal.
- [ ] ROADMAP facts block for activity 009; §12.6 table updated (γ, LR, epochs as-run); packet status flipped; P7 unblocked with the student checkpoint path.
