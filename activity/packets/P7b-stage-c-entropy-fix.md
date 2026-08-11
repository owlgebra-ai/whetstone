# P7b — Stage C rerun: entropy-regulated DAPO (pilot 2 → Phase 1)

STATUS: done (activity 011) — F4 substantive PASS; Stage-C endpoint g1200 (+ g400 efficiency alt); P8 unblocked
MACHINES: turing = rollout worker (+ all screens); spark = trainer (fp32 AdamW, resident frozen π_0). Topology is **forced, not preferred** — turing OOMs on fp32 AdamW (010 f7).
DEPENDS ON: activity 010 (F4 FAIL + diagnosis); all P7 infrastructure (built, 131 tests, reusable as-is)
BLOCKS: P7 Parts 3–5 (Phase 1/2/rescue); P8
DELIVERABLES: pilot 2 (arms below, one variable at a time) with a **re-gated F4 verdict**; on PASS, Phase 1 run to its Pareto endpoint under the fixed config; journal `activity/011-stagec-entropy-fix.md`.

---

## 1. Context — what happened and why this packet exists (read this even if you read nothing else)

Stage C's premise is **sound and measured**: the round-1 Stage-B student has strict Pass@1 46.35% vs pass@8 85.45% on the real 4,000-problem training pool — **39 points of headroom** — and **79.6% mixed groups** (~10× the original checkpoint's usable DAPO signal). RL's job is converting pass@k into pass@1.

The first pilot (activity 010) failed F4 on both clauses, and the diagnosis is **architectural, not the reward** (finding 23): Stage C's think side carried **two entropy-raising mechanisms** — DAPO's clip-higher asymmetry (ε_high 0.28 > ε_low 0.20) and TEA — and **no entropy ceiling anywhere**. That configuration was designed for v1's entropy-*collapsed* checkpoint. This checkpoint arrived at **10× baseline entropy** (Stage B's SED restoration over-delivered; F3c passed at median 10.1×). Result: think entropy rose 1.05 → **3.18 nats** in 60 steps, and the first casualty was the sequence's single most fragile low-entropy decision — emitting `</think>` (a boundary Round 0 had *already* degraded by 4 orders of magnitude, activity 007 f4). `missing_think_close` at the training sampler climbed **5.6% → 35.6%**, so by the end a third of the gradient scored sound reasoning as worthless.

Everything else **worked**: the reward battery caught 3 real defects pre-run; the empty-think attractor never fired (0.000 vs an 8.3% base rate); the 009 loop class is extinct at the 12,288 cap; the answer segment held (no round-2-style collapse); no reward hacking (93% of the widening leniency gap was format failure, checked directly, not grading-hole exploitation). And crucially — **the format failure does not exist at the eval protocol** (top-p 0.95): step-60's eval g-rate was 96.56%, *better* than the init. The model is not rotten; the training configuration overheats it.

Also established, and binding here: TEA was **not** the entropy driver (it reached 91 of 894,860 tokens at τ_c=1.0 — finding 19), so removing/keeping TEA is about its own usefulness, not about the fix.

## 2. Read first

1. Activity [010](../010-stagec-rl.md) — findings 12–14, 16–23 minimum; the findings index at the top is the map
2. [P7](P7-stage-c-rl.md) — §6 (loop), §6b (reward spec — unchanged here), §8 (curriculum — unchanged here)
3. ROADMAP facts blocks 009–010

## 3. Inputs — everything exists; nothing needs rebuilding

| asset | value |
|---|---|
| Init checkpoint | `/data/whetstone/ckpt/stageb/golden/round1/final` — pilot-2 restarts from HERE, not from any pilot-1 checkpoint |
| **Bucket table — REUSE, do not re-run** | `/data/whetstone/runs/stagec_buckets/phase1_init/` (3,184 mixed: bands 1,366 high / 972 mid / 846 low; 580 at 0/8; 5.9% at 8/8). Valid because the init is unchanged — curriculum-from-init is the rule, and the init is the same checkpoint. Re-bucketing costs 6.4 h and buys nothing |
| Init screen numbers (the F4 comparator) | strict Pass@1 **66.25% ± 1.46** (8-draw mean), think median **219**, answer 188, pass@8 93.5%, g 96.31% — from `/data/whetstone/runs/stagec/pilot/f4/`. **Re-screen the init anyway through your harness at verdict time** (010 f22 rule) |
| Pre-RL entropy card | `/data/whetstone/runs/entropy_stagec_init/` — think mean **0.620** / median 0.280 / p80 1.258 on fresh rollouts. The reference for "stable" |
| Reward | `whetstone/reward/{stagec,strict,register_math}.py` as shipped, **including the post-pilot register-leak fix** (line-initial only — the pilot ran the pre-fix module; you run the fixed one). **Re-run the full battery green before step 1** (standing rule: any reward change → battery first) |
| Loop/infra | `whetstone/{dapo,curriculum,rollout_bus}.py`, `scripts/stagec_{train,rollout_worker,bucket,dashboards,f4_screen,rescue}.py` — all tested, all reusable |
| Sampling | rollouts + bucketing T=1.0 / top-p 1.0 (arm C alters top-p, see below), cap **12,288**; screens/evals T=0.7 / top-p 0.95, cap 8,192; **every Pass@1 is the K-draw mean ± between-draw std** |
| Batch | **8 problems/step × K=8, `--prefetch` ON** (010 f12/f14: halves batch noise, ~65 s/step) |
| Sync | every 8 steps; bf16 **copy** export with the `|θ|`-bit-identical assert (010 f10 — `.to()` casts in place and erases 81× the update per sync; the assert is non-negotiable) |
| Other pins | LR 1e-6, group 8, λ_align 0.1, answer band f=32 target 288, B₀ 1,026 (measured), seed-per-rollout `sha1(uid:k:step:seed)` |

## 4. The arms — strictly one variable at a time, in this order

**Arm A (primary — the diagnosis's cheapest test): symmetric clipping.**
`ε_high = ε_low = 0.20`, **λ_TEA = 0**. Nothing else changes from pilot 1 except the batch size/prefetch pins above. 100 steps, checkpoints at 25/50/75/100. Rationale: removes the dominant entropy driver (clip-higher) and the irrelevant term (TEA) in one mechanically minimal change; finding 19 already proved TEA isn't the driver, so this isolates clip asymmetry.

**Arm B (only if Arm A's entropy still trends up): the thermostat.**
Add a bidirectional think-entropy regulator targeting the pre-RL card's band: `L_ceil = λ_ceil · max(0, H_think_batch − H_hi)²` with **H_hi = 1.2 nats** (the card's p80), λ_ceil start 0.05 (placeholder — pin it). This is the design-level amendment finding 23 implies: this checkpoint needs entropy *regulation*, not one-way preservation. Log the term's constant-when-inert statistic (its value when H < H_hi is exactly 0 — assert that in a test before running).

**Arm C (only if boundary failure persists at *stable* entropy): training-sampler top-p 0.995.**
The failure vanishes at top-p 0.95 (010 f21-revision), so clipping the extreme tail during training rollouts is a legitimate lever — but it truncates gradient support, i.e. a deliberate off-policy compromise. If used: label it as such in the journal, record the fraction of sampled tokens outside 0.995, and keep bucketing at the same setting.

**Optional Arm D (only alongside a PASSING arm, never to rescue a failing one): TEA back at τ_c = 3.0**, batch-scoped Cov, with **effective-tokens-protected** on the dashboard (usable window measured in 010 f18: at 1.0 it touches 91 of ~895k tokens; at 10 it is uniform/inert; 3.0 protects ~1,251). Justify keeping it by an observed difference, or leave it off — a term that changes nothing is a liability (the packet's standing rule: every regularizer logs a statistic that goes to a known constant when dead).

## 5. Per-step diagnostics (all existed by pilot's end — keep every one)

`theta_drift_rel` (nonzero, monotone), `H_think` from rollout logprobs, `g_rate` **at the training sampler** AND (per screen) at the eval protocol — the two diverge and mean different things (010 f21 vs its revision), `missing_think_close` rate, `empty_think` (guard curve), `lenient_only`, word-stutter, TEA `uniformity`/`effective_tokens` (if on), clip fractions both sides, answer-KL mean, answer median, batch mean p̂ **beside** acc (010 f20: acc tracks sampling at this batch size — never read a trend off unadjusted training curves; trends live in the fixed screen only).

## 6. The F4 re-gate (at arm completion; ~20 min of screens)

Protocol: fixed 200-problem screen, T=0.7/0.95, K=8, cap 8,192; **all arms AND the init through the identical harness in the same session**; every number the 8-draw mean ± std; paired McNemar per problem per draw.

- **Clause 1:** no named failure worsening monotonically across step windows at the training sampler — specifically `missing_think_close` must NOT reproduce pilot 1's 5.6→35.6 climb (flat ≈6% is acceptable: it's the pre-existing Stage-B property, cure queued for the card revision, not this run).
- **Clause 2:** ≥1 checkpoint **Pareto-dominates the init**: strict Pass@1 up (K-draw mean, paired) at equal-or-less think median. Also report think-per-correct (pilot 1's cleanest inverse signal: 331→367 monotone worsening).
- **Flat-but-healthy is not FAIL:** if entropy is stable, format flat, and Pass@1 unchanged (±1σ), the verdict is **EXTEND** — continue the same arm to 200 steps before judging (pilot 1 moved θ by only ~1e-4 relative; genuine flatness may just be "not enough optimization yet"). Degradation on any axis = FAIL the arm, move to the next arm.

**On PASS → Phase 1 immediately** (P7 Part 3 as written: saturation-paced curriculum from the reused bucket table, 8/step, prefetch, checkpoints every 25, Pareto endpoint, rescue at the boundary). Two Phase-1 obligations carried from 010: the **memorization re-read** is within-level against the pre-RL baseline of **+5.32 pts seen-over-unseen** (010 f11 — pooled reads hide 84% of it; alarm = the within-level delta widening materially, with a GLM spot-check on seen-problem correct rollouts), and the answer-collapse watch stays (correlation of answer median with step ≈ 0 expected; pilot 1 was clean).

## 7. Gotchas (every one earned in 010 — the expensive kind)

- **The bf16 export must be a COPY with the bit-identity assert.** `.to(bf16).to(fp32)` erases 81× an optimizer step per sync while `theta_drift_rel` keeps reading nonzero. The assert exists; never remove it.
- **Compare a statistic only against itself**: single-draw vs 8-draw Pass@1 differ by 4 points on the same model (f22 manufactured a +4.75 "gain" that was really −2.19). Init re-screened through the same code path, always.
- **Training-sampler health ≠ model health.** g_rate 0.658 at T=1.0/1.0 coexisted with 96.56% at eval protocol. Check the eval protocol before diagnosing rot — the two samplers answer different questions.
- **Never read trends off the training curves at this batch size** — length medians CV 0.82, acc tracks batch p̂ at r=+0.77. The fixed screen is the only trend instrument.
- **Loss magnitude ≠ influence**: TEA's scalar ran 5× the policy loss while touching 0.01% of tokens. For any term, report coverage (tokens actually receiving gradient) beside value.
- **Inert-component rule** (earned 4× in 010): every detector/regularizer logs a statistic that hits a known constant when the component is dead, asserted in a test. TEA's is `uniformity`; the thermostat's is exact-zero below H_hi; the leak detector's is the line-initial rule's zero on prose-`⇒` fixtures.
- Topology is forced: turing cannot hold fp32 AdamW (OOM, f7). Don't "simplify" onto one box — that silently changes numerics to 8-bit.
- Prefetch costs one step of curriculum lag and ≤1 sync of staleness — both logged, both fine; don't disable it to "clean up" the logs.
- Buckets and rollouts must share a sampler; if Arm C changes top-p, its Phase-1 re-buckets inherit it.
- Ops canon: venv activation; EngineCore orphans by PID (never `pkill -f` a pattern that matches your own ssh); matplotlib turing-only; chained scripts for unattended handoffs; remote sync = its own verified `git rev-parse` on both ends.

## 8. Definition of done

- [ ] Reward battery green (fixed leak detector in place) before step 1; thermostat/TEA inert-statistics tests added for whichever arms run.
- [ ] Arm A complete with the full diagnostic set; entropy trajectory plotted against the pre-RL card.
- [ ] **F4 re-gate verdict in bold** (PASS / EXTEND / FAIL per arm), init re-screened, all K-draw means, paired stats.
- [ ] Arms B/C/D only per their trigger conditions, one at a time, each re-gated.
- [ ] On PASS: Phase 1 to its Pareto endpoint; endpoint checkpoint named; re-bucket table; memorization within-level re-read + GLM spot-check; rescue round at the boundary.
- [ ] Journal `activity/011-stagec-entropy-fix.md`; ROADMAP facts block; §12.6 pins updated (clip, λ_ceil/H_hi if used, τ_c if TEA returns); P7 + this packet status flipped; P8 unblocked only when a Stage-C endpoint exists.
