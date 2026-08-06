# 010 — Stage C: segment-routed DAPO recovery RL

- **Packet:** [packets/P7-stage-c-rl.md](packets/P7-stage-c-rl.md)
- **Status:** in-progress
- **Machine(s):** mac (code) / turing (rollouts + π_0 anchor) / spark (trainer)
- **Code commit(s):** `624c8b8` (packet claim) →
- **Started:** 2026-08-05

## Goal

Convert pass@k into pass@1 on the round-1 Stage-B student. 009 established the
substrate: pass@8 90.50% (89.50% strict) against Pass@1 66.50% (64.25% strict),
63% mixed groups, 2× baseline entropy, correct rollouts *shorter* than incorrect
ones. Stage C is the stage built for unreliability. Deliverables: the pre-RL
entropy card (F3c debt), a strict-grading reward, the DAPO loop with segment
routing + TEA, a 50–100 step pilot carrying the **F4 gate**, then Phase 1
(recovery, 4,000 problems) and Phase 2 (boost, fresh draw).

---

## Packet corrections found before execution

Two internal contradictions in P7, resolved here and noted so the next reader
does not re-derive them.

**(1) Bucketing temperature.** Part 0.4 says the Phase-1 K=8 bucketing runs at
`T=0.7`; the Part-1 sampling table says bucketing must run at **`T=1.0`, top-p
1.0** and explicitly names "Parts 0.4, re-buckets" as its scope, with the reason
attached: *"must match rollout sampling or buckets mis-predict group
composition."* **Resolved in favour of the table** — it is the later, reasoned
statement, it names the earlier one, and its argument is correct: a bucket table
built at a different temperature than the sampler predicts the wrong group
composition, which is the one thing the table exists to do.

**(2) Rollout sampling params.** The §11 gotcha list says rollouts run at
"T=0.7 top-p 0.95 seed-per-rollout"; the Part-1 table pins **T=1.0 / top-p 1.0**
with a policy-gradient-correctness argument (the gradient assumes samples from π
itself; top-p truncates gradient support). **Resolved in favour of the table.**
The gotcha's surviving content is the *seed-per-rollout* rule
(`sha1(uid:k:step:seed)`), which is independent of temperature and is kept:
byte-identical group members make within-group advantage zero and DAPO silently
learns nothing from them.

Both resolutions are testable at the pilot: Part 1's micro-check (200-problem
K=8 at T=1.0 vs 009's T=0.7 numbers) is exactly the measurement that would
falsify them, and it runs before any long run.

---

## Runs

### Run 1 — 2026-08-05, Part 0.3: stop `spark:8101`

Nothing to stop. `ps aux | grep -iE "vllm|python"` on spark showed **no vLLM
process at all** — neither :8101 (π-round1, Stage B's server) nor :8100
(`scorer_v1`). Both had already exited at the end of activity 009. Port 8000 is
the unrelated `llama-swap` service and was left alone, per activity 001.

Verified listening ports on spark: 22, 53, 80, 631, 8000, 8080, 9400, 11000,
plus tailscale. **No 81xx.** Precondition satisfied; nothing killed, so no risk
of the `pkill` self-match gotcha.

### Run 2 — 2026-08-05 20:00, Part 0.1: entropy card on the round-1 student (F3c)

The F3c measurement 009 left owed, doubling as TEA's calibration baseline.
Protocol copied verbatim from the audit's own `config` block so the student and
the baseline are measured identically.

- machine: turing (GPU idle, 18 MiB at launch)
- commit: `624c8b8`
- command (two-phase — vLLM does not release GPU memory promptly, so generation
  and teacher-forced scoring run as separate invocations):

```
python scripts/entropy_audit.py \
  --model /data/whetstone/ckpt/stageb/golden/round1/final \
  --pool /data/whetstone/data/pool/val_2k.jsonl \
  --out_dir /data/whetstone/runs/entropy_stagec_init \
  --n 200 --seed 0 --temperature 0.9 --top_p 0.95 \
  --max_tokens 16384 --max_len 20480 --chunk 1024 [--generate_only]
```

- generation: 200 rollouts in ~2 min → `rollouts.jsonl`
- outputs: `/data/whetstone/runs/entropy_stagec_init/{audit.json,per_token_entropy.npz,*.png}`

**F3c — the owed measurement. PASSES, on fresh rollouts.**

| think entropy | baseline (original ckpt) | **round-1 student (fresh)** | ratio |
|---|---|---|---|
| mean | 0.31759 | **0.61977** | 1.95× |
| **median** | 0.027817 | **0.27991** | **10.1×** |
| p80 | 0.69234 | **1.25825** | 1.82× |
| collapse mass (<0.1) | 56.8% | **39.6%** | — |
| fork mass (>1.5) | 2.8% (003, native) | **14.6%** | 5.2× |

F3c's criterion is "median entropy ≥ audit baseline"; the student is at **10.1×**
it. The gate is met with enormous margin, and the fresh-rollout number confirms
009's teacher-forced control slice almost exactly (that slice read mean 0.6455 /
median 0.2603 / p80 1.2282 against this run's 0.6198 / 0.2799 / 1.2582) — two
different measurement paths agreeing is worth more than either alone.

Segment split (answers are *not* where the entropy is): answer mean 0.3758,
median 0.0300, collapse mass 57.9%. Think carries the entropy; the answer
segment is still near-deterministic, which is the correct shape.

> **Finding 3 — the second entropy mode is at ≈0.725 nats on the student too,
> confirming activity 003 on a checkpoint two stages downstream.** Histogram of
> the non-collapsed think mass (60 bins over 0.1–3.1 nats) shows a monotone
> decay from the collapse spike to a **dip at 0.525** (8,475), a genuine **local
> maximum at 0.725** (11,185), a **dip at 0.925** (8,072), then a long tail.
> 003 measured this on the *original* checkpoint and warned that "TEA's τ_c = 1.0
> sits above the real second mode"; Stage B did not move it. The packet's
> τ_c ∈ {0.7, 1.0} pilot sweep is therefore well-targeted — **0.7 sits on the
> mode, 1.0 sits in the dip past it.** Mass above the design's 1.5-nat "fork"
> threshold is 14.6%, so a component hard-coding 1.5 would act on the tail and
> miss the mode entirely.

> **Finding 4 — the packet's 8,192 rollout cap was justified on GSM8K numbers
> and truncates 5.6% of well-formed generations on the actual Phase-1 pool.**
> P7 §3 pins 8,192 citing 009 Run 12's "truncates 0/377 well-formed
> generations". That run was **GSM8K only** (think p99 3,197). The Phase-1 pool
> is **50% DeepMath** (2,000 GSM8K L1 + 2,000 DeepMath L2–L10 — measured below),
> and on the DeepMath distribution this student runs far longer: think p50 2,218,
> p95 7,681; total completion p50 3,156, p95 8,445, max 10,248.
>
> | cap | well-formed generations truncated |
> |---|---|
> | 8,192 (packet) | **11/197 = 5.58%** |
> | 10,240 | 1/197 = 0.51% |
> | **12,288 (adopted)** | **0/197 = 0.00%** |
> | 16,384 | 0/197 = 0.00% |
>
> **Cap raised to 12,288** for both bucketing and RL rollouts (they must match,
> same argument as the temperature resolution). This is the same artifact class
> activity 008 caught with its answer budget — "1,024 cap-outs hit 15.4% of
> level 6 and 0% of level 1, an artifact biasing against the hard tier". Under
> RL the bias is worse than a mis-measured eval: a truncated-but-legitimate
> generation is scored `g=0 → R_acc=0`, so the reward would teach the model that
> hard problems are unsolvable when they are merely long. 12,288 still starves
> the loops (2.7× below the 32k budget at which they were characterised), and
> costs little since only ~5% of generations reach past 8,192 at all.

Also from this run, under the init checkpoint on DeepMath val at T=0.9:
`g=0` **1.50%** (3/200, all `missing_think_close`), think p25 **745** — the
latter is the `ThinkBudget` spread floor's order of magnitude on the hard half.

### Run 4 — 2026-08-05, Part 0.4 prep: Phase-1 pool composition

`/data/whetstone/corpora/stagea/subset_stagea_uids.json` (4,000) against the
golden corpus's uid list (2,414). Split confirms the packet exactly: **2,414
seen / 1,586 unseen**, 0 uids missing from `train_30k.jsonl`.

**Source is 50/50 and perfectly confounded with level:** all 2,000 level-1
problems are GSM8K; every level 2–10 problem is DeepMath.

| level | total | seen | unseen | seen % |
|---|---|---|---|---|
| 1 (GSM8K) | 2,000 | 844 | 1,156 | 42.2% |
| 2 | 38 | 5 | 33 | 13.2% |
| 3 | 100 | 88 | 12 | 88.0% |
| 4 | 110 | 105 | 5 | 95.5% |
| 5 | 353 | 327 | 26 | 92.6% |
| 6 | 546 | 494 | 52 | 90.5% |
| 7 | 276 | 234 | 42 | 84.8% |
| 8 | 314 | 235 | 79 | 74.8% |
| 9 | 250 | 82 | 168 | 32.8% |
| 10 | 13 | 0 | 13 | 0.0% |

### Run 5 — 2026-08-05 20:20, Part 1 micro-check: T=1.0 vs T=0.7

Before committing 4,000 problems of bucketing to T=1.0, the packet's own
micro-check (§6): the same 200 GSM8K screening problems 009 measured at T=0.7,
re-run at the pinned RL sampler. Cap 12,288 per finding 4.

`scripts/stagec_bucket.py --uids screen200_uids.json --pool gsm8k_test.jsonl
--temperature 1.0 --top_p 1.0 --max_tokens 12288`

| | 009 @ T=0.7, top-p 0.95 | **this run @ T=1.0, top-p 1.0** |
|---|---|---|
| strict Pass@1 | 64.25% | **48.00%** |
| strict pass@8 | 89.50% | **88.00%** |
| 0/8 | 10.5% | 12.0% |
| **mixed (1–7/8)** | **63.0%** | **84.0%** |
| 8/8 | 26.5% | 4.0% |
| cap-hit | 2.69% (8k cap) | **1.06%** (12k cap) |
| g-rate | 95.94% | 92.06% |
| duplicate groups | — | **0.00%** |

**Micro-check PASSES on the packet's stated criteria** — cap-hit 1.06% is far
under the ~5% ceiling, and the mixed-group fraction does not merely "hold", it
rises from 63% to **84%**. T=1.0/top-p 1.0 is confirmed for rollouts and
bucketing; no fallback to T=0.9 needed.

> **Finding 6 — raising the sampler to T=1.0 costs 16 points of Pass@1 and buys
> 21 points of mixed groups, which is the trade Stage C wants.** Strict Pass@1
> falls 64.25% → 48.00% while **pass@8 barely moves (89.50% → 88.00%)**: the
> capability is still there, the first sample is just drawn from a flatter
> distribution. The saturated 8/8 bucket collapses 26.5% → 4.0% and nearly all
> of it lands in the usable middle. Pass@1 *at the training temperature* is not
> a target metric — continuity evals run at T=0 and gate evals at T=0.7 — so the
> only thing this costs is a number nobody reports, and it buys a third more
> problems that can produce a gradient.
>
> The `dup groups 0.00%` line settles the packet's §11 seeding worry
> empirically: vLLM's ``n=8`` with a per-problem seed produced **zero**
> byte-identical group members in 200 groups, so no group silently carries zero
> within-group advantage.

### Run 6 — 2026-08-05 20:25, topology benchmark (`scripts/bench_trainer_step.py`)

The packet makes pipeline balance the pilot's first deliverable and warns
against pushing a losing topology uphill. Building a two-box pipeline before
checking spark can carry the trainer would have been exactly that, so the
trainer step was benchmarked on both boxes first: real model, real numerics
(fp32 master weights + fp32 AdamW per activity 007), seq 1024, micro-batch 1,
grad-accum 8, bf16 autocast forward, gradient checkpointing.

| box | result |
|---|---|
| **spark (GB10)** | **5.93 s/step**, 1,381 tok/s, **peak 34.7 GB** |
| **turing (RTX 5090)** | **CUDA OOM** at the AdamW step — 31.36 GiB capacity, 11.81 MiB free |

> **Finding 7 — the topology decision is forced by memory, not preference:
> turing physically cannot hold fp32 AdamW state for this model.** The OOM
> reproduces activity 007's arithmetic exactly ("fp32 weights + fp32 grads +
> fp32 Adam + SED shadow = 28.9 GiB and OOMs"), failing inside
> `_multi_tensor_adam` trying to allocate 48 MiB. Spark runs the same
> configuration at **34.7 GB peak against 128 GB of unified memory** — a third
> of the box. So P7 §4's "spark = trainer (fp32 AdamW — NOT 8-bit; memory is
> abundant and bitsandbytes-on-ARM is an unforced risk)" is not an aesthetic
> preference; it is the only placement where the packet's own numerics fit.
>
> **This raises the bar on the documented fallback.** "Full time-multiplex on
> turing" is not a pure scheduling change — it would also force 8-bit Adam
> moments (activity 007's Stage-B configuration, which works, but is a numerics
> change made for the wrong reason). The fallback therefore costs *numerics*,
> and the pilot's balance measurement has to be clearly against the split before
> taking it.

> **Finding 5 — the seen/unseen split is severely confounded with level, so a
> pooled memorization comparison would be uninterpretable.** Seen share runs
> 42.2% at L1, **85–95% across L3–L7**, then back down to 32.8% at L9 and 0% at
> L10. A pooled "seen vs unseen Pass@1" therefore compares a mostly-easy unseen
> set against a mostly-mid-difficulty seen set, and would show a *seen-side
> deficit* from difficulty alone — the opposite sign of the memorization effect
> it is meant to detect. **The memorization watch must be read within level, and
> L2/L10 (38 and 13 problems, one side near-empty) cannot support the comparison
> at all.** The packet's Part 0.4 already says "split by seen/unseen *and* by
> level"; this is why that qualifier is load-bearing rather than decorative.

### Run 3 — 2026-08-05, Part 0.2 + Part 1b: the reward instrument (mac, CPU only)

Built and unit-tested before any GPU time, per the packet's ordering.

**`whetstone/reward/strict.py`** — the strict grader. `verify.py` untouched
(CLAUDE.md invariant). Removes exactly the two leniencies finding 15 measured
and nothing else: the `endswith` suffix fallback, and `_strip_think`'s
whole-text fallback when `</think>` is absent. Normalization
(`_normalize`, `_try_numeric`) is **imported verbatim** from `verify.py` rather
than reimplemented, so the two graders can only ever differ in the two
documented ways; if the normalizer changes they move together. Returns both
verdicts in one object (`strict`, `as_scored`, `lenient_only`) so a caller
cannot report one and label it the other.

**`whetstone/reward/register_math.py`** — register-aware math normalization
(005 finding 14: the register writes `4√2` where the answer writes
`4\sqrt{2}`). Used only by the contradiction detector, to compare two of the
model's *own* strings. `values_agree` returns **`None` when undecidable** —
missing evidence is not evidence of contradiction, or the penalty becomes a tax
on symbolic answers.

**`whetstone/reward/stagec.py`** — the scalar reward. Additive, per §1b:

```
total    = r_acc + r_fmt
r_acc    = 1.0  iff  g == 1 AND strict-correct   else 0.0
r_fmt    = max(floor, r_struct − Σ penalties)
floor    = 0.10 iff well-formed (g == 1 and think ≥ 16 tokens) else 0.0
r_struct = 0.10 + [strict-gated] 0.15·exp(−max(0,T−B)/B) + [strict-gated] 0.10·band(A)
```

Pinned magnitudes: `W_FMT 0.10`, `W_LEN 0.15`, `W_BAND 0.10`,
`MIN_THINK_TOKENS 16`, contradiction `0.20`, register-leak `0.10`,
answer-repeat `0.05/rep` capped `0.10`, n-gram loop `0.10`. The last three are
v1 §4.6/§4.3 "at reduced weight" as the packet directs; contradiction and leak
are *raised* from v1's 0.05 — a 0.05 penalty sits below the length tail's own
range and would never change an ordering.

Measured ordering (`budget_B = 250`, the battery's fixtures):

| case | total | r_acc | r_fmt | penalties |
|---|---|---|---|---|
| correct + compact register + clean | **1.3500** | 1.00 | 0.3500 | 0.00 |
| correct + register leaked into answer | 1.2500 | 1.00 | 0.2500 | 0.10 |
| correct + verbose think | 1.2034 | 1.00 | 0.2034 | 0.00 |
| correct + think contradicting the answer | 1.1500 | 1.00 | 0.1500 | 0.20 |
| correct + EMPTY think | 1.0000 | 1.00 | 0.0000 | 0.00 |
| wrong + well-formed | 0.1000 | 0.00 | 0.1000 | 0.00 |
| loop / cap-hit (g=0) | **0.0000** | 0.00 | 0.0000 | — |

Invariants, asserted in code (`assert_invariants`, called at trainer import):
worst correct **1.00**, best wrong **0.10**, **margin 0.90** against I2's
required 0.30; max structural reward **0.35 < 1.0** so style can never outrank
accuracy.

**The empty-think guard works and is the most important row.** A correct
rollout with an empty think block scores 1.0000 against the compact one's
1.3500 — it loses 0.35, so RL is never rewarded for discovering
`<think>\n</think>`. The guard is needed because `parse_segments` scores empty
think as `g = 1` (correctly — it is not malformed), so the check has to live in
the reward. The length term is the other half: `exp(−max(0,T−B)/B)` is **flat at
1.0 below the budget**, so there is zero gradient toward shorter inside it, only
cost above.

**Tests: 57 green** (`tests/test_stagec_reward.py` 34, `tests/test_strict_grading.py`
23); full suite 81 passed / 1 skipped.

> **Finding 1 — the battery caught a real defect on its first run, in the
> contradiction detector.** The register writes conclusions as `⇒ 12 · 6 = 72`,
> and the detector compared the *whole* captured expression (`12 · 6 = 66`)
> against the boxed answer (`72`). Non-numeric on the left, so `values_agree`
> returned `None` (undecidable) and the penalty **silently never fired** on the
> exact shape the register actually emits. The one contradiction case that did
> fire in testing (`⇒ 6200`) had no `=` and hid the bug. Fixed by taking the
> right-hand side of the last `=`. This is the packet's "craft it like an
> instrument, then unit-test it like one" earning its keep on day one: a
> detector that is *present, configured, logged, and inert* is the failure mode
> that unit tests exist to catch, and it would have been invisible in training —
> the curve would simply have read zero.

> **Finding 2 — two of v1's kept detectors could not be reused as written; they
> encode the *Gemma* register.** v1 §4.7 register-leak matches `**Bold Header**`
> and `1. ` numbered chunks; v1 §4.3 repetition operates on `\n\n`-separated
> "chunks". Neither exists in the v2 symbolic register, which is line-oriented
> (`goal:`, `⇒`, `chk:`). Both were rewritten against this register's actual
> markers: leak fires on line-initial `goal:`/`chk:`/`sub:`/`let:`/`case:` plus
> the symbols `⇒`/`✗`, never on bare substrings (`case` is an English word in
> 10.2% of honest answers — 009 finding 1, asserted as a test); loop fires on
> ≥10 identical consecutive *lines* or ≥6 identical after digit-blanking, which
> is what catches `case 1:`…`case 713:`. "KEEP the penalty" meant "keep the
> intent"; the implementations did not survive the register change.

### Run 7 — 2026-08-05 20:35, Part 1: the pipeline, built and wired end to end

Three new modules plus two scripts, all unit-tested before GPU time:

| path | role |
|---|---|
| `whetstone/dapo.py` | the objective: clip-higher token-level policy loss, TEA, answer-KL, difficulty amplification, dynamic sampling, mask-partition assert |
| `whetstone/curriculum.py` | saturation-paced batch tilt (packet §8) |
| `whetstone/rollout_bus.py` | the trainer↔worker contract over `/data`, temp-then-rename everywhere |
| `scripts/stagec_rollout_worker.py` | turing side: offline vLLM, weight swap, staleness recorded |
| `scripts/stagec_train.py` | spark side: the loop |
| `scripts/stagec_dashboards.py` | the §7 panels, built alongside the trainer per CLAUDE.md |

**Deviation from P7 §4, attested — the π_0 anchor server is gone.** The packet
puts a frozen π_0 on `turing:8002` serving `prompt_logprobs`. That design exists
because activity 007 could not fit a second copy on turing's 32 GB beside the
optimizer state — it is why `precompute_pi0_cache.py` was written. Spark's
128 GB can hold it (3.4 GB bf16), so π_0 is now **resident on the trainer**.
This removes a network round-trip per step and makes the packet's own §11
gotcha — "reloading :8002 with student weights corrupts the KL anchor silently"
— *structurally impossible*: the anchor is loaded once from the original
checkpoint and never written.

**Why files rather than HTTP for rollouts.** The trainer needs **token ids**;
every mask in this project comes from `parse_segments` on ids because the
decoded-string round-trip does not preserve boundaries. vLLM's offline
`LLM.generate` returns `token_ids`; the OpenAI HTTP surface returns text and
string-keyed logprobs. One NFS round-trip per step is noise beside a 6-second
optimizer step.

Wiring verified end to end on **spark alone** (worker + trainer co-resident,
turing busy bucketing), 3 steps, K=4:

```
[train] step 1 | keep 1/2 | acc 0.50 | think 513 | ans 262 | H 1.755 |
        drift 1.49e-05 | wall 89s (gen 83 / train 5)
```

`theta_drift_rel` is non-zero, so the run is not a silent no-op — the failure
mode activity 007 mandates logging for, and which has caught this project twice.
Weight sync works: the worker detected `v1`, swapped in **55.2 s**, and reported
`staleness 0`. `logp_old_mismatch = 0.0128` nats between vLLM's logprobs and the
trainer's own forward, with `ratio_mean = 0.99995` — the two engines agree
closely enough that the importance ratio is sound.

> **Finding 8 — TEA was completely inert, and only a diagnostic caught it.**
> The wiring run logged `l_tea` and `think_entropy_mean` as **identical to eight
> decimal places on every step** (1.7547621131 vs 1.7547619045; diff 2e-07),
> with `cap_hit_frac = 0.0`. TEA's softmax weights were perfectly uniform: the
> term was adding mean think entropy and **selecting nothing**.
>
> Cause: `Cov_t = centered(log p_t)·centered(A_t)`, and **the advantage is
> constant within a rollout** — one scalar group advantage broadcast to its
> tokens. The trainer micro-batches one rollout at a time for memory, so
> `centered(A_t) ≡ 0`, `Cov ≡ 0`, uniform softmax. The design and the packet
> both say "**per batch** `Cov_t`"; the word *batch* is load-bearing and I had
> implemented it per micro-batch.
>
> Fix: `compute_tea_weights()` builds the selection over the whole batch
> **before any forward pass** — both inputs (`logp_old` from vLLM, the
> advantages) are constants w.r.t. θ, so the weights are a pure selector with no
> gradient through them — and each micro-batch is handed a constant slice via
> `tea_term()`. The policy loss and answer-KL now take batch-level denominators
> too, so micro-batch accumulation sums to *exactly* the batch objective rather
> than to a sequence-level average wearing a token-level costume. A
> **`uniformity`** metric (1.0 = selecting nothing) is now logged every step, and
> both failure directions are pinned as tests.
>
> This is the third instance in this packet of the same failure class — a
> component **present, configured, logged, and inert** (finding 1's contradiction
> detector; finding 2's Gemma-era detectors; now TEA). None would have thrown an
> error. TEA's curve would have moved, plausibly, forever. It is worth stating as
> a rule: **for every regularizer, log a statistic that goes to a known constant
> when the term is doing nothing**, and assert against that constant in a test.

**TEA verified live after the fix** (same wiring harness, 3 problems × K=4):

| metric | before (inert) | **after (batch-scoped)** |
|---|---|---|
| `uniformity` (1.0 = selecting nothing) | **1.00000** | **0.00000** |
| `cov` range | [0, 0] | **[−39.33, +19.26]** |
| `cap_hit_frac` | 0.0 | 0.0012 (7 of 5,792 tokens) |
| `weight_max` | = uniform | **0.017265** = the cap `c/|T|` = 100/5792 exactly |
| `L_TEA` vs plain mean think entropy | **identical** | **2.636 vs 1.463** |

`weight_max` landing exactly on `c/|T|` is the cap doing its job: the softmax
wants to concentrate further and is clipped. `L_TEA` is now a genuinely
different quantity from mean entropy, which is the minimum bar for the term
being alive.

Note what it selects: with `cov_min = −39.3` far larger in magnitude than
`cov_max = +19.3`, the top-covariance tokens are *low-confidence tokens in
penalized rollouts*, whose entropy the update would spend by pushing them down.
That is on-target for the formula as written (positive covariance in either
direction means the update sharpens that position), but it is **not** the
"confident-and-rewarded" reading one might assume, and `selected_entropy_mean >
think_entropy_mean` is the observable difference. Whether `τ_c = 0.7` shifts the
selection toward the other lobe is exactly what the pilot's sweep measures.

Dashboards verified on the wiring log (`scripts/stagec_dashboards.py`, turing —
matplotlib is turing-only). Eight panels render; the summary JSON carries the
pipeline-balance numbers the topology verdict needs.

> **Finding 9 — the empty-think attractor is live at step 1: 1 rollout in 12
> (8.3%) already emits an empty think block.** `empty_think_max = 0.083` on the
> very first wiring step, before any training. This is the failure the packet
> flagged from 009's no-floor run ("this model already knows how to emit
> `<think>\n</think>` + a correct answer, the parser scores it `g=1`, and
> 'correct + shorter is better' makes it the global optimum on every easy
> problem"). It is not a hypothetical to be watched for later — it is the
> policy's *current* behaviour at an 8% base rate, which is precisely why the
> guard had to be in the reward before step 1 rather than added on evidence.
> Under the shipped reward such a rollout scores **1.0000 against a compact
> correct rollout's 1.3500**, so the gradient pushes away from it. The rate is a
> first-class dashboard curve; if it climbs from here, the run stops and rollouts
> get read.

### Run 8 — 2026-08-05 20:30 →, Part 0.4: full Phase-1 bucketing (turing, running)

```
python scripts/stagec_bucket.py \
  --model /data/whetstone/ckpt/stageb/golden/round1/final \
  --uids /data/whetstone/corpora/stagea/subset_stagea_uids.json \
  --seen_uids /data/whetstone/corpora/stagea_golden/golden_faithfulness.jsonl \
  --out_dir /data/whetstone/runs/stagec_buckets/phase1_init \
  --temperature 1.0 --top_p 1.0 --max_tokens 12288
```

4,000 problems × K=8 = **32,000 rollouts**. Measured throughput ~3,340 output
tok/s → **~6 h**. This is the single long pole of the packet: it is both Phase
1's curriculum and the memorization control's baseline, and the design rule is
*curriculum-from-init, always* (v1's 2026-05-27 lesson), so it cannot be
shortened by reusing an earlier table.

**The pilot is chained to its completion** on both boxes (`~/chain_worker.sh` on
turing, `~/chain_train.sh` on spark), so no wall-clock is lost to the handoff:
each waits on `phase1_init/buckets*.json`, clears any orphaned `EngineCore`, and
launches. Pilot config: 60 steps (F4 needs 50), 4 problems/step × K=8, cap
12,288, τ_c 1.0, λ_TEA 0.05, λ_align 0.1, sync every 8, checkpoint every 20.

### Run 9 — 2026-08-05, Part 5: rescue driver built (`scripts/stagec_rescue.py`)

`select` picks the 0/K clientele and carries Stage A's conditioning fields
across so `gold+trace` problems keep their traces; `filter` applies **strict
verify + g=1 + in-register** and emits one trace per problem, then prints the
two delegated commands (GLM faithfulness, Stage-B assimilation at LR 5e-6 /
≤1 epoch / fresh EMA). Two attested deviations from design §5.2, both from
measurement rather than preference: **no G_spike threshold** (008 f10b — AUC
0.541 at level 9, a coin flip in exactly the band every 0/K problem lives in)
and **`gold+trace` conditioning wherever a trace exists** (008 f13 — gold-only
confabulates at 73.7% in the hard band). Runs at phase boundaries, so it does
not execute during the pilot.

> **Finding 10 — the bf16 weight export was casting the live fp32 policy in
> place, which would have erased the learning between every sync.** Caught in a
> pre-launch read of the trainer, before the pilot ran.
>
> `policy.to(torch.bfloat16).save_pretrained(...)` followed by
> `policy.to(torch.float32)` looks like a round trip. It is not:
> `nn.Module.to()` mutates parameters **in place**, and casting back cannot
> restore mantissa bits that are already gone. Measured on a toy module:
>
> | | |θ| before → after | verdict |
> |---|---|---|
> | in-place cast round trip | 6.5958328247 → 6.595**9014893** | **changed** |
> | bf16 *copy* of the state dict | 6.5550575256 → 6.5550575256 | unchanged |
>
> The magnitude is what makes it fatal rather than untidy: the round trip
> introduces a mean per-weight perturbation of **8.09e-05**, while an Adam step
> at LR 1e-6 moves a weight by ~1e-6. The export noise is **~81× the size of the
> update**, so every sync (every 8 optimizer steps) would have thrown away the
> learning since the previous one — and `theta_drift_rel` would have kept
> reporting non-zero drift the whole time, because the weights *were* moving,
> just not in the direction of the gradient.
>
> This is **activity 007's finding re-entering through a different door**. 007
> established fp32 master weights because "at LR 1e-5 a bf16 weight update is
> 12× below the format's quantum and rounds to zero silently", and the trainer
> honours that in the optimizer — then handed the same precision back in the
> serialization path. The fix builds a bf16 **copy** of the state dict and
> asserts `|θ|` is bit-identical across the export; the assert is the part worth
> keeping, because the next person to touch this code will reach for `.to()`
> again.

### Run 8 (cont.) — 2026-08-06 02:52, Part 0.4 COMPLETE: the Phase-1 bucket table

4,000 problems × K=8 = 32,000 rollouts in **381.5 min (6.4 h)** on turing.
Artifacts: `/data/whetstone/runs/stagec_buckets/phase1_init/{buckets.jsonl,buckets_summary.json}`.

| | value |
|---|---|
| 0/8 | **14.5%** (580 problems — rescue's clientele) |
| **mixed 1–7/8** | **79.6%** (3,184 problems — the curriculum) |
| 8/8 | 5.9% |
| strict Pass@1 | 46.35% (as-scored 47.95%) |
| strict pass@8 | **85.45%** |
| g-rate | 92.67% |
| cap-hit | **0.78%** (at 12,288 — finding 4 vindicated) |
| lenient-only | 1.78% |
| think median / answer median | 978 / 403 |
| duplicate groups | **0.00%** |

**79.6% mixed groups.** Against the original checkpoint's 8.0% on GSM8K, that is
**~10× the usable DAPO signal**, and the headroom is **39.1 points** (46.35 →
85.45). The substrate claim from 009 finding 16 holds on the real training pool,
not just the easy validation tier.

> **Finding 11 — the student is measurably better on problems it saw in Stage-B
> SFT, by +5.32 points, and the pooled comparison hides 84% of that.** This is
> the memorization control's *pre-RL baseline*, and it is the exact
> Simpson's-paradox trap finding 5 predicted.
>
> | | strict Pass@1 delta (seen − unseen) |
> |---|---|
> | **pooled over all 4,000** | **+0.85 pts** — looks like nothing |
> | **inverse-variance weighted within level** | **+5.32 pts (SE 1.86, z = +2.87)** |
>
> | level | n seen | seen | n unseen | unseen | delta |
> |---|---|---|---|---|---|
> | 1 | 844 | 51.66% | 1,156 | 47.49% | **+4.17** |
> | 5 | 327 | 40.37% | 26 | 30.77% | **+9.60** |
> | 6 | 494 | 39.68% | 52 | 34.62% | **+5.06** |
> | 7 | 234 | 48.72% | 42 | 40.48% | **+8.24** |
> | 8 | 235 | 44.68% | 79 | 31.65% | **+13.04** |
> | 9 | 82 | 51.22% | 168 | 48.81% | **+2.41** |
>
> **6 of 6 levels favour seen** (sign test p = 0.031); the weighted effect is
> **6.3× the pooled one**. Levels 2, 3, 4 and 10 have fewer than 25 problems on
> one side and are excluded rather than quoted.
>
> The cause is the one predicted: unseen problems are concentrated in the *easy*
> L1 tier (1,156 of 1,586) while seen problems sit in the mid-difficulty band, so
> pooling lets an easy-problem majority on the unseen side cancel a real
> seen-side advantage almost exactly.
>
> **Binding on the pilot and on Phase 1:** the memorization watch compares
> against **+5.32 pts, not against zero**, and it is read *within level*. A
> post-RL within-level delta of, say, +6 pts is roughly no change; the alarm
> condition is the delta widening materially beyond this baseline, which would
> say RL is rewarding recall rather than derivation. Is +5.32 "dramatically
> easier", the thing the packet says to flag loudly before training? It is a real
> and statistically clear effect, but it is a fifth of the 24-point Pass@1 hole
> Stage B left, so it does not by itself argue for dropping the seen problems —
> it argues for measuring the delta the right way, which now has a number.

### Run 10 — 2026-08-06 03:00 →, Part 2: THE PILOT (turing generates, spark trains)

Launched automatically off the chained scripts the moment Part 0.4's bucket
table landed; the startup model-match check fired correctly —
`worker already on .../round1/final (v0) — skipping the redundant initial
publish` — saving a 55 s engine rebuild against an identical checkpoint.

Config: 60 steps, 4 problems/step × K=8, T=1.0/top-p 1.0, cap 12,288, τ_c 1.0,
λ_TEA 0.05, λ_align 0.1, LR 1e-6, sync every 8, checkpoint every 20.
Curriculum: **3,184 mixed problems** (bands: 1,366 high / 972 mid / 846 low).

Step 1: `keep 4/4 | acc 0.53 | think 1176 | ans 620 | H 1.055 | teaU 0.000 |
drift 1.45e-05 | wall 120s (gen 60 / train 60)`, `B_0` measured at 1,026 tokens.
All four groups survived dynamic sampling, `theta_drift_rel` is non-zero, and
`teaU 0.000` confirms TEA is selecting rather than idling.

> **Finding 12 — the two boxes are almost perfectly balanced (gen 60 s / train
> 60 s), and the loop as written makes them take turns, so half the wall-clock
> is one box idling.** The packet's topology question was "does spark's trainer
> step exceed ~2× turing's generation?" — the answer is **no, the ratio is
> 1.0**, so the split topology is correct and the turing-multiplex fallback is
> not needed. But the *scheduling* leaves 50% on the table: the trainer posts a
> request, blocks on the response, then trains while turing sits idle.
>
> A balanced 1:1 split is the best possible case for prefetching, which is why
> the fix is worth the complexity: queue step N+1's rollouts *before* spending
> the ~60 s on step N's gradient. Implemented as `--prefetch`, **off by default
> so this pilot's balance measurement stands unaltered**, and on for Phase 1,
> where it should take the step from ~120 s to ~65 s. The costs are a one-step
> lag in the curriculum (step N+1's batch is drawn before step N's `observe()`
> lands) and up to one extra sync of policy staleness — both inside what DAPO's
> clipping is built to absorb, and both logged.
>
> Note this also answers the packet's plan to have "π_0 anchor scoring ride
> turing because turing idles while spark trains": with the anchor moved onto
> the trainer, the *right* thing to give turing during that window is the next
> batch of rollouts, not anchor scoring.

> **Finding 13 — at τ_c = 1.0 the covariance softmax is so peaked that TEA
> protects ~3 tokens out of ~50,000, and the packet's sweep explores the wrong
> direction.** First three pilot steps:
>
> | step | policy loss | λ·L_TEA | tea/policy | think tokens | **tokens at the cap** | weight mass kept |
> |---|---|---|---|---|---|---|
> | 1 | +0.0875 | +0.1367 | 1.56 | 57,124 | **4** | — |
> | 2 | +0.1195 | +0.1048 | 0.88 | 42,801 | **5** | **1.25%** |
> | 3 | +0.0601 | +0.1070 | 1.78 | 59,516 | **1** | — |
>
> The covariance range is about **84 nats wide** (`cov_min −61.0`, `cov_max
> +22.8`). Dividing by τ_c = 1.0 leaves an 84-wide logit range, so
> `softmax(Cov/τ_c)` is dominated by its maximum and everything past the top
> handful is numerically zero — `uniformity` reads **1.5e-34**. The cap then
> clips those few to `c/|T|` and **98.75% of the softmax mass is thrown away**
> (`weight_sum` 0.0125).
>
> Two things follow. First, the **`c = 100` cap is inert as designed**: a cap at
> 100× uniform is meant to stop any one token dominating a field of ~100+
> participants, but here only 1–5 tokens clear the noise floor at all, so `c`
> is not the binding constraint the design intends it to be. Second — and this
> is the actionable part — **the packet's τ_c ∈ {0.7, 1.0} sweep moves the wrong
> way.** Smaller τ_c *sharpens* the softmax, so τ_c = 0.7 will concentrate on
> even fewer tokens. Getting a meaningful population under protection needs τ_c
> *larger* by roughly an order of magnitude, or the covariance standardized
> before the softmax.
>
> Note the ratio column is about the loss **value**, not its influence: with the
> weights concentrated on ~3 tokens the gradient reaching any token is ~λ_TEA/3,
> so TEA is not overpowering the policy — it is applying a visible-looking number
> to almost nothing. This is the *third* variant of "present, configured, logged,
> and nearly inert" in this packet, and the reason it was caught this time in
> three steps rather than never is that `uniformity`, `cap_hit_frac` and
> `weight_sum` were all put on the dashboard after finding 8.
>
> **Not changed mid-pilot** — the pilot is the measurement, and 003/010's τ_c
> concern was always scheduled for the sweep. The recommendation for Phase 1 is
> to sweep **τ_c ∈ {1.0, 10, 30}** (or standardize `Cov` and keep τ_c ≈ 1), and
> to report tokens-under-protection alongside it, since that is the number that
> says whether the term is doing anything.

> **Finding 14 — at 4 problems/step the per-step length medians are noise, and
> reading a trend from them would be a mistake.** Over the first 10 pilot steps
> `think_median` ranges **308 → 3,404** with a coefficient of variation of
> **0.82**, while `acc_rate` sits at CV **0.10**. The swing tracks *which
> problems were drawn*, not the policy: correlation between a step's mean batch
> `p_hat` and its think median is **−0.355** (easier batches, shorter thinks),
> and the curriculum deliberately mixes bands every step.
>
> Two consequences. **(1)** This is why F4's second clause is defined on a
> *fixed* 200-problem screen rather than on the training curves — the screen
> holds the problem set constant, so a change in it is a change in the policy.
> Any statement of the form "think length is falling" taken from the training
> log at this batch size is unsupported. **(2)** For Phase 1, **4 problems/step
> is too few** for a low-variance length signal. With `--prefetch` taking the
> step to ~65 s, going to 8 problems/step costs ~110 s/step and roughly halves
> the variance; that is the recommended Phase-1 setting, chosen from this
> measurement rather than from the packet's silence on the parameter.
>
> **Reward integrity over the same 10 steps is clean on every axis the packet
> names:** `empty_think` **0.0000 on every step** (the guard is holding — recall
> finding 9 measured an 8.3% base rate before training), `lenient_only` max
> 0.0625 and not widening, loop penalty max 0.0042, `g_rate` 0.875–1.000, group
> drop rate 0–0.25, and `theta_drift_rel` **monotonically increasing**
> 1.45e-05 → 5.76e-05.

### Run 11 — 2026-08-06, per-checkpoint rollout investigation (v1 §7.6–7.7)

F4's first clause is "no critical rollout-investigation flag", and 009 finding 11
is the standing law that only *generative* inspection catches death. 672 rollouts
over the pilot's first 21 steps were scanned for the named rot patterns and then
**read verbatim**.

| pattern | rate |
|---|---|
| empty think | **0.00%** |
| exact line-loop (≥10 identical consecutive) | **0.00%** |
| `case N:` enumeration ≥20 deep | **0.00%** |
| runaway `chk:` chain (≥15 lines) | **0.00%** |
| template line-loop (≥6 digit-blanked identical) | 0.30% |
| `missing_think_open` | 0.15% |
| **`missing_think_close`** | **6.10%** |
| register leakage in the answer | 1.19% → **0.00% after finding 15** |

> **Finding 15 — the register-leak penalty was firing on correct answers for
> using standard mathematical notation: 9 of 9 detections were false.** Every
> detection came from the bare-symbol rule matching `⇒` *mid-sentence in English
> prose*; **zero** came from the line-initial marker rule. Verbatim:
> *"If a polynomial has a root ⇒ it has a linear factor"*, *"Total: $32 + 18 +
> 98 = \$138 ⇒ this shares 72 cents"*. Qwen3 writes `⇒` as ordinary notation in
> mathematical English, and the answer segment is *supposed* to be mathematical
> English. The register, by contrast, writes conclusions **line-initially**
> (`⇒ 12 · 6 = 72`), so requiring line-initial position is simultaneously more
> faithful to the register and free of these false positives. Fixed, with the
> four verbatim shapes added as regression tests and the battery re-run green
> per the packet's "a reward change of any kind re-runs the battery first". The
> running pilot is unaffected (it imported the module at start), so the change
> takes effect from Phase 1.
>
> Worth noting *why* 009 did not see this: it measured `answer_leak_rate`
> 0.0013 on **GSM8K**, where answers are arithmetic prose. This pool is half
> DeepMath, where answers are real mathematics and `⇒` is native vocabulary.

> **Finding 16 — the loop tail is EXTINCT, and the malformation that replaced it
> is a completely different failure: the model finishes its work and never emits
> `</think>`.** Cap-hit `g=0` runs **0.89% → 0.00% → 0.00% → 0.00%** across the
> pilot's step windows — activity 009 finding 14's runaway class, which consumed
> 77.7% of the decode budget and made full evals take 13–17 h, does not appear at
> the 12,288 cap on this policy.
>
> What remains is `missing_think_close` at a **flat ~5–6.5%**, and it is not
> rumination. Of 44 such rollouts, **41 ended with `finish_reason: "stop"`** (not
> `length`), median total length **1,106 tokens**, and **82% contain a
> `\boxed{}` answer**. Read verbatim, they are *complete, correct-looking
> solutions*:
>
> ```
> <think>
> goal: calculate total extra recess by considering each student's category.
> ...
> chk: all categories added correctly and final sum is positive; ...
> The students would get a total of 27 minutes of additional recess.
> **Final Answer**
> \boxed{27}
> ```
>
> The model reasons in the register, transitions into a normal write-up, delivers
> a boxed answer — and never closes the block. One missing token.
>
> **The reward scores these 0.0000, below a wrong-but-well-formed rollout's
> 0.1000, and that is deliberate.** An unclosed rollout has no answer segment at
> all, so the answer-KL anchor and the SCA length band have nothing to act on —
> crediting it would train a mode the loss cannot regulate, and the whole
> segment-routing architecture depends on the boundary existing. The packet's
> `r_fmt = 0.10` for well-formed-but-wrong is precisely the gradient that teaches
> closing, and it is already in place.
>
> **But the rate is flat, not falling**, over 28 steps. Recorded as an open item
> rather than an F4 critical flag: it is a pre-existing property of the Stage-B
> student (the bucketing measured the same thing — `g` 92.67% against a 0.78%
> cap-hit rate), not RL-induced rot, and v1 §7.6–7.7's rot patterns are about
> *degeneration*. If it has not moved by the end of Phase 1, the cure is a format
> intervention in the card (P3a's territory), not more reward pressure —
> 6% of the gradient currently says "that reasoning was worthless" about
> reasoning that was fine.

> **Finding 17 — entropy is rising 46% over 30 steps while accuracy stays flat,
> and TEA's loss value runs 1.3–5.0× the policy loss.** Ten-step windows:
>
> | steps | acc | think med | **H think** | L_TEA | answer KL | clip-low | θ drift | **λ·L_TEA / policy** |
> |---|---|---|---|---|---|---|---|---|
> | 1–10 | 0.534 | 1,152 | **1.048** | 1.938 | 0.175 | 0.00011 | 3.9e-05 | **1.33** |
> | 11–20 | 0.585 | 821 | **1.312** | 1.823 | 0.204 | 0.00007 | 7.3e-05 | **4.96** |
> | 21–30 | 0.560 | 1,071 | **1.594** | 1.976 | 0.188 | 0.00006 | 9.9e-05 | **2.34** |
>
> `corr(step, H) = +0.695`; first-5-step mean 1.090 → last-5 1.595. Training-rollout
> accuracy is flat (0.535 → 0.506, inside the CV-0.10 noise floor established in
> finding 14).
>
> The packet's F4 entropy criterion is "**no collapse** below the Part-0 card",
> and there is none — the card was 0.620 and the run is at 1.6. But a 46% *rise*
> with flat accuracy is not automatically health either: it is what a policy
> diffusing rather than sharpening looks like, and RL normally sharpens.
>
> The likely mechanism ties back to finding 13. Clipping is essentially inactive
> (`clip_frac_low` ~6e-05, so every importance ratio sits inside [0.8, 1.28]) and
> θ drift is only 1.1e-04 relative, i.e. **the policy is barely moving under the
> policy-gradient term** — while TEA, the one term explicitly pushing entropy
> *up*, carries 1.3–5.0× the policy loss's magnitude and concentrates its
> gradient on a handful of tokens at ~500× the policy's per-token coefficient.
> That combination raises entropy without a compensating accuracy signal.
>
> **Not acted on mid-pilot.** The binding adjudicator is F4 clause 2's fixed
> 200-problem screen: if a checkpoint Pareto-dominates the init, the run is
> learning and the entropy rise is TEA doing its job; if no checkpoint beats the
> init while entropy climbs, the run is diffusing and **λ_TEA must come down**
> (it is on the design's own run-1 sweep list) alongside finding 13's τ_c
> correction. Recording the prediction before the measurement, so the reading
> afterwards cannot be fitted to the result.

### Run 12 — 2026-08-06, the τ_c sweep, run **offline** on the pilot's own rollouts

TEA's weights are `softmax(Cov/τ_c)` capped at `c/|T|`, and `Cov` is built from
`logp_old` and the advantages — **both constants with respect to θ**, both stored
in the pilot's response files. So the packet's τ_c sweep needs no second
training run and no GPU: it can be recomputed exactly, on the same data the live
run selected from. Pooled over 20 steps, **894,860 think tokens**, `Cov` range
[−60.7, +45.7].

| τ_c | tokens at the cap | **effective tokens protected** | softmax mass kept | uniformity |
|---|---|---|---|---|
| **0.7** (packet) | 32 | **44.6** | 0.0044 | 0 |
| **1.0** (packet, what the pilot ran) | 58 | **91.1** | 0.0086 | 0 |
| 3.0 | **383** | **1,251** | 0.086 | 2e-13 |
| 10.0 | 0 | 587,943 | 1.0002 | 2.4e-05 |
| 30.0 | 0 | 887,349 | 1.0001 | 2.9e-02 |
| 100.0 | 0 | 894,290 | 1.0001 | 3.4e-01 |

("Effective tokens" is the participation ratio `1/Σp²` — how many tokens
actually share the protection. Uniform would be 894,860.)

> **Finding 18 — TEA has a narrow usable window, the packet's sweep sits
> entirely below it, and above it the term is inert.** Between τ_c = 3 and
> τ_c = 10 the behaviour flips completely: at **3.0** the cap binds on 383 tokens
> and ~1,251 share the protection, which is the regime `c = 100` was designed
> for; by **10.0** the cap never binds at all, the mass is uniform, and `L_TEA`
> degenerates to plain mean entropy — the exact inert case finding 8 diagnosed.
> Below that window, the packet's **τ_c ∈ {0.7, 1.0} spans 44.6 → 91.1 effective
> tokens out of 894,860 — 0.005% to 0.010% of the batch**, two points 2× apart
> inside a regime where the term touches almost nothing. That sweep could not
> have distinguished anything. **Recommended for Phase 1: τ_c = 3.0**, with
> "effective tokens protected" reported beside it, because that is the statistic
> that says whether the knob is in the window at all.

> **Finding 19 — this refutes my own explanation in finding 17.** I hypothesised
> the 46% entropy rise came from TEA over-protecting, on the grounds that its
> loss *value* ran 1.3–5.0× the policy loss. The sweep kills that: at the
> pilot's τ_c = 1.0, TEA's gradient reaches **91 of 894,860 tokens (0.01%)**. A
> term touching one ten-thousandth of the batch cannot drive a global entropy
> rise, however large its scalar value looks.
>
> The rise must therefore come from the policy-gradient side — most plausibly
> **DAPO's clip-higher**, whose asymmetric bound (ε_high 0.28 > ε_low 0.20)
> exists precisely to leave low-probability tokens room to gain probability, and
> which the design calls "the entropy-preserving half of the algorithm". That is
> the term working as intended, not a defect. The correction matters practically:
> **lowering λ_TEA would not have fixed the entropy trend**, and if the screen
> had come back flat I would have turned the wrong knob. Recorded as a lesson
> about the magnitude/coverage distinction — a loss term's scalar value says
> nothing about how many parameters it moves.

> **Finding 20 — the pilot's apparent accuracy decline is a sampling artifact,
> and the answer segment is NOT collapsing.** Two checks that would each have
> been misread from the curves alone.
>
> *Accuracy.* Ten-step window means fall 0.534 → 0.585 → 0.560 → 0.530 →
> **0.491**, which reads as decay. It is not: `corr(batch mean p̂, acc_rate) =
> **+0.770**` and the mean absolute gap between a step's accuracy and its
> batch's own mean p̂ is **0.050**. Accuracy is tracking *which problems were
> drawn*. The "trend" `corr(step, acc) = −0.204` is fully accounted for by
> `corr(step, batch mean p̂) = −0.422` — sampling drift, not policy decay. And it
> is not the curriculum hardening by design either: after 50 steps only **200 of
> 3,184 problems (6.3%) have been drawn at all**, **14 retired**, and **0 tilt
> shifts**, so the pool has barely moved.
>
> *Answer length — the round-2 collapse watch.* Window means 379 → 294 → 352 →
> 333 → 354, `corr(step, answer_median) = **−0.008**`, first-5 330 → last-5 352,
> against a **288** baseline. The single 108 at step 50 is the run's minimum and
> one step wide. Activity 009's round-2 collapse (288 → **19**, −93%) is **not**
> recurring: the SCA answer band in the scalar reward and the forward-KL to π_0
> on answer tokens (`kl.mean` stable at 0.175 → 0.213) are holding the segment
> where they were designed to.
>
> Together with finding 14 this is now a general statement about this run:
> **at 4 problems/step, neither accuracy nor either length median can support a
> trend claim.** Only the fixed 200-problem screen can, which is exactly what F4
> clause 2 is defined on.

### Run 13 — 2026-08-06 04:40, pilot complete: 60/60 steps, clean exit

Checkpoints at steps 20/40/60. Median wall 105 s/step
(**rollout 60.1 s / trainer 41.6 s / scoring 0.01 s / sync ~0**),
`trainer_over_rollout` **0.69**.

> **Finding 21 — the format degrades badly in the second half: `g_rate`
> collapses 0.935 → 0.658, and it is one failure mode, not several.**
>
> | steps | `g_rate` | `missing_think_close` | lenient-only | word stutter | H think |
> |---|---|---|---|---|---|
> | 1–10 | **0.9354** | 5.6% | 0.028 | 0.0027 | 1.048 |
> | 11–20 | 0.9292 | — | 0.047 | 0.0024 | 1.312 |
> | 21–30 | 0.8604 | — | 0.092 | 0.0017 | 1.594 |
> | 31–40 | 0.8781 | — | 0.087 | 0.0040 | 1.705 |
> | 41–50 | 0.8448 | — | 0.058 | 0.0079 | 2.081 |
> | **51–60** | **0.6583** | **35.6%** | **0.140** | **0.0120** | **3.176** |
>
> The malformation is the *same* mode throughout — `missing_think_close` with
> `finish_reason: "stop"` (109 of 114 in the last window, median 1,187 tokens,
> only 5 at the cap). It simply becomes **6.4× more frequent**. This is finding
> 16's early-stop failure, amplified by RL rather than cured by it.
>
> **The widening strict-vs-as-scored gap is a consequence, not reward hacking.**
> The packet reads a widening gap as "the policy found a grading hole". Checked
> directly: of 45 lenient-only rollouts in the last window, **42 (93%) are
> `g=0`** — the answer is sitting in an unclosed scratchpad — and only **3 (7%)**
> are genuine `g=1` grading-hole exploits. The reward is not being farmed; the
> format is failing, and `verify.py`'s leniency is picking the wreckage up. Worth
> stating because the two diagnoses have opposite fixes.
>
> **The mechanism is coherent and single-cause.** Entropy climbs monotonically
> across the same windows (1.048 → **3.176**, now **5.1× the pre-RL card of
> 0.620**). A less decisive policy misses a single low-entropy token — `</think>`
> — more often; activity 007 already measured that boundary as *the* fragile one
> ("inoculation degrades the native `</think>` boundary: entropy 7.6e-05 →
> 0.045 → ~1.8–2.0"). Word stutter rising 4.4× (009 finding 19 calls it "the
> loop in embryo") is the same diffusion showing up lexically. Per finding 19 the
> entropy rise is **not** TEA — which touched 91 of 894,860 tokens — so the
> unchecked driver is clip-higher, working as designed with nothing calibrated
> to oppose it.
>
> **This is a critical rollout-investigation flag under v1 §7.6–7.7**: a named
> structural failure that is *worsening monotonically* rather than being
> extinguished. **F4 clause 1 fails on it.**

---

## Status at 2026-08-05 21:00

**Everything the packet gates the pilot on is done, verified, and committed. The
pilot itself is blocked on the ~6 h bucketing run and is chained to start the
moment it lands.**

| Definition-of-done item | state |
|---|---|
| Part 0.1 entropy card (F3c debt + TEA baseline) | **done — F3c PASSES**, median 10.1× baseline |
| Part 0.2 strict wrapper, unit-tested on finding-15 cases | **done**, 23 tests |
| Part 0.3 `spark:8101` stopped | **done** (already down; :8100 also down, :8000 llama-swap untouched) |
| Part 0.4 Phase-1 buckets, seen/unseen split | pool composition **done**; K=8 table **running (~6 h)** |
| Part 1b reward battery green before pilot step 1 | **done**, 34 tests + I1–I3 property checks |
| Part 1 DAPO loop, segment routing, TEA | **built and wired end to end**, 131 tests |
| Part 2 pilot + F4 verdict | **chained, not yet run** |
| Parts 3–4 Phase 1 / Phase 2 to Pareto endpoint | not started (multi-day compute) |
| Part 5 rescue | driver **built**; runs at phase boundaries |
| Dashboards | **built and verified** |

**Test totals: 131 passing** (`tests/test_stagec_reward.py` 34,
`test_strict_grading.py` 23, `test_dapo.py` 30, `test_curriculum.py` 13, plus
the pre-existing segment/SED suites).

### What the pilot must still deliver (packet §7)

1. topology verdict from the wall-clock split — the wiring run's split
   (rollout 90 s / trainer 10 s) is **spark-generating** and not representative;
   turing's numbers are the real test
2. **F4 PASS/FAIL in bold** — 50 steps, no critical rollout flag, ≥1 checkpoint
   Pareto-dominating the init on the 200-screen
3. reward-integrity check (loop rate → 0, `lenient_only` not widening)
4. entropy trajectory + the τ_c ∈ {0.7, 1.0} sweep
5. memorization read (seen-vs-unseen **within level**, per finding 5) + GLM
   spot-check
6. continuity evals

### Substitution recorded in advance

The packet asks for continuity evals *every 10 steps* during the pilot. Running
them inline stalls the pipeline (each needs the GPU that is generating
rollouts). Substituted: **per-step degeneracy curves from the training rollouts
themselves** (`empty_think`, `lenient_only`, loop penalty, word-stutter, g-rate
— all already logged every step, and a strictly faster collapse signal than a
T=0 eval), plus checkpoints every 20 steps evaluated **after** the pilot on the
200-screen at the full protocol. The trend line is preserved; only its cadence
and its position in the pipeline change.
