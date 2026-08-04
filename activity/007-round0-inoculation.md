# 007 — P4: Round-0 scorer inoculation and the F1 band-existence gate

- **Packet:** [packets/P4-round0-inoculation.md](packets/P4-round0-inoculation.md)
- **Status:** done
- **Machine(s):** mac (code), turing (all scoring, training, meter tests)
- **Code commit(s):** `4adbabd` → `39f2e73`
- **Started / finished:** 2026-08-03 → 2026-08-03

## Goal

Calibrate the scorer so compact-register tokens read as a low hum while genuine
reasoning leaps still spike, and answer **F1: does that band exist?** If
register-hum and leap-spike are inseparable the whole design pivots to the
prefix/LoRA scorer arm. The product of this packet is a trustworthy measuring
instrument, not a capable model. On PASS, run activity 006's binding
`G_spike` × branch-retention check, which gates P5.

---

## New code

| path | role |
|---|---|
| `whetstone/round0.py` | shared sequence construction (packet §4), card §2 whitelist, `TokenScores`/d_t contract, `g_spike`, percentile |
| `whetstone/sed.py` | SED kernel (CurioSFT), **shared verbatim with Stage B**; `row_logits` helper |
| `whetstone/round0_eval.py` | S1/S2/S3 + hum, the π_0 cache reader, `assert_alignment` |
| `tests/test_sed.py` | the four §12.4 bug-pinning unit tests (+2 extra) |
| `scripts/build_register_tokenset.py` | Part 1 — the R token set |
| `scripts/precompute_pi0_cache.py` | frozen-π_0 reference values (see deviation 2) |
| `scripts/inoculate_scorer.py` | Part 3 — the inoculation trainer + four curves |
| `scripts/build_corrupted_probe.py` | Part 4 — corrupted/clean twin pairs |
| `scripts/meter_tests.py` | Part 4 — the three meter tests + F1 verdict |
| `scripts/gspike_branch_check.py` | Part 5 — G_spike × branch retention |

Environment note: turing's checkout is at **`~/workspace/whetstone`**, not
`~/git/whetstone` as CLAUDE.md's environment section implies (spark's *is* at
`~/git/whetstone`). `pytest` and `bitsandbytes` were added to `pyproject.toml`.

---

## Deviations from the packet, with reasons

### 1. The whitelist admits only single-token variants (Part 1)

The packet says `R = R_stats ∪ whitelist` with each card §2 string "tokenized
bare AND space-prefixed, **all piece-ids included**". That instruction assumes
each whitelist string is one token. Three of card §2's entry classes are not:

| card §2 entry | tokenizes to | what the pieces are |
|---|---|---|
| `1.` … `9.` | `[16..24]`, `[13]`, `[220]` | bare digits, `.`, space |
| ` 1.` … ` 9.` | `[220, 16..24, 13]` | same |
| bare `✗` | `[245, 25521]` | undecodable byte fragments |

Expanding them puts **26.9% of all 195,898 think tokens** into R — the CE mask
would be mostly digits and whitespace:

| id | surface | mean surprisal | count |
|---|---|---|---|
| 16 | `'1'` | 1.126 | 10,244 |
| 220 | `' '` | 0.568 | 10,873 |
| 13 | `'.'` | 0.147 | 6,168 |
| 17–24 | `'2'`–`'9'` | 0.319–0.519 | 25,471 |
| 245 / 25521 | byte fragments | 1.702 / 7.461 | 27 |

Two reasons this is not a nitpick:

1. **Those types are not style vocabulary by the packet's own rule.** At
   0.15–1.13 nats they sit *below* the p75 threshold of 1.7803, so `R_stats`
   would never have selected them. They enter only through piece-expansion.
2. **It would sabotage the decisive test.** Meter test (c)'s value-substitution
   corruption edits a *digit* token. Training CE on digit types in register
   context is training away the surprise probe (c) has to detect — the scorer
   would pass (a) and (b) and fail the one test that invalidates it, or worse,
   pass all three while blind to exactly one corruption class.

The byte fragments are worse than useless: they occur 27 times in the corpus
and **`✗` occurs zero times**, so every one of those occurrences is a piece of
some *other* multi-byte character.

So a variant contributes its id only when it is a single token. The dropped
variants are printed with their piece-ids and corpus coverage, rather than
disappearing silently — the failure class activity 005 findings 3 and 5 both
belong to. `--whitelist-all-pieces` restores the packet's literal behaviour.

### 2. π_0 is a precomputed cache, not a resident model

The packet budgets "θ 3.4 + grads 3.4 + AdamW ~13.6 + φ 3.4 + π_0 frozen eval
copy 3.4 ≈ 27 GB". Two things are wrong with it (see deviation 3), and the
frozen π_0 copy does not fit. Its stated fallback — move the π_0-side metrics to
the spark server with `--max-logprobs 512` — would ship ~60k × 512 logprob
payloads over HTTP every eval.

Since π_0 is *frozen*, there is a strictly better third option: compute its
contribution once. `scripts/precompute_pi0_cache.py` stores, at a fixed seeded
sample of control think positions, π_0's top-512 (ids, logprobs), its
actual-token logprob and its top-512 entropy. 60,000 positions, 247 MB, 39 s.
S2 and meter test (b) become exactly reproducible, and no π_0 copy sits on the
GPU.

**It also validates itself.** At step 0 the trainee *is* π_0, so the eval must
reproduce the cache: measured S2 KL = 0.00094 and Δlogp = +0.00057, both ≈ 0.
That residual is the bf16 numerical floor between the two forward paths and is
**S2's noise floor** — worth knowing, because κ_max is set in those units.

Control positions are subsampled (300/trace, seeded) rather than all ~6,000:
the forward still runs over the whole sequence, only the scored positions are
sampled, and the sample is identical at every eval and every checkpoint.

### 3. fp32 master weights + 8-bit Adam moments

The packet's budget (θ 3.4 GB + AdamW ~13.6 GB) describes bf16 weights with
fp32 optimizer moments. That combination is not expressible in torch, and it
would not work anyway: **at LR 1e-5 an Adam update is ~1e-5, while bf16's
quantum at a typical weight magnitude of 0.02 is ~1.2e-4.** Every update would
round to zero — the run would report a falling loss and change nothing.

With fp32 weights, full AdamW is 6.4 + 6.4 + 12.8 GiB resident plus the 3.2 GiB
SED shadow = **28.9 GiB**, which OOMs on activations (measured: died at 30.51
GiB allocated). Quantizing only the *moments* keeps the fp32 master weights —
the part that matters — and brings the run to 19.2 GiB. Verified to produce the
same 1e-5 step magnitude as full AdamW.

`theta_drift_rel` is logged at every eval so a silent no-op can never recur.
This mattered immediately: the first smoke run reported **exactly zero drift**,
which turned out to be `get_cosine_schedule_with_warmup` setting the LR
multiplier to `0/20 = 0` at construction, so the *first* optimizer step always
runs at LR 0. Harmless in a 120-step run, fatal in the 1-step smoke test, and
invisible without the drift metric.

### 4. Loss is computed on think rows only

Only think tokens carry loss, so `lm_head` runs on those rows alone
(`whetstone.sed.row_logits`). Memory then scales with the think segment (median
150 tokens) rather than the sequence (median 1,003, max 4,491) — the full
`(1, T, V)` tensor is 1.36 GB on the longest record before autograd saves a
copy. SED think positions are additionally capped at 1,024/record (p99 = 1,039)
so one outlier cannot spike the fp32 log-softmax intermediates; R positions are
never dropped.

### 5. Eval forwards run under bf16 autocast

Not a speed tweak. With fp32 weights, SDPA cannot use the flash/mem-efficient
kernels and falls back to the math backend, materializing a full `(T, T)`
attention matrix — a 10 GiB allocation on a 6,212-token control trace, which
OOMed the first eval.

### 6. The `</think>` sanity anchor applies to *native* traces only

The packet's alignment check is "entropy/surprisal of the `</think>` token
itself must be near zero (activities 003/005 measured ~1e-4–0.02)". Measured
under π_0 with identical code:

| corpus | `</think>` entropy, median |
|---|---|
| **native** verbose traces (`verbose_control`) | **0.000080** |
| compact register traces (`heldout_register`) | **0.275** |

The audit's own reference is 6.6e-05 over 182 native traces, so the native
number reproduces it and **the alignment is correct**. The compact number is a
real property: a verbose-CoT native does not expect a compact trace to end where
it does. Applied to the compact set the anchor reads 0.275 and looks like an
off-by-one bug that isn't one.

So the assertion (`assert_alignment`) runs on the control set, where it can
distinguish a genuine misalignment from the register's own accent; the compact
value is kept as a descriptive metric. Two further guards were added because
this check is load-bearing: it is asserted at step 0 (measured **7.59e-05**) and
merely *recorded* at later checkpoints, since alignment is a property of the
code, not of the weights.

---

## Runs

### Run 1 — R token set (turing, π_0 vLLM server on :8000) — 2026-08-03

```bash
python scripts/build_register_tokenset.py \
  --train /data/whetstone/corpora/seed_register_qwen/train.jsonl \
  --server http://127.0.0.1:8000/v1 --concurrency 32 \
  --out /data/whetstone/runs/round0/R_tokenset.json \
  --scores-out /data/whetstone/runs/round0/type_stats.json
```

Scored on the **P3-era Qwen3-1.7B server that was still resident on turing**
(1 h 49 m uptime, healthy) — 960 sequences, all `g == 1`, 195,898 think tokens,
**20 s**. The server was killed afterwards to free the GPU for training.

| quantity | value |
|---|---|
| distinct think-token types | 4,228 |
| eligible (≥ 10 occurrences) | 946 |
| mean-surprisal p75 threshold | **1.7803 nats** |
| median across-occurrence std | **1.7563** |
| `R_stats` | 13 types |
| whitelist (card §2, single-token variants) | 37 ids |
| overlap | 2 |
| **\|R\|** | **48** |

**The eyeball** (packet §5 step 5) — top of R by mean surprisal under π_0:

| id | surface | mean | std | count | source |
|---|---|---|---|---|---|
| 33939 | `goal` | **39.979** | 1.723 | 925 | stats |
| 35896 | `chk` | 20.168 | 3.755 | 296 | whitelist |
| 144016 | `⇒` | 15.321 | 3.738 | 913 | whitelist |
| 1149 | `let` | 10.167 | 3.984 | 809 | whitelist |
| 52375 | `␣✓` | 9.300 | 4.792 | 357 | whitelist |
| 5638 | `case` | 6.677 | 0.000 | **1** | whitelist |

Ordinary English words do **not** dominate, so the p75 threshold is right for
this corpus. The 13 `R_stats` members are the register's prose connectives —
`Thus` 3.03, `Since` 2.47, `hence` 2.42, `Hence` 2.41, `However` 2.05,
`tends` 2.09, `satisfying` 1.78 — which is consistent with activity 005 finding
7 ("step bodies carry more English connective prose than the card's exemplars
do"). One member is questionable and is recorded rather than hand-removed:
**`␣Yes` (4.365, n=80) is an answer *value*, not style.** Hand-tuning R per
token is how R becomes unfalsifiable; the packet's prescribed retry on a
(c) failure is a stricter std filter, and (c) did not fail.

### Run 2 — SED kernel unit tests (turing, CPU) — 2026-08-03

`python -m pytest tests/test_sed.py tests/test_segments.py -q` → **30 passed**
(6 SED + 24 segment parser). The four packet-mandated tests plus two more: the
gate is read off the shadow and not the trainee, and `Delta_t`'s direction is
large at forks and ~0 at collapse tokens (an inverted gate injects entropy into
deterministic continuations and leaves fork tokens collapsed — invisible in the
loss curve).

### Run 3 — π_0 reference cache (turing) — 2026-08-03

```bash
python scripts/precompute_pi0_cache.py \
  --corpus /data/whetstone/corpora/seed_register_qwen \
  --out /data/whetstone/runs/round0/pi0_cache.npz
```

200 control traces → 60,000 sampled think positions, 247 MB, **39 s**.
π_0 control think entropy: median 0.0173 / mean 0.2910 / p80 0.6518 nats,
against the audit baseline's 0.0278 / 0.3176 / 0.6923 — same regime, so the
control set is representative of what the audit measured.

### Run 4 — inoculation training (turing) — 2026-08-03

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/inoculate_scorer.py       # all defaults
```

Config actually used: full-FT Qwen3-1.7B, **fp32 weights + bf16 autocast**,
`AdamW8bit`, LR 1e-5, warmup 20, cosine, per-device batch 1, grad-accum 8,
grad checkpointing on, `alpha_sed` 1.0, `gamma_e` 1.0, H_pivot 0.6707,
Delta_max 0.7, `sed_max_think` 1024, KL-to-π_0 on ¬R **off**.
Loss normalizer `norm_r` = 6.26 (corpus mean R∩think tokens per record; median
5, 10 records with zero R tokens).

**120 optimizer steps, 13.8 min, 13 evals, peak 26.0 GB.** No stopping
threshold fired — the run reached the 1-epoch cap, so the winner is chosen
retroactively, exactly as the packet anticipates.

Alignment asserted at step 0: `</think>` entropy median on **native** control
traces = **7.59e-05** nats (audit reference 6.6e-05).

π_0 self-consistency at step 0 (θ *is* π_0, so the eval must reproduce the
cache): S2 KL = **0.00094**, Δlogp = **+0.00057**. Both ≈ 0. That residual is
the bf16 numerical floor between the cache pass and the eval pass and is
**S2's noise floor**.

Curves: [round0_curves.png](assets/007/round0_curves.png),
[round0_overshoot.png](assets/007/round0_overshoot.png).

### Run 5 — corrupted probe + meter tests (turing) — 2026-08-03

```bash
python scripts/build_corrupted_probe.py \
  --probe /data/whetstone/corpora/seed_register_qwen/probe_pool.jsonl \
  --out /data/whetstone/runs/round0/corrupted_probe.jsonl
python scripts/meter_tests.py --ckpts /data/whetstone/ckpt/round0/step* \
  --tau-spike 1.2 --out /data/whetstone/runs/round0/meter_tests.json \
  --assets activity/assets/007
```

**110 twin pairs from 120 probe traces** (51 chunk-deletion, 59
value-substitution, 10 ineligible). All 13 checkpoints scored.

### Run 6 — G_spike × branch retention under scorer_v1 (turing) — 2026-08-03

```bash
python scripts/gspike_branch_check.py --scorer /data/whetstone/ckpt/scorer_v1 \
  --out /data/whetstone/runs/round0/gspike_branch_check.json \
  --assets activity/assets/007
```

All 1,200 `seed_register_qwen32b` traces scored under scorer_v1.

### Run 7 — scorer_v1 frozen and served (spark) — 2026-08-03

```bash
# on spark
cd ~/workspace/whetstone-scorer && source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve /data/whetstone/ckpt/scorer_v1 \
  --port 8100 --host 0.0.0.0 \
  --gpu-memory-utilization 0.35 --max-model-len 8192 \
  --served-model-name whetstone-scorer
```

**d_t contract re-verified over HTTP** against the served scorer_v1 (P0's
check, on real heldout register sequences rather than a toy sentence):
**4,932 / 4,932 positions computable; all 4,188 rank-1 positions had d_t == 0
exactly.** Mean think d_t 0.498, p95 3.363 over 5 traces — consistent with the
3.500 measured on all 120 heldout traces through the local HF path, so the HTTP
and in-process paths agree.

---

## Findings

### 1. `goal` is the style tax, and it is 40 nats

Under π_0 the token `goal` — which opens 925 of 960 compact traces — has a mean
surprisal of **39.979 nats**. That is not "unusual", it is *categorically
impossible* to a verbose-CoT native: the model has essentially never seen a
think block begin that way. `chk` follows at 20.2 and `⇒` at 15.3.

This is the concrete form of the problem Round 0 exists to solve, and it makes
the scale of the ask clear: an untreated scorer does not mildly prefer verbose
prose, it assigns the register's opening token a probability of order e⁻⁴⁰.

After inoculation `goal` sits at **1.207** nats mean gap on the teacher corpus
(finding 7's table) — a 97% reduction.

### 2. The register p95 anchor the packet inherited does not hold on this corpus

The packet sets τ_spike = 1.2 "strictly between the verbose baseline (0.750)
and the step-0 register level (~2.375)", both from activity 004's bake-off
corpus. Measured here at step 0 with the same code:

| quantity | activity 004 | this run |
|---|---|---|
| verbose-control p95 d_t | 0.750 | **0.750** |
| clean-register p95 d_t | 2.375 | **6.375** |

**The verbose baseline reproduces exactly; the register level is 2.7× higher.**
Since both come from the same measurement path, this is a corpus difference,
not a measurement one — P3's inputs are 8,796 verbose tokens at the median
against the bake-off's 5,404, and compress 1.7× harder (activity 005 finding 7).

So τ_spike = 1.2 was never the midpoint it was designed to be. On this corpus
the equivalent midpoint is ≈ 3.2. This is the single reason the packet's
literal F1 criterion fails (finding 5).

### 3. The band exists — 91% of the style tax removed, leap detection intact

The load-bearing result. Design §8 Risk 1 asks whether calibrating away the
style tax *dulls the leap detector*. Measured across all 13 checkpoints on 110
paired corrupted/clean twins:

| step | probe AUC | corrupted p95 | clean p95 | hum_R | R-gap p95 | p95 all think | Δlogp | native `</think>` H |
|---|---|---|---|---|---|---|---|---|
| **0 (π_0)** | **0.823** | 6.972 | 2.475 | **13.065** | 40.826 | 6.375 | 0.0000 | 7.6e-05 |
| 10 | 0.813 | 6.059 | 2.269 | 5.801 | 13.159 | 5.375 | −0.0027 | 1.8e-04 |
| **20** | **0.810** | 4.000 | 2.234 | **1.154** | 5.250 | 3.500 | **−0.0446** | **0.0447** |
| 30 | 0.799 | 3.437 | 1.659 | 1.106 | 3.694 | **3.125** | −0.0754 | 1.114 |
| 40 | 0.796 | 3.906 | 1.753 | 1.461 | 8.200 | 3.500 | −0.0851 | 2.001 |
| 60 | 0.807 | 3.925 | 1.775 | 0.960 | 4.125 | 3.625 | −0.0807 | 1.776 |
| 80 | 0.816 | 4.228 | 1.781 | **0.826** | 4.762 | 3.625 | −0.0854 | 1.811 |
| 120 | 0.816 | 4.194 | 1.903 | 1.203 | 9.375 | 3.625 | −0.0845 | 1.776 |

**The register hum falls 13.065 → 1.154 nats (91% of the tax removed) while the
corrupted-trace AUC moves 0.823 → 0.810 — a change of −0.014.** At step 80 the
tax is 94% removed and the AUC is 0.816 (−0.007). Register-hum and leap-spike
are **separable**. The prefix/LoRA scorer arm is not needed.

Evidence plots: [probe_step0000.png](assets/007/probe_step0000.png) (π_0) vs
[probe_step0020.png](assets/007/probe_step0020.png) (scorer_v1), plus the two
neighbours [step0010](assets/007/probe_step0010.png) /
[step0030](assets/007/probe_step0030.png).

**The separation is real but shallow.** AUC 0.81 means that ~19% of the time a
random corrupted span scores below a random clean span. That is for a *single*
localized corruption inside a 30-token window; G_spike aggregates over a whole
think segment at high β, where a genuinely unfaithful trace breaks more than one
step. It is an adequate instrument, not a sharp one, and P5 should not assume
better.

### 4. Inoculation damages the native `</think>` boundary — a stopping signal the packet does not have

The rightmost column above. The scorer's entropy at the native think→answer
boundary goes **7.6e-05 → 0.045 (step 20) → 1.11 (step 30) → ~1.8–2.0 (step 40+)**
— four to five orders of magnitude, while the *averaged* verbose-control
likelihood barely moves (Δlogp −0.085).

This is exactly the failure mode CLAUDE.md's segment-reporting invariant warns
about: **one combined number is how drift hides.** Test (b) says "verbose
intact" and passes comfortably at every checkpoint, because a single structurally
critical position is invisible in a mean over ~60,000 tokens.

It is probably harmless *downstream* — the boundary token is excluded from
`think_mask`, so G_spike and the Stage-B gates never score it, and the segment
parser is token-level and never consults the model. But it is a real,
large, targeted degradation, and it gives Round 0 a fourth stopping signal the
packet did not specify. It independently selects step 20.

### 5. **F1 verdict: PASS on the design question, FAIL on the packet's literal threshold**

Stated in both halves because they differ, and conflating them would be the
kind of moved goalpost this project's method notes keep catching.

**PASS — the band exists (design §8 Risk 1, §11 F1's actual purpose).** 91% of
the register style tax is removed with the corrupted-trace detector essentially
unchanged (AUC 0.823 → 0.810). Calibrating away the accent does *not* dull the
leap detector. **No pivot to the prefix/LoRA scorer arm. P5 is unblocked.**

**FAIL — no checkpoint passes all three meter tests simultaneously at
τ_spike = 1.2.** The binding failure is (a), at every one of the 13 checkpoints:

- **(a) register hum — FAIL everywhere.** Best overall heldout p95 = 3.125
  (step 30); 3.500 at the selected step 20. Never within 2.6× of 1.2.
- **(b) verbose intact — PASS everywhere.** Max |Δlogp| = 0.0865 against
  ε = 0.2. *(Qualified by finding 4.)*
- **(c) corrupted probe — PASS everywhere.** AUC 0.796–0.823.

And (a) is **not satisfiable at any coherent threshold** on this corpus, not
merely unreached: clean compact-register text has a 5% tail above 4.8 nats under
scorer_v1, while τ_leap sits at 3.175. Any τ_spike low enough to call the
register "quiet" is below the level at which real leaps are detected. The
register's *mean* is a proper hum — 1.154 nats against a 0.23-nat native
baseline — but its **tail still spikes**.

Per marker class at step 20, the pass is additionally **QUALIFIED**: structural
p95 = 3.609, branch p95 = 8.819 (n = 68 heldout tokens). The branch vocabulary
is the least calibrated class, exactly as activity 005 finding 7 predicted and
as the packet's §7 metric 1 warned it would be.

**Selected checkpoint: `step0020` → `/data/whetstone/ckpt/scorer_v1`.**
**RATIFIED by the user, 2026-08-04**, after the trade-off below was put to them
explicitly with step0080 named as the defensible alternative. Not the best on
any single axis, and chosen on design §2's own criterion — *the smallest dose
that creates recognition without infection*:

| | step0020 | step0080 |
|---|---|---|
| style tax removed | 91% | 94% |
| probe AUC | 0.810 | 0.816 |
| S2 KL drift | **0.053** | 0.121 |
| Δlogp | **−0.045** | −0.085 |
| native `</think>` H | **0.045** | 1.811 |

The last 3% of tax removal costs 2.3× the KL drift and 40× the boundary damage.
Neighbours per the packet: step0010 is undertrained (only 56% of the tax
removed); step0030 matches step 20's hum but has the *worst* AUC of any
checkpoint (0.799) and 25× the boundary damage — so the band is genuinely
narrow, and it is centred on step 20.

**Non-winner checkpoints were NOT deleted** (50 GB on a 4 TB volume). The packet
asks for deletion on PASS. The *selection* is now ratified (above), so the
original reason for holding them — that step 20 rested on a criterion the packet
did not specify (finding 4) — is discharged. They are kept anyway on the weaker
ground that disk is free and P5 may want to re-measure against a neighbour when
it tunes λ/β; **deleting them is a safe cleanup at any point.** The two that
matter if pruning: keep `step0080` (the alternative on the headline axes) and
`step0000` (π_0, the Risk-1 baseline that every AUC comparison is against).

### 6. Pinned values

| knob | value | how |
|---|---|---|
| **τ_spike** | **2.25 nats** | clean-span median under scorer_v1 (2.234) — the level at which uncorrupted compact-register text sits. The packet's 1.2 is retired (finding 2). |
| **τ_leap** | **3.175 nats** | Youden-optimal on the 110 paired probes at step 20; AUC 0.810, TPR 0.78, FPR 0.26 |
| **κ_max** | **0.3174** | 3× the max of the first three evals, pinned by the run itself; never fired (max reached 0.1236) |
| **entropy floor x** | **10%** | never fired — entropy *rose* 30% (finding 8) |
| **ε** | **0.2 nats/token** | passed everywhere; max |Δlogp| = 0.087 |
| **γ_e** | **1.0** | declared placeholder, unchanged; SED behaved (finding 8) |
| S2 noise floor | 0.00094 | π_0-vs-itself, bf16 path difference |

### 7. Part 5 — G_spike does **not** select against branch retention, but **does** select against verification

Activity 006's gating question, run in its binding form under scorer_v1 over all
1,200 32B traces (`branch_kept` 155/1,200 = 12.9%; source-eligible 98.8%).

**The answer to the question asked is clean:**

| subset | β | r_pb | p |
|---|---|---|---|
| all | 5 | −0.021 | 0.47 |
| all | 10 | −0.023 | 0.44 |
| source-branching only | 5 | −0.011 | 0.72 |
| source-branching only | 10 | −0.013 | 0.66 |

At n = 1,200 these are indistinguishable from zero. **No negative correlation →
per packet §9 step 4, P5 proceeds with the unchanged product reward.** The 006
fear — that best-of-K would select back out the branch preservation the 32B
teacher exists to provide — is **not** realised.

**But the same failure mode exists one marker class over.** Testing the other
structural properties on the same 1,200 traces:

| property | rate | r_pb (β=5) | p | rank r_pb | p |
|---|---|---|---|---|---|
| `branch_kept` | 12.9% | −0.021 | 0.47 | −0.019 | 0.50 |
| **`verify_kept`** | 74.2% | **−0.113** | **<0.0001** | **−0.117** | **<0.0001** |
| `structural_pass` | 58.2% | −0.062 | 0.032 | −0.113 | 0.0001 |

Median G_spike is 4.9e-05 for verification-keeping traces vs 7.2e-05 for those
that dropped it — a **32% penalty for keeping the verification step**.

**The mechanism is identified.** Per-marker residual tax under scorer_v1 on the
teacher corpus:

| marker | mean d_t | p95 | n |
|---|---|---|---|
| **`chk`** | **7.919** | 12.000 | 781 |
| `;` | 2.301 | 6.625 | 1,329 |
| `case` | 1.827 | 7.762 | 279 |
| `✓` | 1.231 | 6.125 | 1,384 |
| `goal` | 1.207 | 5.875 | 1,159 |
| `✗` | 0.813 | 1.544 | **2** |
| `→` | 0.741 | 3.625 | 1,541 |
| `⇒` | **0.087** | 0.125 | 4,644 |
| `let` | **0.065** | 0.000 | 854 |

`⇒` and `let` are fully calibrated (0.09, 0.07). `goal` came down from 39.98 to
1.21. **`chk` did not: 7.92 nats mean, 12.0 at p95** — the single largest
residual, and it is what drives the `verify_kept` anti-correlation.

Frequency in the Round-0 corpus does not explain it on its own (`chk` has 308
occurrences, `case` only 22, yet `chk` is 4× worse). The likelier cause is
**context**: the 1.7B writes `chk:` as a trailing line after its numbered steps,
whereas the 32B writes `chk` mid-trace inside richer expressions. So calibration
learned on the student's *usage* of a marker did not transfer to the teacher's
different *usage* of the same token. That is a caution worth carrying: R is a
set of token ids, but the tax it removes is context-dependent.

**Recommendation for P5.** Two options, and the first is nearly free:

1. **Use `verify_kept` (and `branch_kept`) as selection terms** alongside
   `R_acc · G_spike · G_budget`. This is 006 open item 2's designated fix,
   already argued safe: with a *frozen* teacher, best-of-K faces one-shot
   Goodhart pressure only, so a crude structural criterion is safe to select on
   even though it is too crude to train against.
2. **Re-inoculate `chk` on teacher-style contexts** before Stage A. The packet's
   own rule — "re-inoculate between teacher rounds rather than lowering λ" —
   points here, but note this would mean inoculating on teacher text, which
   CLAUDE.md forbids for Round 0. A middle path is to enrich the *student*
   corpus with `chk`-heavy traces rather than to use 32B text.

Do **not** simply lower λ: at β ∈ {5,10} the tax is carried by the p95 tail, not
the mean, so λ is the wrong knob (design §7's failure-cascade note says the same
thing).

### 8. SED works, and S3 guards the wrong direction

Control think entropy **rose** under inoculation: mean 0.2910 → 0.3778 (+29.8%),
median 0.0173 → 0.0664 (+283%). S3 fires on a *drop* of more than x = 10%, so it
was never going to fire — the restoration-mode SED term is pushing the other way
by construction, which is what activity 003's audit verdict asked for.

Two consequences:

* **S3 as specified cannot fire in restoration mode.** It is not a live stop
  criterion for Round 0 on this checkpoint; the real entropy risk here is
  inflation, not collapse, and a rising floor makes *everything* look more
  surprising (it inflates surprisal and compresses gaps). The guard against
  that is meter test (c), which held.
* **The packet's median-based S3 has no resolving power anyway.** The audit's
  think-entropy median is 0.0278 nats with 56.8% collapse mass, so a 10% move on
  it is 0.003 nats of noise. Mean and p80 are reported alongside for this
  reason, and the mean is what the trainer trips on.

SED's own health: the K2 term rose 0.74 → ~2.9–3.2 as the masked-CE term fell
10.71 → 0.83, i.e. the two terms traded off as designed rather than one
collapsing.

### 9. Where the run actually stopped, and the overshoot signature

No S1/S2/S3 threshold fired; the run hit the 1-epoch cap at 120 steps. The
useful signal is in the shapes, and it is a variant of design §7's overshoot
signature rather than the textbook one:

* the register p95 **never crosses τ_spike**, so "register p95 keeps dropping
  past τ_spike" cannot literally occur;
* instead both halves bottom together and then diverge — S1 and hum_R bottom at
  step 30 (3.125 / 1.106) while Δlogp keeps degrading to a −0.085 plateau by
  step 40, and the structural-class p95 climbs back from 3.6 (step 20) to 10.4
  (step 120).

Read as "continued register gains bought with verbose-likelihood loss", the
rollback target is the earliest checkpoint holding the bulk of the benefit —
step 20. The per-class curves oscillate a lot (structural n = 466, branch
n = 68 heldout tokens), so single-class p95 movements between adjacent
checkpoints are noise; the overall and hum curves are the stable ones.

### 10. Method note — three would-be false findings, caught by construction

Continuing activity 005's tally, all three were caught by a check that existed
*before* the number was believed:

1. **The whitelist would have put 26.9% of the corpus into R** (deviation 1),
   including the digit tokens that meter test (c) corrupts. The packet's own
   "eyeball the top-50" instruction is what surfaced it.
2. **Zero weight drift** looked like a training bug and was a scheduler
   artefact: `get_cosine_schedule_with_warmup` sets the LR multiplier to
   `0/20 = 0` at construction, so the first optimizer step always runs at LR 0.
   Only visible because `theta_drift_rel` is logged.
3. **The `</think>` sanity anchor read 0.275** and looked like an off-by-one.
   Testing it on native traces gave 8.0e-05 against the audit's 6.6e-05 — the
   alignment was correct and the anchor was being applied to the wrong corpus
   (deviation 6). Had it been "fixed", every alignment in the packet would have
   been shifted by one.

The standing lesson holds and extends: on this project, a number that looks like
a finding should be checked against a measurement whose answer is already known.
Findings 2 and 3 above were both resolved that way — by reproducing activity
004's verbose baseline (0.750, exact) and activity 003's native boundary entropy
(6.6e-05 vs 7.6e-05).

---

## Conclusion

**F1 PASSES on the question it exists to answer — the calibration band exists.**
The register style tax can be removed (91%, `goal` from 40.0 nats to 1.2) with
the corrupted-trace leap detector essentially intact (AUC 0.823 → 0.810). The
design's largest risk is retired and the prefix/LoRA scorer arm is not needed.

**It fails the packet's literal three-tests-at-τ_spike=1.2 criterion**, at every
checkpoint, because τ_spike = 1.2 was inherited from a corpus whose step-0
register p95 was 2.375 where this one's is 6.375 — and because clean register
text keeps a 5% tail above where real leaps are detected. τ_spike is re-pinned
at **2.25** and τ_leap at **3.175**.

`scorer_v1` = step 20, frozen at `/data/whetstone/ckpt/scorer_v1`, serving on
**spark:8100** as `whetstone-scorer`, d_t contract re-verified over HTTP.

**P5 is unblocked with a quantitative answer**: G_spike does *not* select against
branch retention (r = −0.02, p = 0.47), so the activity-006 decision stands and
the product reward is unchanged. But it *does* select against verification
retention (r = −0.113, p < 0.0001), driven by a 7.92-nat residual tax on `chk`
that the Round-0 corpus could not calibrate. P5 should add `verify_kept` as a
selection term, and must not respond by lowering λ.

### What the next packet must know

* The instrument is **shallow but honest**: AUC 0.81 on single localized
  corruptions. Do not design a stage that needs a sharp threshold.
* **Calibration is context-dependent**, not token-dependent. `chk` is calibrated
  in the student's usage and not in the teacher's; expect the same wherever the
  teacher's register differs from the corpus Round 0 saw.
* **The native `</think>` boundary is degraded** (7.6e-05 → 0.045 at step 20,
  worse later). Nothing downstream scores that position today. Anything that
  starts to should re-measure first.
* Round 0's EMA copy belongs to Round 0. Stage B builds a **new** one, from the
  **original** checkpoint — the student never starts from `scorer_v1`.
