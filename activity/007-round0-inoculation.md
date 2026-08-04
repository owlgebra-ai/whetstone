# 007 — P4: Round-0 scorer inoculation and the F1 band-existence gate

- **Packet:** [packets/P4-round0-inoculation.md](packets/P4-round0-inoculation.md)
- **Status:** in-progress
- **Machine(s):** mac (code), turing (all scoring, training, meter tests)
- **Code commit(s):** `4adbabd` → `<this commit>`
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

*(filled in below)*

---

## Findings

*(filled in below)*
