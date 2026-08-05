# 009 — P6 Stage B: assimilation SFT (ZPD band-pass + SED) and the F3 gate

- **Packet:** [packets/P6-stage-b-assimilation.md](packets/P6-stage-b-assimilation.md)
- **Status:** done — **F3 FAIL**
- **Machine(s):** turing (baseline evals, training), spark (gate scoring), mac (code)
- **Code commit(s):** `b9ec970` → `995b54d`
- **Started / finished:** 2026-08-05 → 2026-08-05

## Goal

Train the student — a fresh copy of the **original** Qwen3-1.7B — on the certified
Stage-A teacher corpus with an unprivileged prompt, so the register enters the
weights. Loss is cross-entropy with ZPD band-pass token weights (Diagnosis #1 fix)
plus the SED self-distillation term in restoration mode (Diagnosis #3 fix). Verdict
is **F3**: accuracy within 1 pt of the starting checkpoint at ≤50% of its median
think tokens, with median per-token entropy above the audit baseline.

## Machine state at claim time (2026-08-05)

- turing HEAD `2317e8a`, spark HEAD `65f4dc3`, Mac HEAD `04f9494` — both boxes lag;
  synced first (ROADMAP standing rule / activity 001 gotcha 6).
- **turing's GPU was fully held (31,434 / 32,607 MiB) by the P5 32B teacher server**
  — `vllm serve nvidia/Qwen3-32B-NVFP4 … --port 8000`, PID 2724436, up 1 d 00:39,
  reparented to init, idle (no established connections on :8000, 4 CPU ticks over a
  2 s window). P5 is done and Stage B never uses the teacher. **Stopped with user
  authorization**; SIGTERM was clean, no orphaned `VLLM::EngineCore` (activity 003
  gotcha 1 checked explicitly), card back to 18 MiB.
- spark: `whetstone-scorer` (`scorer_v1`) live on **:8100**, untouched and verified
  alive after every subsequent step. **:8101** now serves the π-of-round model.
- `/data` 3.9 TB free.

## Deviation from the packet — eval scope (user direction, 2026-08-05)

The packet's Part 0b specifies a baseline card over 7 suites including
`standard_eval_300`. The user re-scoped it: **GSM8K test is the validation set;
MATH-500 / AMC23 / MinervaMath / AIME24 / AIME25 are the benchmark test sets, to be
measured across all candidates together in a later window; report Pass@1 with
decoding-seed standard deviations.** GSM8K's baseline runs first and alone.

⚠ The same direction states "there is no standard_eval_300". **There is** —
`/data/whetstone/eval/standard_eval_300.jsonl`, 300 rows, built by activity 002 and
listed in the ratified eval plan as the internal-continuity tier. It is left in
place and unused for now; F3a is therefore evaluated on `gsm8k_test` alone unless
the user reinstates it. Recorded here because F3a's wording ("within 1 pt of
baseline on **both** suites") assumes two.

## Runs

### Run 1 — Part 0a: build `gsm8k_test` (2026-08-05, spark, `1d0fa6d`)

The ROADMAP TODO that had slipped since the eval plan was ratified 2026-08-02.

- code: `scripts/build_eval_sets.py` — suite `gsm8k_test` (`openai/gsm8k`, config
  `main`, split `test`, revision `740312add88f781978c0658806c59bc2815b9866`,
  resolved from the HF API 2026-08-05), a dedicated `_norm_gsm8k`, and an
  `EXPECTED_ROWS` build-time assert.
- command:
  `PYTHONPATH=. python scripts/build_eval_sets.py --out_dir /data/whetstone/eval --suites gsm8k_test`
  (spark, `~/workspace/whetstone-scorer/.venv` — the CPU venv at
  `~/git/whetstone/.venv` has no `transformers`).
- result: **1,319 rows, 0 skipped**. 0 schema violations, 1,319 unique `_uid`, 1,319
  unique prompts, 0 non-numeric golds, 14 golds carrying thousands commas, no `####`
  or `<<…>>` leakage into prompt or gold.
- **Why a dedicated normalizer:** GSM8K's `answer` field is the *whole* reference
  derivation including `<<48/2=24>>` calculator annotations; the gold is only the
  tail after the `####` marker. The generic `_norm_math` path takes `answer`
  verbatim via `_first(rec, "answer", …)`, which would have graded a number against
  a paragraph and scored the suite near 0% — a data failure that reads as a model
  failure.
- **Gold stored verbatim (stripped only)**, against the packet's word "normalized",
  because this module's standing rule is verbatim golds and `verify._normalize`
  already deletes commas at compare time. Round-tripped: gold `1,000` verifies
  against both `\boxed{1000}` and `\boxed{1,000}`; `\boxed{71}` against gold `72`
  is False.
- deviation, minor: `build_eval_sets.py` dumped only the current run's summary over
  the shared `eval_stats.json`, so building one suite into a directory of seven
  would have erased their pinned revisions. Changed to merge (`1d0fa6d`).
  `eval_stats.json` now holds all nine entries.

### Run 2 — `run_eval.py` gains the four baseline-card numbers (`c6f2f93`)

F3 compares against Pass@1 ± seed std, think and answer medians **separately**,
cap-hit rate and g-rate. `run_eval.py` reported **none** of them — only
`n_tokens_total`, the combined think+answer number CLAUDE.md forbids. Added:

- `pass_at_1_mean` / `_std` / `_per_draw`: draw *k* is sample index *k* over the
  whole suite, so the std is the spread **between decoding draws**. That is what
  makes a 1-point F3 threshold interpretable. `strict_accuracy_at1` (draw 0 alone)
  is kept for continuity with pre-009 runs.
- `think_tokens_median` / `answer_tokens_median` via `segments.parse_segments` on
  the generated ids, over **g=1 generations**, with `*_median_all` beside them: a
  truncated rollout has no `</think>`, so its think length is "everything generated"
  and its answer length 0, and mixing those in makes a cap-hit read as verbosity.
- `cap_hit_rate` from `finish_reason`, `g_rate` from the SCA quality gate.
- `--limit N` for smoke runs, stamping `limited: true` into the summary.

**Verified the draws are genuinely independent before trusting the ± std**: on a
12-problem smoke, 0/12 problems had byte-identical draws and all 4 draws differed in
token count on every problem (problem 0: think 1,924 / 2,675 / 2,186 / 2,985). A
seeded `n=8` could in principle have collapsed to one sample repeated, which would
have made every ± number in this project meaningless.

### Run 3 — Part 0b: GSM8K baseline for the original checkpoint (turing, **in flight**)

- command:
  ```
  PYTHONPATH=. python scripts/run_eval.py --model Qwen/Qwen3-1.7B \
    --suites gsm8k_test --suite_dir /data/whetstone/eval \
    --output_dir /data/whetstone/eval/baselines/qwen3-1.7b-original \
    --K 8 --temperature 0.7 --top_p 0.95 --max_tokens 32768 \
    --max_model_len 36864 --enable_thinking --no_system_prompt --seed 0
  ```
- 10,552 generations in **6,547 s (1 h 49 m)**; log
  `/data/whetstone/runs/009/baseline_gsm8k_test.log`. Card written to
  `/data/whetstone/eval/baselines/qwen3-1.7b-original/CARD.md`.

| metric | value |
|---|---|
| **Pass@1 (mean over 8 decoding draws)** | **90.49% ± 0.25** |
| per-draw Pass@1 | 90.75, 90.45, 90.07, 90.45, 90.52, 90.22, 90.60, 90.83 |
| pass@8 | 95.07% |
| **median think tokens** | **1,477** |
| **median answer tokens** | **288** |
| cap-hit rate | 0.00% |
| g-rate | 100.00% |

g-rate is 100%, so the g=1 medians and the all-generation medians are identical —
no truncation mass is hidden behind them. The ± 0.25 pt seed std means F3a's
1-point threshold is ~4σ, a real bar rather than noise.

**Derived F3 goalposts (gsm8k_test): F3a ≥ 89.49% · F3b think ≤ 738 and answer in
the band around 288 · F3d g ≥ 99%.**

> **Finding 5 — there are now TWO length baselines and confusing them would
> invalidate F3b.** They differ by 4×:
>
> | protocol | suite | T | cap | think median | answer median |
> |---|---|---|---|---|---|
> | eval (§12.7) — **F3b is measured here** | gsm8k_test | 0.7 | 32,768 | **1,477** | **288** |
> | entropy audit (activity 003) — **F3c is measured here** | val_2k | 0.9 | 16,384 | **6,099** | **679** |
>
> The famous 6,099 is the *audit* number: `val_2k` is DeepMath-heavy and sampled
> hotter. On GSM8K under the eval protocol the model thinks 1,477 tokens. Quoting
> 6,099 as the F3b baseline would set the bar at ~3,000 instead of 738 — a gate
> that passes on arrival. Both numbers stay; each belongs to exactly one gate.

> **Finding 6 — the answer segment is expected to GROW, and F3b's wording assumed
> the opposite.** Baseline answer median **288**; the corpus's is **475**
> (finding 1). The packet says to "confirm it stays in the baseline's band —
> answers must NOT compress"; the live risk is a **+65% expansion**, not
> compression. Total output therefore goes 1,765 → ~726 (**2.4×**) against the
> think-only reading of 1,477 → 251 (**5.9×**). All three numbers get reported;
> the think-only one alone overstates the result by 2.5×.

**F3c protocol pinned** from the audit's own `config` block, so the student is
measured identically: `entropy_audit.py --pool /data/whetstone/data/pool/val_2k.jsonl
--n 200 --seed 0 --temperature 0.9 --top_p 0.95 --max_tokens 16384 --max_len 20480
--chunk 1024`, top-k 512. Baseline think entropy mean **0.31759**, p50 **0.027817**,
p80 **0.69234**, collapse mass 56.8%.

**Continuity substitution (recorded):** the packet's per-checkpoint cheap-mode runs
target `standard_eval_300`, which the user's re-scoping removed. Substituted:
`gsm8k_test --limit 300 --K 1 --temperature 0` — same role (cheap trend line,
never the verdict), same suite as F3a so the trend and the gate cannot diverge.

### Run 4 — Parts 1–3: dataset, ZPD gates, γ (spark + turing, `7ec4859`)

**Part 1 — assembly** (`scripts/stageb_build_dataset.py`, new). Every sequence built
through `round0.build_sequence` and the token ids written to disk, so the gate pass
and the trainer consume identical ids and nothing re-tokenizes downstream.

- output: `/data/whetstone/corpora/stageb/golden/train.jsonl`
- **2,414 records, 0 assembly failures.** Every record cleared: no boundary tokens in
  `compact_think`/`answer`, segment gate g=1, `verify_response` on the rebuilt
  completion.
- **think tokens 750,087 — matches `GOLDEN_HANDOFF.md` exactly.** Think-token share
  by level L1 15.2%, L≥6 56.3% — also an exact match. The construction reproduces
  the corpus's own accounting, which is the check that matters.

> **Finding 1 — the corpus's answers are longer than its think blocks, and the
> headline compression ratio depends entirely on which segment you count.**
> Median think **251** tokens; median answer **475**; mean answer 481, max 1,755.
> Verified against the corpus's own `answer_tokens` field: 0 mismatches over 2,414.
> Against a verbose source of median 4,078 think tokens:
> **think-only ratio 16.2×, whole-completion ratio 5.5×.**
> 47.3% of answers are sectioned write-ups (`###` / `**Step` / `Step N`), i.e. a full
> re-derivation after the compact scratchpad rather than a statement of the result.
> **Consequence for F3b:** the think-length win will look ~3× larger than the
> output-length win. The packet already mandates reporting answer length separately;
> this run adds that **total completion length must be reported beside the two
> segment numbers**, or the headline overstates the result. It also sharpens
> activity 008's "compression is flat in absolute terms" — some of the reasoning did
> not compress, it *migrated across the boundary*.
>
> Register leakage into answers is nonetheless low: marker density **0.042 per
> 100 chars in answers vs 2.061 in think (49×)**, median 0. 18.0% of answers contain
> some marker string, but that is dominated by the English word `case` (10.2%) and
> `✓` (6.1%); the register-specific ones are rare — `⇒` 1.5%, `goal:` 0.7%,
> `chk:` 0.5%. F3d's leakage metric must be register-specific (line-initial
> `goal:`/`chk:`/`case N:`, plus `⇒`/`✗`), not a bare substring count, or the
> English word `case` alone will fail the gate.

**Part 2 — gate precompute** (`scripts/stageb_zpd_gates.py`, new). π-of-round server
`vllm serve Qwen/Qwen3-1.7B --served-model-name whetstone-pi-round1 --max-model-len
8192 --gpu-memory-utilization 0.35 --port 8101` with `VLLM_USE_FLASHINFER_SAMPLER=0`;
ready in ~75 s; `:8100` verified alive afterwards.

- `2,414 / 2,414 scored, 0 failures, 132 s at concurrency 24 (18.2 rec/s)`.
- output `/data/whetstone/corpora/stageb/golden/gates_round1.npz` (S_t over
  completion positions) + sidecar recording π_S and a content hash, which the
  trainer will assert against its own round.
- **think S_t under the ORIGINAL checkpoint: mean 1.168, p50 0.00147, p90 3.652,
  p99 15.387, max 45.328 nats.**

**Part 3 — γ** (`scripts/stageb_pin_gamma.py`, new).
**γ pinned at the design init, ln(1e-4) = −9.2103**, κ=1, α_nov=0.5, s_cap=4.
Gate closes below 0.1 above **S = 11.41 nats**; novelty boost starts at S = 0.4.
Plots: [`zpd_gamma.png`](assets/009/zpd_gamma.png),
[`zpd_by_level.png`](assets/009/zpd_by_level.png); numbers
[`gamma_round1.json`](assets/009/gamma_round1.json).

> **Finding 2 — activity 006's masking fear does NOT materialise, and masking is
> flat in difficulty.** Overall masked fraction **2.09%**, *below* the packet's
> 5–30% sanity band rather than above it. Per level:
>
> | | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
> |---|---|---|---|---|---|---|---|---|---|
> | masked % | **4.80** | 2.82 | 2.19 | 1.46 | 1.47 | 1.50 | 1.62 | 1.80 | **2.16** |
> | boosted % | 26.97 | 29.57 | 23.84 | 21.31 | 22.19 | 23.10 | 24.16 | 26.87 | **32.16** |
> | mean w_t | 1.191 | 1.244 | 1.213 | 1.196 | 1.208 | 1.218 | 1.225 | 1.251 | 1.306 |
>
> The packet's failure signature was "masked > 40% in the hard band". The hard band
> is at **2.16%**, and the *most*-masked tier is **L1**, not L9 — masking here
> tracks register density (L1 has the highest marker density, 3.49/100ch) rather
> than reasoning difficulty. Activity 006 open item 2 is answered under the right
> model: **the 32B teacher's compact traces are within the 1.7B student's reach at
> every level.**
>
> ⚠ **Scope of that claim.** This measures teacher-forced token predictability, not
> the student's ability to *produce* the reasoning unprompted. Low surprisal given
> the preceding gold tokens is a much weaker statement than reachability, and it is
> the only thing a ZPD histogram can measure. Do not quote finding 2 as evidence the
> student can already reason this way.

> **Finding 3 — the register markers are heavily gated but they still flow, and
> their weight is *above* average, not ≈0.**
>
> | class | n | share of think | S mean | S p50 | S p90 | S max | masked | mean w_t |
> |---|---|---|---|---|---|---|---|---|
> | structural (`⇒ → goal let ;`) | 18,963 | 2.53% | 9.896 | 5.218 | 38.877 | 45.328 | **27.0%** | **1.323** |
> | branch (`case ✗ chk ✓`) | 7,006 | 0.93% | 10.198 | 9.779 | 22.112 | 33.844 | **43.8%** | **0.929** |
>
> The mean surprisal (~10 nats) reproduces activity 007's ~40-nat `goal` in its
> **p90/max**, but the **median is 5.2 / 9.8 nats** — the distribution is bimodal,
> and the majority of marker occurrences are predictable *in context*. So 27–44% of
> marker tokens are gated off, while the surviving majority land in the novelty
> band and carry structural mean w_t **1.323 against a corpus mean of 1.218**.
> **The register flows from epoch 1.** The packet's expectation — "these tokens
> arrive with weight ≈ σ(κ(−40+9.2)) ≈ 0 … the register enters gradually as π_S
> catches up" — held only for the tail. Branch markers, at 43.8% masked and mean w_t
> 0.929, are the half-throttled class and the one to watch: they are also the class
> activity 008 found the teacher abandoning on hard problems.

> **Finding 4 — γ is not a sensitive knob for this corpus.** Sweeping γ over
> −11.5 → −5 moves the masked fraction only **1.3% → 4.9%**. The design init needs
> no adjustment and nothing downstream should be tuned on it. (Design §12.6 table:
> γ = −9.2103, measured, not placeholder.)

- fix during the run (`7ec4859`): the reported gate-threshold annotation had a sign
  error — `gate < 0.1` solves to `S > −γ + ln9 = 11.41`, not `−γ − ln9 = 7.01`. The
  measured fractions were never affected (they come from the gate array directly),
  but the JSON field and the histogram's threshold line both pointed at 7.01, where
  the gate is actually 0.90. Verified numerically before and after.
- gotcha for the next agent: **matplotlib is on turing only** — neither spark venv
  has it. Gate scoring runs on spark, plotting on turing, both off `/data`.

### Run 5 — Part 4: the trainer (`8668c13`, GPU-blocked pending Run 3)

`whetstone/zpd.py` (new) holds the band-pass formula, its masked-fraction
threshold and the sequence normalizer; `stageb_pin_gamma.py` and
`stageb_train.py` both import it. Verified the refactor reproduces Run 4's
numbers bit-for-bit (masked 2.09%, structural mean w_t 1.323, branch 43.8%).

`scripts/stageb_train.py` (new). Pinned decisions, recorded because the packet
asked for them explicitly:

- **CE over all completion tokens; SED over think tokens only.** The student has
  to learn to write the answer segment too, while entropy restoration is a
  reasoning-channel job (design §4.2).
- Per-sequence normalization by `Σw` with the `0.25·n_completion` floor.
  **The floor binds on 0 / 2,414 sequences** — independent confirmation that γ
  is not mis-set for this corpus (finding 4).
- Numerics: fp32 master weights, bf16 autocast, `AdamW8bit`, LR 2e-5, warmup 30,
  cosine, accum 8, grad checkpointing. `theta_drift_rel == 0` after step 2 raises
  rather than warns.
- EMA cadence is *checked*, not trusted: every eval compares `sed.n_syncs`
  against `step // sync_every` and shouts if micro-batches are being counted.

Two efficiency choices that were correctness choices in disguise:

- CE runs **inside** the autocast block through `F.cross_entropy` rather than
  `log_softmax` + `gather`. Identical fp32 arithmetic (`cross_entropy` is on
  autocast's fp32 list), but the fused kernel recomputes in backward instead of
  saving an `(N, 151936)` fp32 log-softmax — **1.7 GB of activation for a single
  sequence** on this corpus's longest record (2,830 completion tokens).
- The generative spot-check batches with **left padding** on its own coarser
  cadence (`--spot-every 100` vs `--eval-every 25`), toggling `use_cache` and
  gradient checkpointing around the call. One-at-a-time it is ~30 min *per eval*
  before assimilation, when every rollout still runs to the 2,048 cap.

**F3d leakage detection is register-specific** — `goal:`, `chk:`, `⇒`, `✗` — not
the full marker set, because `case` is an English word appearing in 10.2% of the
corpus's own answers (finding 1). A bare substring count would fail F3d on prose.

Guards verified on CPU before spending GPU time:

| guard | result |
|---|---|
| gate sidecar π_S = `Qwen/Qwen3-1.7B`, round 1 | **passes** (sha `64739e94b69fddda`) |
| π_S = `scorer_v1` | **refused** — STALE GATES |
| π_S = `Qwen/Qwen3-4B` | **refused** — STALE GATES |
| `train.jsonl` edited after gates were built | **refused** — length mismatch |
| record load | 2,414 seqs; w over 1,916,972 completion tokens, mean 1.1619, p50 0.9999, 0.75% masked, 2.05% at the novelty peak; per-problem weights all 1.0 |

### Run 6 — Round 1 golden: **STOPPED at step ~225 / 602, diagnosed** (2026-08-05)

Launched at `0763f8b` after a worst-case smoke (40 longest records: peak 28.1 GB
of 32.6, drift non-zero, CE/SED trading off, EMA cadence exact). Config as the
packet: 2 epochs, LR 2e-5, warmup 30, accum 8, α_sed 1, γ −9.2103.
Log `/data/whetstone/runs/009/round1_golden.log`; checkpoints step0050–0200 and
10 metric rows preserved.

**The run was stopped because it was provably going to fail for a now-understood
reason.** Trajectory of the generative spot-check (20 held-out val problems,
greedy, 2,048 cap):

| step | think median | answer median | g-rate | marker density |
|---|---|---|---|---|
| 0 (original) | 2,047 (capped) | 0 | 0.20 | 0.164 |
| 100 | **1** | 434.5 | 0.95 | **0.0** |
| 200 | **1** | 279.0 | 0.80 | **0.0** |

The student emits `<think>\n</think>` — an **empty scratchpad** — and then a
competent answer. Note the answer median at step 100 (434.5) is close to the
corpus's 475: **the answer segment assimilated correctly and the think segment
collapsed.** Not recovering — g fell 0.95 → 0.80 between the two readings.

Entropy meanwhile rose and held far above the audit baseline (control-slice mean
0.329 → 0.557, median 0.0125 → 0.166, p80 0.649 → 1.050), so SED is doing its job;
this is not an entropy failure.

> **Finding 7 — the ZPD band-pass cannot install a register whose entry token is
> outside the student's reach, and no γ fixes it.**
>
> Weight by position inside the think block, over all 2,414 traces:
>
> | think position | most common token | S mean | w mean | masked |
> |---|---|---|---|---|
> | 0 | `\n` (100%) | 0.000 | 0.9999 | 0.0% |
> | **1** | **`goal` (100%)** | **40.082** | **0.0000** | **100.0%** |
> | 2 | `:` (100%) | 1.983 | 1.9723 | 0.0% |
> | 3 | ` total` (14%) | 8.738 | 0.9517 | 33.2% |
>
> **Every one of the 2,414 traces opens `<think>` `\n` `goal` `:`, and position 1
> is masked in 100.0% of them.** The single highest-leverage token in the corpus —
> the one that starts every register trace — contributes nothing to the loss. The
> student is never taught to emit it, so at generation time the most probable
> continuation after `<think>\n` is `</think>`.
>
> The pattern is **line-initial markers**, not markers in general:
>
> | marker | n | S mean | w mean | masked |
> |---|---|---|---|---|
> | `goal` | 2,424 | 39.96 | 0.005 | **99.7%** |
> | `chk` | 1,828 | 21.29 | 0.033 | **98.4%** |
> | `⇒` (bare, line-initial) | 2,424 | 13.55 | 0.535 | **58.2%** |
> | ` ⇒` (mid-line) | 5,654 | 3.61 | 1.716 | 3.7% |
> | ` →` (mid-line) | 2,827 | 3.03 | 1.656 | 2.9% |
> | `let` | 1,603 | 9.56 | 1.449 | 22.5% |
>
> Mid-line symbol *usage* is learnable; the line-opening *keywords* that carry the
> register's skeleton are not.
>
> **γ is not the knob** (γ sweep against `goal`, S mean 39.9):
>
> | γ | `goal` w mean | corpus masked | corpus w mean |
> |---|---|---|---|
> | −9.2103 (pinned) | 0.010 | 1.54% | 1.215 |
> | −20 | 0.022 | 0.38% | 1.299 |
> | −30 | 0.023 | 0.30% | 1.306 |
> | −42 | 2.388 | **0.00%** | 1.313 |
>
> γ must reach ≈ −42 before `goal` carries weight, and by then the gate is
> **entirely off** — which is v1's Diagnosis-#1 failure restored. There is no γ
> that installs the register and still gates unreachable tokens. The packet's
> instruction not to slide γ mid-run was correct for a reason it did not name.
>
> ⚠ **This corrects finding 3.** "Structural markers carry above-average weight
> (mean w_t 1.323)" is arithmetically right and diagnostically wrong: that class
> pools 5,654 mid-line ` ⇒` at w 1.72 with 2,424 line-initial `goal` at w 0.005,
> and the high-count majority swamps the load-bearing minority. **A mean over a
> marker class is the wrong statistic**; the entry tokens need their own row. The
> packet's stated tripwire — "marker-token mean w_t stays ≈ 0 through round 1" —
> would never have fired here.

> **Finding 8 — SED compounds it, and any fix must address both terms.** SED
> distills the student toward its EMA shadow, which is initialised from the
> original checkpoint and therefore also assigns `goal` ≈ e^−40. K2 =
> ½(log π_θ − log π_φ,τ̂)², so if CE were to lift log π_goal from −40 to −3, K2 on
> that token becomes ½(37)² ≈ 684 — against a current whole-sequence SED mean of
> 0.24. **SED actively holds the register entry tokens at zero probability.**
> Ungating them in the CE term alone would put the two terms into direct
> opposition at ~10× the current SED magnitude.

### Run 7 — the register-whitelist floor (user-ratified fix), and a refuted alternative

**Fix ratified by the user 2026-08-05:** card §2 tokens get a weight floor of 1.0
inside the think segment, and are exempted from SED. Implemented in
`whetstone/zpd.py` as a **floor** (`np.maximum`), not a replacement — mid-line
markers already earn more than 1.0 and overwriting would demote them. Verified
monotone: `goal` 0.000 → 1.000, `chk` 0.000 → 1.000, line-initial `⇒` 0.039 →
1.000, mid-line `⇒` 2.795 unchanged, ordinary token 1.250 unchanged. 37 card
token ids; **32,127 floored tokens = 1.68% of completion tokens**; corpus mean
masked 1.89% → 0.75%.

**It works.** Same config, same seed, spot-check at the first reading:

| | think median | answer median | g-rate | marker density |
|---|---|---|---|---|
| no floor, step 100 | **1** | 434.5 | 0.95 | **0.000** |
| no floor, step 200 | **1** | 279.0 | 0.80 | **0.000** |
| **floor, step 50** | **869** | 86.5 | 0.70 | **0.493** |

The think block is alive and register markers are being emitted (0.164 at step 0
→ 0.493), against a corpus density of ~2.0. Register-marker mean w_t rose
1.19 → 1.53. Partial no-floor and floor-only runs preserved under
`/data/whetstone/runs/009/failed_nofloor/` and `floor_only_partial/`.

> **Finding 9 — REFUTED: stripping `goal:` does not fix the collapse, because the
> 40 nats is the POSITION, not the token.**
>
> User hypothesis (2026-08-05): delete the `goal:` label from the corpus so the
> model implicitly learns to verbalise the goal, avoiding any change to design
> §4.1. Motivated by a measurement in this journal — the goal-statement text
> after `goal:` costs only **1.40** nats mean (p50 0.00, 2.4% masked).
>
> Tested properly: built `/data/whetstone/corpora/stageb/golden_nogoal/`
> (2,414 records, 745,573 think tokens, `--strip-markers "goal:"`) and re-scored
> the gates under the original checkpoint. Result:
>
> | corpus | think pos 1 top token | S mean | w mean | masked |
> |---|---|---|---|---|
> | with `goal:` | `goal` (100%) | 40.082 | 0.0000 | **100.0%** |
> | **stripped** | `total` (14%) — diverse | **35.876** | **0.0000** | **100.0%** |
>
> **Position 1 is still masked in 100% of traces.** After `<think>\n` the
> original checkpoint expects *its own verbose opening* ("Okay, so I need
> to…"); any terse register opening is ~36–40 nats surprising whatever its first
> word. Removing the label just moves the cost onto the first content word.
>
> ⚠ **This corrects the 1.40-nat measurement that motivated the hypothesis.**
> That number was taken from positions *after* `goal:` — it is the cost of the
> goal statement **given the label**, not cold. The stripped corpus asks the model
> to produce that text cold, and cold it costs 35.9 nats.
>
> **Corollary, and the reason the strip is worse than neutral:** `goal` is a card
> §2 token, so the whitelist floor can catch it. `total`, `evaluate`, `find` are
> not, so nothing can floor them. **Stripping removes the only handle the fix had
> on the trace-opening position**, and the strip arm would collapse the same way
> with no available remedy. The stripped corpus and its gates are kept for the
> record; they were not trained.
>
> The user's design judgment that `goal:` adds little to the *reasoning* may still
> be right — but it cannot be delivered by deletion alone. It would need a
> positional floor on the first think tokens, which is a strictly larger change
> than the card-token floor already ratified.

> **Finding 10 — a natural opener (`goal:` → `Okay,`) buys a free door and pays
> for it in every room.** Third user hypothesis (2026-08-05), and the sharpest
> test of finding 9: if the ~40 nats is positional, hand the model the opening it
> natively writes and the entry point should cost nothing.
>
> It costs nothing — and the wall moves one slot later:
>
> | corpus | pos 1 | pos 2 | pos 3 | whole think block masked | think S mean |
> |---|---|---|---|---|---|
> | `goal:` | `goal` **40.082**, 100% masked | `:` 1.983 | ` total` 8.738, 33% | **1.66%** | 1.168 |
> | stripped | `total` **35.876**, 100% masked | 5.026, 8% | 2.265, 2% | 1.87% | 1.195 |
> | **`Okay,`** | `Okay` **0.001, 0% masked** | `,` 0.000, 0% | ` total` **30.554, 100% masked** | **3.39%** | **1.505** |
>
> Two results, pulling opposite ways:
>
> 1. **Conditioning on a natural opener genuinely reduces the entry cost** — the
>    first *content* token goes 40.082 → 35.876 → **30.554** across the three. A
>    real 24% effect, and direct confirmation that the cost is contextual rather
>    than lexical.
> 2. **But it makes the whole trace worse.** Block-wide masking nearly doubles
>    (1.66% → 3.39%) and mean think surprisal rises 1.168 → 1.505. `Okay,` primes
>    the model for its own verbose continuation, so every terse register line that
>    follows is *more* surprising than it would have been after `goal:`. The
>    opener buys one cheap token and taxes the ~250 that follow.
>
> **Ranking on masking: `goal:` (1.66%) < stripped (1.87%) < `Okay,` (3.39%).**
> The original corpus plus the whitelist floor is the best of the three, and the
> floor is the only intervention that addresses the entry point without making the
> rest of the trace harder. All three corpora and their gate files are kept under
> `/data/whetstone/corpora/stageb/{golden,golden_nogoal,golden_okay}/`.

### Run 8 — Arm A (whitelist floor) COMPLETE: round 1, 602 steps, 32 min

`/data/whetstone/runs/009/round1_golden/`, checkpoints
`/data/whetstone/ckpt/stageb/golden/round1/` (13 saves + `epoch1` + `final`, 54 GB).

**Numerics sanity — all clean.** EMA syncs **120 = 602 ÷ 5 exactly** (optimizer
steps, not micro-batches); `theta_drift_rel` **3.400e-03**, never the silent-no-op
zero; peak memory **28.3 GB** of 32.6; Z-floor bound on 0 / 2,414 sequences.

**Assimilation trajectory** (20 held-out val problems, greedy, 2,048 cap):

| step | think | answer | g | markers/100ch | leak |
|---|---|---|---|---|---|
| 0 | 2,047 | 0 | 0.20 | 0.164 | 0.00 |
| 100 | 490 | 223 | 0.80 | 0.511 | 0.00 |
| 300 | 484 | 236 | 0.70 | 0.633 | 0.00 |
| 600 | **558** | 344 | 0.75 | **0.788** | **0.00** |

**Entropy — restoration mode delivered on every statistic** (control slice,
teacher-forced, top-512):

| | step 0 | final | audit baseline |
|---|---|---|---|
| mean | 0.3286 | **0.6455** | 0.3176 |
| median | 0.0125 | **0.2603** | 0.0278 |
| p80 | 0.6490 | **1.2282** | 0.6923 |

Median entropy ends **9.4× the audit baseline**. Diagnosis #3 (entropy collapse
before RL) is not merely avoided here, it is reversed. Note this is the
teacher-forced control slice; **F3c is decided on the student's own fresh
rollouts** via `entropy_audit.py`, still to run.

**Open concern — g-rate 0.75 against F3d's ≥0.99.** Unresolved at this point: the
training panel is greedy with a **2,048-token cap** on DeepMath val problems,
and a cap that tight produces `missing_think_close` failures that say nothing
about the model's formatting. The GSM8K eval runs at a 32k cap and separates
them. The gate-reason counter added at `3a77b27` reports this directly from
round 2 onward; arm A predates it, deliberately (arms A and B must be read on
the same statistic, so turing stayed pinned at `fd54aca` for both).

**Panel caveat found mid-run.** `spot_answer_median` spans *every* generation, and
a cap-hit has no `</think>` so it contributes answer_len 0. At g≈0.6 that made
the answer median read 21.5 at step 400 and 317.5 at step 450 — the swing is
which rollouts failed, not what the model writes. `spot_*_median_ok` (g=1 only)
added at `3a77b27` for later runs.

### Run 9 — Arm B (`goal:` stripped) COMPLETE: the refutation, measured

602 steps, same seed and config as arm A. **Never emitted a single register
marker.** Failure cycled through three modes — no boundary tags at all
(steps 50–100, g=0.00), `<think>` opened and never closed (150–250), empty
`<think>` with a competent answer (300) — and ended cap-hitting 95% of rollouts.

| step | A think / mark / g | B think / mark / g |
|---|---|---|
| 100 | 490 / 0.511 / 0.80 | **0 / 0.000 / 0.00** |
| 300 | 484 / 0.633 / 0.70 | **1 / 0.000 / 0.60** |
| 600 | **558 / 0.788 / 0.75** | **2047 / 0.000 / 0.05** |

Corpus verified clean first: **0 / 2,414 malformed**, think median 249, goal
statements intact. The collapse is the training result, not a pipeline bug.

> **Finding 11 — loss and entropy cannot distinguish a working run from a dead
> one.** Final: CE **0.5399 (A) vs 0.5524 (B)**; SED K2 **0.16606 vs 0.16610**;
> entropy mean **0.6456 vs 0.6381**, p80 **1.2295 vs 1.2127**. Every conventional
> SFT signal says the two runs are equivalent and healthy. One produces a working
> student; the other produces nothing parseable. **Only the generative panel
> separates them**, and it separates them completely. This is why the packet
> makes the spot-check a first-class deliverable, and it is why the original
> collapse went unnoticed for 100 steps — CE and entropy both looked fine.

### Run 10 — Arm A round 2 COMPLETE (301 steps, 41 min)

Gates recomputed under the round-1 student (fingerprint asserted), fresh EMA from
the round-1 student, 1 epoch, seed 1. EMA syncs **60 = 301 ÷ 5**, drift 2.47e-03,
peak 28.7 GB.

**The recomputed gates visibly did their job:** register-marker mean w_t fell
**1.53 → 1.33** (the student now finds the markers less surprising, so the floor
carries less), and SED K2 fell **0.166 → 0.048** (fresh shadow starts adjacent to
the trainee rather than 602 steps behind).

**Assimilation, on well-formed generations** (the `_ok` medians added at `3a77b27`):

| | round-1 final | round-2 final | corpus |
|---|---|---|---|
| think (g=1 only) | 420 | **246** | **249** |
| markers / 100 chars | 0.860 | **1.703** | ~2.0 |
| answer leakage | 0.00 | **0.00** | — |

**Think length lands within 3 tokens of the corpus median.**

> **Finding 12 — the panel's g-rate was never a formatting problem.** Gate-failure
> reasons across every round-2 spot-check are **100% `missing_think_close`** — zero
> `empty_answer`, zero `duplicated_think_*`. The failures are rollouts that think
> past the panel's 4,096-token cap on hard DeepMath val problems, i.e. the
> measurement budget, not the model. The raw `think_all` median of 495 was those
> cap-hits contributing 4,096 each; the g=1 median is 246. **F3d must be read off
> the GSM8K eval at a 32k cap**, never off this panel.

> **Finding 13 — round-2 entropy declines, then stabilises.** Peak **0.6898**
> (step 50) → **0.6318** (step 301): −8.4% mean, −14.6% median, −8.7% p80 from
> peak, but flat over the last ~75 steps (0.6310 / 0.6413 / 0.6324 / 0.6319 /
> 0.6318). It reads as an equilibrium between CE fitting the corpus and SED
> pushing back, not a collapse. Final still **2.0× the audit baseline mean, 9.5×
> the median, 1.67× p80**.
>
> ⚠ Two caveats. Rounds 1 and 2 used **different control slices** (seed 0 vs 1),
> so cross-round absolute values are loose; the within-round-2 decline is real
> because that slice is fixed. And this is teacher-forced entropy on corpus text —
> **F3c is decided on the student's own fresh rollouts**, not measured yet.
>
> **Gap owned:** SED's own diagnostics were not being logged, so whether τ̂ was
> saturating at its 1.5 ceiling — the packet's named diagnosis for exactly this
> decline — could not be checked. Fixed at `995b54d` (τ̂ hi/lo/mean, Δ, H_shadow,
> plus collapse mass and fork mass at both 0.7 and 1.5 nats). Applies from the
> control arm onward.

**Ops note.** The round-1 eval was launched on spark and **abandoned at 27% with a
14h50m ETA** — throughput had decayed to 645 tok/s. The same eval on turing runs
at **6,800 tok/s (10×)**. CLAUDE.md's "never schedule decode-heavy generation on
spark" is confirmed quantitatively; the earlier 1h14m estimate was an artifact of
the easy head of the batch. Killing it left an **orphaned `VLLM::EngineCore`
(PID 2358915, PPID 1) holding 51 GB** — activity 003 gotcha 1, reproduced — and
`pkill -f "scripts/run_eval.py"` self-matched the ssh command line running it, so
the orphan had to be killed by PID.

### Run 11 — F3 evaluation: **F3 FAILS**

Full-protocol evaluation proved impractical: both checkpoints ran **13–17 h ETAs**
on turing at N=8/32k, against the baseline's 1 h 49 m. Cause measured below
(finding 14). Screening runs used instead — **200 problems × K=2, full sampling
protocol otherwise**, and the subset was validated rather than assumed.

**Subset validity.** GSM8K carries no level annotation (every row level 0), so the
check is the baseline's own accuracy on the identical 200: **90.62% ± 0.52 vs
90.49% ± 0.25 on the full 1,319 — a +0.14 pt bias against a ±4.04 pt binomial CI
at n=200.** Think median 1,540 vs 1,477 and answer median 291 vs 288, i.e. if
anything marginally harder. Prompt length matches (mean 242 vs 239). The subset
misses the extreme gold-magnitude tail (max 95,200 vs 2,880,000), which cannot
move a 23-point gap. **Comparisons below use 90.62%, the subset baseline.**

| | Pass@1 as-scored | Pass@1 strict | think | answer | cap-hit | g-rate |
|---|---|---|---|---|---|---|
| **baseline** (same 200) | **90.62 ± 0.52** | **90.19** | 1,540 | 291 | 0.00% | 100.0% |
| **round 1** | **67.25 ± 0.35** | **65.00** | **215** | 180 | 3.25% | 94.25% |
| **round 2** | **63.50 ± 1.41** | **59.75** | **173** | **19** | 3.50% | 96.25% |

### **F3 VERDICT: FAIL**

| gate | bar | round 1 | round 2 | |
|---|---|---|---|---|
| **F3a** accuracy | within 1 pt (≥89.62%) | **67.25% (−23.4)** | **63.50% (−27.1)** | **FAIL** |
| **F3b** think | ≤770 tokens | 215 ✓ | 173 ✓ | **PASS** |
| **F3b** answer | in band around 291 | 180 (−38%) ✓ | **19 (−93%)** ✗ | r1 pass / r2 **FAIL** |
| **F3c** entropy | above audit baseline | not measured on fresh rollouts | | **UNMEASURED** |
| **F3d** form | g ≥ 99% | **94.25%** | **96.25%** | **FAIL** |

**Round 1 is the better checkpoint on every axis that matters** — 3.75 pts more
accurate, answers intact at 180 tokens against round 2's collapse to 19. Round 2
compressed further and adopted the register harder (markers 0.79 → 1.70) and paid
for both in accuracy. **The packet's two-round prescription is contraindicated by
this run: round 2 is strictly worse.**

Compression itself worked: think 1,540 → 215 is **7.2×**, far past F3b's bar. The
stage compresses; it does not preserve accuracy.

> **Finding 14 — the register has no backtracking escape hatch, and that is what
> the degenerate loops are.** 3.25–3.5% of generations never emit `</think>` and
> run to the full 32,768-token cap. **Those 3.5% consume 77.7% of the decode
> budget** (458,752 of 590,387 tokens; mean tokens/generation 1,476 with them,
> 341 without) — the entire cause of the 13–17 h evals.
>
> Classification is unambiguous **degenerate repetition**, in three modes, all
> opening correctly in register and none ever closing:
> * **announce-but-never-execute (8/14)** — `let's find the total number of
>   kittens from both cats` repeated **2,729 times**, byte-identical
> * **runaway `chk:` chain (3/14)** — a *single line* of 35,085 characters,
>   `chk: 3·60=180 ✓; 0.5·30=15 ✓; …` cycling with period 42, 2,506 `✓`
> * **degenerate `case N:` enumeration (3/14)** — reached `case 713:`, every body
>   byte-identical
>
> **`Wait`/`wait` appears zero times across all 14.** The compression stripped the
> backtracking token a thinking model uses to break out of a bad line. The card
> teaches `goal:`, `⇒`, `chk:`, `case` and nothing meaning *"this isn't working,
> back up"*. Consistent with everything else measured: the loops fire on **hard**
> problems (surviving sibling correct 29% vs 67% on clean), are **stochastic not
> problem-determined** (0/200 problems had all K fail; 14 observed vs 13.8
> expected under independence), and are **near-absorbing** (~26% escape).
> **This is a register-design finding and it belongs to P3a/the card, not to
> Stage B's loss.**
>
> The other 96% are clean: g=1, leakage 0.0%, and the register does exactly what
> it was designed to do.

> **Finding 15 — `verify.py`'s suffix fallback inflates degenerate models 14×
> more than the baseline.** `verify_response` ends its cascade with
> `npred.endswith(ngold)` / `ngold.endswith(npred)`, so **gold `200` with
> prediction `0` grades correct**, as do gold `90`/pred `0` and gold `2`/pred
> `42`. Separately `_strip_think` **falls back to the whole text** when
> `</think>` is absent, so a runaway generation has its entire scratchpad mined
> for `\boxed{}` — one candidate was graded correct on the extracted string
> `"let u = number of photographs in Jamal's phone = 6"`.
>
> Measured inflation: **baseline +0.27 pts (29 / 10,552) vs round-2 student
> +3.75 pts (15 / 400) — a 14× difference in rate.** The leniency systematically
> rewards exactly the degeneracy this student has, so it **narrows** the measured
> gap; strict grading widens it from 23.4 to 25.2 pts (round 1) and 27.1 to 30.4
> (round 2).
>
> **`verify.py` was NOT modified** — CLAUDE.md marks it kept-unchanged, leniency
> belongs in `whetstone/reward/`, and altering it would retroactively move P5's
> F2 numbers. But every figure this project quotes is affected, the baseline card
> included (90.49% → 90.21% strict on the full suite). **P8 owns the decision;
> until then report both.**

## Conclusion

**F3 FAILS.** Stage B assimilates the register and compresses think length 7.2×,
with entropy restored rather than collapsed — and loses **23–25 accuracy points**
doing it. The gate needed 1.

What is established:

1. **The ZPD band-pass cannot install a specified register unaided.** Its entry
   token is unreachable by construction, no γ fixes it, and deleting or
   re-dressing the token does not help because the cost is **positional**
   (findings 7, 9, 10). The register-card whitelist floor is the working fix and
   is the only one of three tested that works.
2. **Two rounds are worse than one.** Round 2 is worse on accuracy, answer length
   and entropy. The packet's prescription should be revised to a single round
   pending evidence otherwise.
3. **The accuracy loss is not primarily the loops.** They are 3.5% of generations;
   the gap is 23–25 points. Compressing 1,540 think tokens to 215 costs the
   reasoning itself. **This is the finding for P7/design review.**
4. **The register lacks a backtracking primitive**, and that omission is
   sufficient to explain the non-terminating tail (finding 14). A card revision
   adding an explicit backtrack marker is the indicated experiment, and the card
   deserves its own bake-off (the 008 precedent).
5. **Loss and entropy cannot detect any of this** (finding 11). Only generative
   evaluation can.

**Next packet should not proceed to P7.** F3's accuracy floor is the gate that
matters and it is missed by 23 points, not by a margin that a hyperparameter
sweep closes.

### Artifacts

| what | where |
|---|---|
| baseline card | `/data/whetstone/eval/baselines/qwen3-1.7b-original/CARD.md` |
| round-1 student (**best**) | `/data/whetstone/ckpt/stageb/golden/round1/final` |
| round-2 student | `/data/whetstone/ckpt/stageb/golden/round2/final` |
| arm B (refuted) | `/data/whetstone/ckpt/stageb/golden_nogoal/round1/final` |
| screening evals | `/data/whetstone/runs/009/_diag_r1/`, `_diag_r2/` |
| corpora + gates (4 variants) | `/data/whetstone/corpora/stageb/{golden,golden_nogoal,golden_okay,control}/` |

### Not done

- **Control arm (Part 6)** — corpus and gates built (11,954 traces / 3,994
  problems, weight sum 3,994.0), never trained. With F3 failing on the golden
  corpus by 23 points, judge-filtering is not the binding constraint.
- **F3c on fresh rollouts** — `entropy_audit.py` on the student, not run.
- **Full-protocol evals** — impractical until the loops are fixed (finding 14).
- **Primary benchmark suites** — deferred by user direction; moot while F3 fails.
