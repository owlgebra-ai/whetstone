# 009 — P6 Stage B: assimilation SFT (ZPD band-pass + SED) and the F3 gate

- **Packet:** [packets/P6-stage-b-assimilation.md](packets/P6-stage-b-assimilation.md)
- **Status:** in-progress
- **Machine(s):** turing (baseline evals, training), spark (gate scoring), mac (code)
- **Code commit(s):** `b9ec970` → `7ec4859` (running)
- **Started / finished:** 2026-08-05 → —

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
- 10,552 generations; log `/data/whetstone/runs/009/baseline_gsm8k_test.log`.
- **Early reading from the 12-problem smoke (not a reportable number): median think
  ≈ 1,681 tokens on GSM8K, not 6,099.** The 6,099 figure (activity 003) is the
  native median on the *training pool* at T=0.9 — DeepMath-heavy and much harder.
  GSM8K is grade-school arithmetic and the model thinks far less on it. **F3b's
  goalpost on this suite is therefore ~50% of ~1.7k, not of ~6.1k**, which is
  exactly why the packet refuses to compare against a differently-measured baseline.

(result pending)

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

## Conclusion

(pending — F3 verdict)
