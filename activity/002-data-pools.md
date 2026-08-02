# 002 — P1: data pools and eval suites (DeepMath-103K + GSM8K, SCA-matched evals)

- **Packet:** [packets/P1-data-pools.md](packets/P1-data-pools.md)
- **Status:** done
- **Machine(s):** mac (code), spark (all builds — CPU only, turing untouched)
- **Code commit(s):** `d6238f0` → `<this commit>` (builders), all runs below on the tip at run time
- **Started / finished:** 2026-08-01 → 2026-08-01

## Goal

Replace the v1 pool (openr1 / nemotron) with the v2 pool per design §12.7: DeepMath-103K
(main, difficulty-labelled) + GSM8K (easy tier), the SCA-comparison three-stage curriculum,
the seven SCA-matched eval suites, `standard_eval_300`, and a train↔eval contamination
report. Everything downstream reads these files, so the acceptance checks are strict.

## Decisions taken before running (deviations from the packet — read these)

1. **`_uid` recipe follows the packet's schema line, not v1's actual code.** The packet
   specifies `"<source>:<sha8-of-normalized-prompt>"` but also says "copy the v1 function
   as-is". Those contradict: v1's `_uid()` hashes the **raw** prompt and appends the **row
   index** (`f"{source}:{sha1(prompt)[:8]}:{idx}"`), which is unstable under any re-shuffle
   of the source dataset. The v2 pool is built from different sources, so no v1 resume file
   survives anyway. **Chosen: the packet's stable recipe** — `sha1(whitespace-collapsed
   prompt)[:8]`, case preserved, no index. Uniqueness is guaranteed by the pool-wide dedup
   on the same normalized key (verified: 0 collisions in 29,998 rows). The recipe lives in
   one place, `whetstone/poolutil.py`, so the three builders cannot drift apart.

2. **Metadata is in sidecar `*.meta.json`, not a JSONL header line.** The packet asks for a
   "header line" carrying pinned revisions / `grading: code-exec-pending`, but the definition
   of done requires `head -1` of every JSONL to parse **and match the record schema**, and
   every existing reader (`harvest.py`, `run_eval.py`, `sft_train.py`) json-loads every line
   into a record. A header line breaks both. Each JSONL therefore has `<name>.meta.json` next
   to it, plus aggregate `pool_stats.json` / `eval_stats.json`. To protect the intent of
   gotcha 4, **every HumanEval record itself** carries `"verifier": "code-exec"` and
   `"grading": "code-exec-pending"` — stronger than a header line, because a record-level
   flag actually reaches `run_eval.py`'s `_verdict()`, which already skips non-`whetstone.verify`
   records.

3. **DeepMath `difficulty` is a float, not an int** (values in 0.5 steps, e.g. `4.5`) — the
   packet says int. Stored `level = round-half-up(difficulty)` (so `4.5 → 5`; explicitly not
   Python's banker's rounding) and kept the raw value in an extra `difficulty` field, which
   the SCA arm's low/high bands use.

4. **SCA stage-3 "low"/"high" bands** are undefined in both the packet and design §12.7.
   Chosen: low = difficulty ≤ 4 (same threshold as stage 2), high = difficulty ≥ 7.

5. New shared module `whetstone/poolutil.py` (uid / normalize / dedup / stratify / jaccard /
   IO) instead of triplicating helpers across three scripts. v1's `_stratified_sample`
   machinery carried over unchanged.

6. **DeepMath is downloaded, not streamed** (2.1 GB across 10 parquet shards) into the shared
   `HF_HOME=/data/cache/huggingface`, and `select_columns` drops the three `r1_solution_*`
   traces before iteration — ~90% of the bytes, and the blindness invariant says they must
   never enter the pool.

## Runs

All on spark. Setup once:

```bash
ssh bajajra@192.168.1.253
mkdir -p ~/git && cd ~/git && git clone https://github.com/owlgebra-ai/whetstone.git
cd ~/git/whetstone && uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python datasets huggingface_hub   # datasets 5.0.1, CPython 3.12.12
```

spark had **no repo clone** before this packet (only `~/workspace/whetstone-scorer` with the
vLLM venv from activity 001). The P1 work is CPU-only, so it gets its own light venv —
no vLLM, no torch. Every run below is prefixed with:

```bash
cd ~/git/whetstone && source .venv/bin/activate
export HF_HOME=/data/cache/huggingface PYTHONPATH=$PWD HF_HUB_DISABLE_PROGRESS_BARS=1
```

### Run 1 — dry run of the pool builder (2026-08-01 22:40)

- command: `python scripts/build_train_pool.py --out_dir /tmp/p1dry/pool --n_train 200 --n_val 50 --limit 3000`
- result: OK end-to-end in 1m45s (first DeepMath download). Confirmed field mapping, GSM8K
  `####` extraction, stratification.

### Run 2 — full pool + SCA arm, first attempt (2026-08-01 22:46)

- command: as Run 3 below.
- result: 30,000 train / 2,000 val written, **but** the level histogram contained a
  `"-1": 1` stratum. Cause: one DeepMath row carries `difficulty = -1.0` (sentinel; the
  card claims [1,10]). 4 rows total are out of range.
- fix: `build_train_pool.py` now drops rows with `difficulty ∉ [1,10]` rather than clamping
  them — a clamp would silently push sentinels into level 1 and pollute the easy tier.

### Run 3 — full pool + SCA arm (2026-08-01 22:52) ✅

```bash
python scripts/build_train_pool.py \
  --out_dir /data/whetstone/data/pool \
  --sca_out_dir /data/whetstone/data/sca_arm \
  --n_train 30000 --n_val 2000 --seed 0
```

- config: `gsm8k_frac 0.20`, seed 0; **pinned revisions**
  `zwhe99/DeepMath-103K @ 5cf055d1fe3d7a2eb19719ac020211469736ae44` and
  `openai/gsm8k (main) @ 740312add88f781978c0658806c59bc2815b9866`.
- inputs: HF hub → `/data/cache/huggingface`
- outputs: `/data/whetstone/data/pool/{train_30k,val_2k}.jsonl` + `.meta.json`,
  `pool_stats.json`; `/data/whetstone/data/sca_arm/sca_stage{1,2,3}.jsonl`
- source dedup: gsm8k 7,473 read → 7,473 unique; deepmath 103,016 read → **101,846 unique**
  (1,170 near-identical classics removed by normalized-prompt hash).

### Run 4 — eval suites + standard_eval_300 (2026-08-01 22:52) ✅

```bash
python scripts/build_eval_sets.py --out_dir /data/whetstone/eval \
  --standard_eval_from /data/whetstone/data/pool/val_2k.jsonl
```

| suite | repo @ revision | rows |
|---|---|---|
| math500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2…` (test) | 500 |
| amc23 | `math-ai/amc23` @ `80815d37…` (test) | 40 |
| minervamath | `math-ai/minervamath` @ `ee46ddc4…` (test) | 272 |
| aime24 | `math-ai/aime24` @ `83a7f387…` (test) | 30 |
| aime25 | `math-ai/aime25` @ `563bb840…` (test) | 30 |
| gpqa_diamond | `Idavidrein/gpqa` (`gpqa_diamond`) @ `633f5ee8…` (train) | 198 |
| humaneval | `openai/openai_humaneval` @ `7dce6050…` (test) | 164 |
| standard_eval_300 | sampled from `val_2k.jsonl`, stratified by level, seed 0 | 300 |

- **GPQA was not blocked.** The repo is `gated: auto` on the hub, but the download succeeded
  with **no `HF_TOKEN` present anywhere on spark** — packet gotcha 3's fallback was not
  needed. AIME24 ships no `answer` field (only `solution: "\boxed{204}"`); the v1
  boxed-extraction path handles it and was kept.
- GPQA record shape: options deterministically shuffled with `random.Random(1000+idx)`,
  `ground_truth` = the letter, `choices` = the shuffled options, and the prompt carries
  "Give the letter of the correct option inside `\boxed{}`".

### Run 5 — contamination check (2026-08-01 22:53) ✅

```bash
python scripts/check_contamination.py \
  --train /data/whetstone/data/pool/train_30k.jsonl \
  --eval_dir /data/whetstone/eval --apply
```

- 30,000 train × 1,534 eval prompts, 0.8s (inverted 8-gram index over the eval side).
- **2 hits, both near-dups, both against `standard_eval_300`; zero against any real eval
  suite.** Removed from train only → **29,998 rows**. Pre-removal file kept at
  `/data/whetstone/data/pool/_backup/train_30k.precontam.jsonl`.

| kind | score | train `_uid` | eval `_uid` | prompt |
|---|---|---|---|---|
| near | 1.0000 | `deepmath:a5295b85` | `deepmath:7e1b1058` | `Evaluate the integral: \int_{-1}^1 dx/((e^x+1)(x^2+1))` |
| near | 0.9118 | `deepmath:8310cada` | `deepmath:4f0bc029` | roots of `x^3+3x+5=0` … |

Both are train↔val leakage inside DeepMath (standard_eval_300 is drawn from val), which is
exactly what the near-dup pass exists to catch. **Zero MATH-500 / AIME / AMC / Minerva /
GPQA / HumanEval overlap** — consistent with DeepMath's published decontamination.

- **Positive control** (the detector is not silently dead): 10 verbatim eval prompts, 10
  lightly-perturbed (8 words appended), 10 heavily-perturbed (every other word dropped),
  scored against the real eval dir → **10/10 exact caught, 7/10 near caught, 0/10 heavy
  false-positives**. The 3 misses are the shortest AIME/MATH prompts, where appending 8 words
  drops the 8-gram Jaccard below 0.8. So: the near pass is real but conservative on short
  prompts — it will not catch paraphrases, only close textual copies.
- **GSM8K-train vs GSM8K-test** (the packet's other named risk pair; GSM8K test is not one of
  the eval suites, so this was run as a supplementary check against a temp suite):
  **0 hits** → `/data/whetstone/data/pool/contamination_gsm8k_test.json`.

### Run 6 — acceptance checks (2026-08-01 22:57) ✅

```bash
python scripts/validate_data_artifacts.py \
  --paths /data/whetstone/data/pool /data/whetstone/data/sca_arm /data/whetstone/eval
```

`schema errors: 0` across all 14 JSONLs — every line parses (not just `head -1`), all four
required fields present with the right types, `_uid`s unique per file, no empty prompt/gold.
Gold round-trip `verify_response("</think>\n\boxed{gold}", gold)` **20/20 per source per
file**, HumanEval correctly skipped as `verifier=code-exec`.

Then the same round-trip over **every row**, not a sample:

| file | n | round-trip failures |
|---|---|---|
| train_30k | 29,998 | **0 (0.00%)** |
| val_2k | 2,000 | 0 |
| math500 / minervamath / amc23 / aime24 / aime25 / gpqa_diamond / standard_eval_300 | 500 / 272 / 40 / 30 / 30 / 198 / 300 | 0 each |

Zero gold-format defects anywhere, including DeepMath's LaTeX golds and GSM8K's normalized
integers. This is the strongest available evidence that the verifier's yield downstream will
not be capped by data-side formatting.

## Final artifacts

```
/data/whetstone/data/pool/
  train_30k.jsonl           29,998   (+ train_30k.meta.json)
  val_2k.jsonl               2,000   (+ val_2k.meta.json)
  pool_stats.json                    per-level counts, source mix, revisions, contamination block
  contamination_report.json          2 hits, with prompts
  contamination_gsm8k_test.json      supplementary GSM8K train-vs-test: 0 hits
  _backup/train_30k.precontam.jsonl  30,000 (pre-removal)
/data/whetstone/data/sca_arm/
  sca_stage1.jsonl  2,000   2000 GSM8K
  sca_stage2.jsonl  2,000   1400 GSM8K + 600 DeepMath ≤4
  sca_stage3.jsonl  2,000   1000 GSM8K + 500 DeepMath low(≤4) + 500 DeepMath high(≥7)
/data/whetstone/eval/
  math500 500 | amc23 40 | minervamath 272 | aime24 30 | aime25 30 |
  gpqa_diamond 198 | humaneval 164 | standard_eval_300 300   (+ .meta.json each, eval_stats.json)
```

Distributions:

- **train (29,998):** gsm8k 5,999 / deepmath 23,999. Levels
  `1:6002, 2:38, 3:767, 4:1510, 5:4833, 6:7488, 7:3788, 8:4303, 9:1256, 10:13`.
- **val (2,000):** gsm8k 401 / deepmath 1,599. Levels
  `1:401, 2:2, 3:51, 4:100, 5:323, 6:500, 7:253, 8:287, 9:83`.
- `train ∩ val = 0` uids.

## Notes the next agent must know

1. **The level histogram is heavily peaked at 5–8 and nearly empty at 2–3 and 10** (38 rows at
   level 2, 13 at level 10) — that is DeepMath's own difficulty distribution, preserved
   faithfully by proportional stratification. Any later stage that wants *equal* counts per
   level (curriculum bands, pass-rate stratification, level-stratified probes) must resample
   with replacement or merge bands; there is no level-2 or level-10 mass to draw on.
2. **The SCA arm is NOT disjoint from the main pool** — the packet only requires the three
   stages be disjoint from each other (verified in-script; the builder raises if not). Overlap
   with the main pool: stage1 1,872/2,000, stage2 1,462/2,000, stage3 1,148/2,000. This is
   intended (both arms train from the same base checkpoint on comparable data), but do not
   describe the SCA arm as held out.
3. **`standard_eval_300` is now frozen.** `build_eval_sets.py` refuses to regenerate it if the
   file exists. Never delete it; the whole point is cross-checkpoint comparability.
4. **`run_eval.py` needs a P2 fix (packet Part 3, note 5).** Checked: `_build_prompt()` uses
   `tokenizer.apply_chat_template(...)` with **no hardcoded Gemma template** — good. But it
   does **not** pass `enable_thinking=True`, and its defaults are v1's (`K=1, T=0.0,
   max_tokens=12288`). Design §12.7 requires `N=8, T=0.7, top_p=0.95, max_tokens=32768` and
   `enable_thinking=True` on every Qwen3 template call. **Required fix, owned by P2.**
5. **HumanEval is unscored.** Records carry `verifier: "code-exec"`, `grading:
   "code-exec-pending"`; `run_eval.py` already returns `correct: None` for non-`whetstone.verify`
   records, so it cannot silently produce a bogus number. The sandboxed grader is P8's.
6. **GPQA needed no token today** despite the hub marking it `gated: auto`. If a future
   rebuild hits a gated-repo error, accept the terms on huggingface.co and export `HF_TOKEN`;
   the builder already degrades to "skip this suite, keep the rest".
7. Rebuilds are cheap now — DeepMath and GSM8K are cached under `/data/cache/huggingface`
   (shared via NFS, so turing sees them too).

## Conclusion

P1 is complete and every acceptance check in the packet passes: all files present with
recorded counts and per-level distributions, zero schema errors, 100% gold round-trip on
**all** rows (not just the 20-per-source spot check), contamination measured (2 rows removed,
zero real-eval leakage, detector validated with a positive control), dataset revisions pinned
in the sidecars and above. P2 and P3 are unblocked. The one deliverable P1 hands forward as a
*required fix* is `run_eval.py`'s eval protocol (`enable_thinking=True` + §12.7 sampling),
which belongs to P2's parser work.
