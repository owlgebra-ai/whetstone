# P1 — Data pools and eval suites (DeepMath-103K + GSM8K, SCA-matched evals)

STATUS: done (activity 002)
MACHINES: spark preferred (CPU work; keeps turing free) — no GPU needed
DEPENDS ON: P0
BLOCKS: P2, P3
DELIVERABLES: train/val pool JSONLs, SCA-arm curriculum JSONLs, eval suite JSONLs, standard_eval_300, contamination report — all under /data/whetstone/, plus updated builder scripts committed.

## Objective

Replace the v1 pool (openr1-math / nemotron) with the v2 pool per design §12.7: **DeepMath-103K** as the main pool (verified golds + difficulty labels) with **GSM8K** as the easy tier. Build the SCA-matched eval suites so every later checkpoint is measured the same way. Everything downstream consumes these files; schema errors here poison every stage, so the acceptance checks are strict.

## Read first

- Design doc §12.7 (pool + eval protocol), §6 (baselines / reporting)
- `scripts/build_train_pool.py` and `scripts/build_eval_sets.py` (v1 versions — you are modifying, not rewriting; keep their stratification/dedup machinery)
- v1 procedure `trashed/WHETSTONE_PROCEDURE.md` §1 for the record-schema conventions
- Activity 001 gotchas + ROADMAP "Facts pinned by activity 001"

## Step 0 — sync the checkouts (activity 001 gotcha 6)

turing's clone lags the Mac (Mac-local commits; `pyproject.toml` and smoke scripts were scp'd, so its tree is dirty). Before anything: push `main` from the Mac, then on the box you work on: `git status` → stash/discard the scp'd duplicates → `git pull` → `uv pip install -p .venv/bin/python -e .` if pyproject changed. If working on spark, note it has no repo clone yet (only `~/workspace/whetstone-scorer` with the venv) — clone the repo there first and reuse the existing `.venv` or make one per P0 Part 3. Always `source .venv/bin/activate` before running scripts.

## Part 1 — Training pool

Modify `scripts/build_train_pool.py`:

- **Sources:** `zwhe99/DeepMath-103K` (fields: `question`, `final_answer`, `difficulty` ∈ [1,10], plus solution traces you should ignore) and `openai/gsm8k` config `main`, split `train` (answer is the text after `#### ` — strip commas and whitespace).
- **Schema (unchanged from v1, non-negotiable):** `_uid` = `"<source>:<sha8-of-normalized-prompt>"`, `prompt`, `ground_truth`, `level`. Map `level` = DeepMath `difficulty` (int); GSM8K → `level: 1`.
- **Sizes:** `n_train 30000 / n_val 2000`, stratified over levels so every difficulty band is represented; include ~20% GSM8K in the main pool as the easy tier. Seed 0. Pin the HF dataset **revision** for both sources and write it into the output header line and your activity file.
- **Outputs:** `/data/whetstone/data/pool/train_30k.jsonl`, `val_2k.jsonl`, plus `pool_stats.json` (per-level counts, source mix).

**Gotchas:**

1. **GSM8K gold extraction:** `#### 1,234` → `1234`. The `####` line sometimes carries units or `$` — strip to the bare numeric/string answer; run `whetstone.verify._normalize` on it and store the normalized form as `ground_truth`.
2. **DeepMath golds are strings that may be LaTeX** (`\frac{3}{4}`, intervals, sets). Do NOT try to normalize them to numerics at build time — store verbatim; the verifier handles normalization at compare time. Changing gold formatting here silently shifts verifier yield everywhere.
3. **Dedup within the pool** by normalized-prompt hash before sampling (DeepMath has near-duplicates of classic problems).
4. **`_uid` stability:** the sha8 must be computed on the *normalized* prompt (strip whitespace runs, lowercase NO — keep case, just collapse whitespace). v1 resume invariants key on `_uid`; if you change the hashing recipe, every resume file breaks. Copy the v1 function as-is.

## Part 2 — SCA-comparison curriculum arm

Separate files (used only by the SCA-baseline arm, design §12.7): three stages exactly —

- `sca_stage1.jsonl`: 2,000 GSM8K
- `sca_stage2.jsonl`: 1,400 GSM8K + 600 DeepMath difficulty ≤ 4
- `sca_stage3.jsonl`: 1,000 GSM8K + 500 DeepMath low + 500 DeepMath high difficulty

Disjoint from each other; sampled with seed 0; same schema. Put under `/data/whetstone/data/sca_arm/`.

## Part 3 — Eval suites

Update `scripts/build_eval_sets.py` to emit one JSONL per suite under `/data/whetstone/eval/`:

| Suite | HF source (verify current id before use) | Notes |
|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500` | main math suite |
| AMC23 | `math-ai/amc23` | 40 problems |
| MinervaMath | `math-ai/minervamath` | |
| AIME24 | `math-ai/aime24` | 30 problems |
| AIME25 | `math-ai/aime25` | 30 problems |
| GPQA-Diamond | `Idavidrein/gpqa` (diamond split) | multiple-choice — see gotcha 3 |
| HumanEval | `openai/openai_humaneval` | code — see gotcha 4 |

Same `_uid/prompt/ground_truth/level` schema (`level: 0` where a suite has no difficulty notion).

**Gotchas:**

1. **Tiny suites need many seeds, not one number.** AIME/AMC have 30–40 problems; the eval protocol (N=8, T=0.7, top-p 0.95, 32k max tokens — design §12.7) reports Pass@1 ± std over seeds. Nothing to do at build time except keep the suites intact — do not subsample them.
2. **Answer formats per suite differ** (MATH-500 gold is LaTeX; AIME is an integer 0–999). Store verbatim; the verifier normalizes.
3. **GPQA-Diamond is MCQ.** Store `ground_truth` as the letter (A–D) AND add a `choices` field; the eval prompt must instruct "answer with the letter in \boxed{}". GPQA is **gated on HF** — if `datasets` raises a gated-repo error, the user must accept the terms on huggingface.co and provide `HF_TOKEN`; flag in your activity file and continue with the other suites rather than blocking.
4. **HumanEval cannot be graded by `whetstone.verify`** — it needs code execution. Build the JSONL now (prompt = function stub + docstring, `ground_truth` = the test block) but grading needs a sandboxed executor; that lands with the eval-harness work in the ROADMAP. Mark the suite `"grading": "code-exec-pending"` in its header line so nobody accidentally reports verifier numbers on it.
5. **Do not touch `scripts/run_eval.py` defaults yet** — eval-protocol changes belong to the agent running evals (P2 onward) with the design §12.7 settings (`N=8, T=0.7, top_p=0.95, max_tokens=32768`, and Qwen3 `enable_thinking=True` in the template call). But DO check that `run_eval.py`'s prompt path uses `apply_chat_template` and does not hardcode a Gemma template — if it does, file that as a required fix in your activity journal (the fix itself belongs to P2's parser work).

## Part 4 — standard_eval_300 (internal dashboard set)

300 problems sampled from `val_2k.jsonl`, stratified by level, fixed seed 0 → `/data/whetstone/eval/standard_eval_300.jsonl`. This is the per-checkpoint auto-eval set (v1 continuity dashboard) — it must never change once created; treat the file as append-only history.

## Part 5 — Contamination check

Script `scripts/check_contamination.py` (new, commit it): for every (train-pool problem, eval problem) pair, compare normalized prompts — exact match after whitespace/case collapse, plus a cheap near-dup pass (e.g. 8-gram Jaccard > 0.8 on the first 400 chars). Remove hits from the **train pool** (never from eval), regenerate stats, and report counts per suite in the activity file. GSM8K-train vs GSM8K-test overlap and DeepMath vs MATH-500 overlap are the two known risk pairs.

## Definition of done

- [ ] All output files exist under `/data/whetstone/` with recorded row counts and per-level distributions.
- [ ] `head -1` of every JSONL parses and matches the schema; `python -c` spot-check that `verify_response(gold_wrapped_in_boxed, gold) is True` for 20 random rows per source (catches gold-format mistakes immediately).
- [ ] Contamination report in the activity file; removed rows logged.
- [ ] Dataset revisions pinned and recorded.
- [ ] Modified builders + contamination script committed; activity file `NNN-data-pools.md` written; packet status flipped.
