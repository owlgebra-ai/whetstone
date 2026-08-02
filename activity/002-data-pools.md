# 002 — P1: data pools and eval suites (DeepMath-103K + GSM8K, SCA-matched evals)

- **Packet:** [packets/P1-data-pools.md](packets/P1-data-pools.md)
- **Status:** in-progress
- **Machine(s):** mac (code), spark (builds — CPU only)
- **Code commit(s):** _(filled per run)_
- **Started / finished:** 2026-08-01 → _

## Goal

Replace the v1 pool (openr1 / nemotron) with the v2 pool per design §12.7: DeepMath-103K
(main, difficulty-labelled) + GSM8K (easy tier), the SCA-comparison three-stage curriculum,
the seven SCA-matched eval suites, `standard_eval_300`, and a train↔eval contamination
report. Everything downstream reads these files, so the acceptance checks are strict.

## Decisions taken before running (deviations from the packet — read these)

1. **`_uid` recipe changed from v1's actual code, per the packet's explicit schema.**
   The packet specifies `"<source>:<sha8-of-normalized-prompt>"` but also says "copy the v1
   function as-is". Those contradict: v1's `_uid()` hashes the **raw** prompt and appends the
   **row index** (`f"{source}:{sha1(prompt)[:8]}:{idx}"`), which is unstable under any
   re-shuffle of the source dataset. The v2 pool is built from scratch from different
   sources, so no v1 resume file survives anyway. **Chosen: the packet's stable recipe**
   (`sha1(whitespace-collapsed prompt)[:8]`, case preserved, no index). Uniqueness is
   guaranteed by the pool-wide dedup on the same normalized key. Recipe now lives in one
   place, `whetstone/poolutil.py`, so the three builders cannot drift apart.

2. **Metadata goes in sidecar `*.meta.json`, not a JSONL header line.** The packet asks for
   a "header line" carrying pinned revisions / `grading: code-exec-pending`, but the
   definition of done requires `head -1` of every JSONL to parse *and match the record
   schema*, and every existing reader (`harvest.py`, `run_eval.py`, `sft_train.py`)
   json-loads every line into a record. A header line would break both. Each JSONL therefore
   gets `<name>.meta.json` next to it, plus the aggregate `pool_stats.json` /
   `eval_stats.json`. To protect the intent of gotcha 4, **every HumanEval record itself**
   also carries `"verifier": "code-exec", "grading": "code-exec-pending"` — stronger than a
   header line, since a record-level flag reaches the eval runner.

3. **DeepMath `difficulty` is a float, not an int** (values like `4.5`, in 0.5 steps) —
   the packet says int. Stored `level = round-half-up(difficulty)` (so `4.5 → 5`,
   deterministic, not Python banker's rounding) and kept the raw value in an extra
   `difficulty` field, which the SCA arm's low/high bands use.

4. **SCA stage-3 "low"/"high" bands** are undefined in the packet and design §12.7. Chosen:
   low = difficulty ≤ 4 (same as stage 2), high = difficulty ≥ 7.

5. New shared module `whetstone/poolutil.py` (uid/normalize/dedup/stratify/jaccard/IO) rather
   than triplicating those helpers across three scripts. v1's `_stratified_sample` machinery
   is carried over unchanged.

## Runs

_(in progress)_
