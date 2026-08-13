# Phase-2 training pool — dataset provenance (activity 011)

Canonical copy: `activity/assets/011/pool_provenance.md` (this file, versioned).
A duplicate sits at `/data/whetstone/data/pool/PROVENANCE.md` beside the data.

The pool file is `/data/whetstone/data/pool/phase2_pool.jsonl` — **9,350 rows**
at close of activity 011. Schema: `_uid / prompt / ground_truth / level /
source` (+ `year` on the AIME/AMC-12 rows). Every addition passed **two
contamination gates** — (1) exact-normalized text match against every eval
suite in `/data/whetstone/eval/` AND the pre-existing pool; (2) the P1 8-gram
near-duplicate gate (`scripts/check_contamination.py --apply`, threshold 0.8)
— plus source-specific metadata exclusions noted below. Reports:
`/data/whetstone/data/pool/contamination_report.json`, pre-removal backups
under `/data/whetstone/data/pool/_backup/`.

| source | uid prefix | rows | origin | census status |
|---|---|---|---|---|
| gsm8k | `gsm8k:` | 2,000 | P1 pool (activity 002), level 1 | measured (phase2_endpoint) |
| deepmath | `deepmath:` | 2,000 | P1 pool (activity 002), levels 2–10 | measured (phase2_endpoint) |
| hendrycks_math | `math:` | 3,780 | HF `EleutherAI/hendrycks_math`, **train split only**, all 7 subject configs | measured (phase2_endpoint) |
| aimo_amc | `amc:` | 43 | HF `AI-MO/aimo-validation-amc` | measured (phase2_endpoint) |
| aime_hist | `aimeh:` | 850 | HF `gneubig/aime-1983-2024` | measured (phase2_aime); 619 rows re-rated by injection at the cap raise |
| amc12 | `amc12:` | 677 | **User-curated** AoPS scrape (`~/Claude/Projects/whetstone/amc_problems`) | **uncensused** (nominal p̂ 0.375) |

## Per-source detail

### hendrycks_math (3,780) — added 2026-08-07, `scripts/build_phase2_additions.py`
- Train split only (the test split is MATH-500's source). Draw: **all** L4
  (1,285) + **all** L5 (1,733) + 467 of L3 + 300 of L1–L2 (seeded), keeping
  native MATH levels 1–5.
- Filtered: 1,024 duplicates already in `train_30k` (DeepMath derives partly
  from MATH), 50 eval-exact, 75 with no extractable `\boxed{}` answer, 5
  MATH-500 **near**-duplicates caught only by the 8-gram gate.
- Gold = last `\boxed{}` of the official solution.

### aimo_amc (43) — added 2026-08-07, same builder
- The dataset mixes AMC 2022 **and 2023**; the exact gate removed 40 rows —
  the AMC-2023 twins of the `amc23` eval suite. **Only AMC-2022 trains.**
- Golds were float-formatted ("5.0"); **int-normalized 2026-08-09** (the
  as-scored suffix diagnostic mis-graded pred "0" against gold "5.0").
  Level 6 (reporting label; curriculum is p̂-driven).

### aime_hist (850) — added 2026-08-07, `scripts/build_aime_additions.py`
- Hard **year filter ≤ 2023** (aime24 eval = AIME 2024; aime25 out of the
  dataset's range) + both gates. Filtered: 14 year-2024, 1 eval-exact, 62
  duplicates (DeepMath/MATH contain AIME problems), 5 near-dups (4 MATH-500,
  1 standard_eval_300).
- 1 row later dropped (`aimeh:15358d4e`, gold "080 or 081 (both were
  accepted)" — unmatchable under exact grading): 851 → 850. Level 8.
- At the cap raise (global 900), the 619 cap-sensitive aimeh rows were NOT
  re-censused; their 0/8 members entered 2c-ii as `uncensused_promotion`
  (nominal p̂ 0.125) — see below.

### amc12 (677) — added 2026-08-09, user-curated set
- Source: AoPS-scraped AMC 12, 2000–2025, key-verified (deterministic
  answer-key parse; 529/533 solution cross-check; **99.2% independent blind
  re-solve**; 4 key-conflicts quarantined upstream). Main split only —
  749 integer-answer, figure-free, self-contained problems.
- **ALL of year 2023 excluded by metadata** (28 rows) — the `amc23` eval
  rewrote statements (`p/q` → `p+q`, paraphrases), so text gates alone cannot
  be trusted against paraphrased twins; year-level exclusion is
  paraphrase-proof. Plus 3 eval-exact, 37 pool duplicates, 4 MATH-500
  near-dups (8-gram). Years 2000–2025 minus 2023. Level 6.
- The AoPS `solution` column was **never copied** into the pool (no training
  need; licence caution per the dataset's own README).
- Entered 2c-ii as `uncensused_addition` at nominal p̂ 0.375 (near the
  measured 43.6% on aimo AMC-2022).

## Census tables (curriculum inputs)

| table | built under | sampler | covers |
|---|---|---|---|
| `stagec_buckets/phase1_init/` | Stage-B init | T=1.0/1.0, cap 12,288 | original 4,000 |
| `stagec_buckets/phase2_endpoint/` | g400 ckpt | same | 7,823 (orig + math + amc) |
| `stagec_buckets/phase2_aime/` → merged `phase2_full/` | g400 ckpt | same | + 851 aimeh |
| `stagec_buckets/phase2c_injected/` | table edit, no GPU | — | + 867 `uncensused_promotion` (ex-0/8 cap-sensitive) + 677 `uncensused_addition` (amc12), nominal p̂ |

Uncensused rows carry their flag in the table so any later analysis can
separate measured from nominal ratings. Standing rules: benchmark problems
never enter training or rescue; every future source addition repeats both
gates + metadata exclusion for paraphrase-suspect sources; census and
rollouts must share a sampler for *measured* rows.
