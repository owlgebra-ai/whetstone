# Stage-A → Stage-B handoff: the CERTIFIED corpus

**Designated Stage-B input (user decision, 2026-08-05):**

    /data/whetstone/corpora/stagea_golden/golden_faithfulness.jsonl
    2,414 problems · one certified trace each · 750,087 think tokens

Full context: `activity/008-stagea-teacher-corpus.md` (findings 10b–10d).

## What "certified" means here

Every trace satisfies all four, simultaneously:

| property | how |
|---|---|
| well-formed | segment gate g=1 via `whetstone/round0.py` (the shared construction) |
| verified correct | deterministic `verify_response` on the answer segment |
| in the compact register | selection ranks register adherence first |
| **faithful to its own verbose source** | GLM-5.2, faithfulness rubric |

The fourth is what no other corpus in this project has. The rubric forbids
dropping a step's value, a case split, a rejected branch, or a self-correction;
it permits dropping narration, self-talk and repetition.

## Level distribution

| level | problems | share | eligible pool | yield | think tok (median) | mark/100ch |
|---|---|---|---|---|---|---|
| 1 | 844 | 35.0% | 853 | 98.9% | 126 | 3.49 |
| 2 | 5 | 0.2% | 5 | 100% | 219 | 2.20 |
| 3 | 88 | 3.6% | 92 | 95.7% | 252 | 2.56 |
| 4 | 105 | 4.3% | 110 | 95.5% | 392 | 1.66 |
| 5 | 327 | 13.5% | 353 | 92.6% | 390 | 1.67 |
| 6 | 494 | 20.5% | 546 | 90.5% | 402 | 1.58 |
| 7 | 234 | 9.7% | 276 | 84.8% | 370 | 1.65 |
| 8 | 235 | 9.7% | 314 | 74.8% | 332 | 1.51 |
| 9 | 82 | 3.4% | 125 | 65.6% | 256 | 1.11 |

**By think tokens, not problems:** level 1 is only **15.2%** and levels ≥6 are
**56.3%**. Uniform sampling over problems and uniform sampling over tokens give
very different curricula here — choose deliberately.

"Yield" is against the *eligible* pool (problems that have a verbose source).
The 260-problem shortfall is 200 exhausted + 2 unjudged, so yield measures the
teacher, not the judge budget.

## Fields Stage B needs

* `compact_think` / `answer` — rebuild the sequence with
  `whetstone.round0.build_completion_text`. **Do not** re-split `raw_text` on the
  decoded string; it does not round-trip at the `<think>` boundary.
* `verbose_think` — the source trace. Present for every record in this file, by
  construction.
* `think_surprisal_hist` + `surprisal_bin_edges` — per-draft histogram of
  student-side surprisal, the **ZPD sizing input**. ⚠ Measured under `scorer_v1`,
  which has had 91% of the register style tax removed, so it is a **lower bound**
  on the masked fraction. **Re-measure under the original checkpoint before
  pinning γ.**
* `judge_verdict` / `judge_rubric` / `judge_note` — provenance of the
  certification. Evaluation metadata; not a training weight.
* `g_spike_b5` / `g_spike_b10` / `g_budget` — selection inputs, kept for audit.
  **Not training weights** — see the warning below.

## Rules that will silently corrupt Stage B if ignored

1. **One trace per problem here, so per-problem weighting is automatic** — but
   if you mix in the unfiltered corpus (`stagea_selected/selected.jsonl`, 1–3
   traces per problem), weight `1/n_kept` or sample one per problem per epoch.
   `n_kept` is the teacher's sampling luck, not the problem's value.
2. **The student starts from the ORIGINAL checkpoint, never `scorer_v1`.**
   Round 0's EMA copy belongs to Round 0; Stage B builds a new one.
3. **Recompute scorer gates after every assimilation round.** Stale gates are a
   named drift failure.
4. **Do not use G_spike as a training signal in the hard band.** Measured
   against 5,955 judged drafts, faithful-vs-wrong AUC decays 0.800 (L1) → 0.633
   (L6) → 0.541 (L9), and at level 9 the underlying `d_t` statistics *invert*:
   faithful traces are more surprising to the student than fabricated ones,
   because honest hard reasoning contains steps a 1.7B cannot anticipate while
   confabulation is fluent and predictable.

## Known deficiencies, carried forward deliberately

* **Level 9 is weak on every axis** — 82 problems at a 65.6% yield, the lowest
  marker density (1.11/100ch) and shorter traces than level 6 despite harder
  problems.
* **Compression is flat in absolute terms**: compact output stays 126–402 tokens
  while verbose input grows 7.5× across levels, so hard problems are being
  *truncated* rather than compressed. Faithfulness filtering does not remove
  this.
* **This corpus is judge-filtered**, which the pipeline otherwise forbids
  (`faithfulness_audit.py`: "if a judge verdict ever becomes a filter on
  training data…"). Attested in activity 008. **The unfiltered corpus at
  `/data/whetstone/corpora/stagea_selected/` is the control arm and must not be
  deleted** — the difference between Stage-B runs on the two is itself a
  measurement.

## Alternatives, if this set proves too small

| corpus | problems | traces | certification |
|---|---|---|---|
| **certified (this file)** | **2,414** | 2,414 | verified + faithful vs source |
| golden, incl. self-contained rubric | 3,652 | 3,652 | verified + sound, two different rubrics |
| unfiltered selected | 3,994 | 11,954 | verified only |
