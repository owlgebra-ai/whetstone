# WHETSTONE Stage 5 — Reward and Penalty Function Design

Prescriptive guidance for designing (or iterating on) the DAPO reward function used in Stage 5 of the WHETSTONE procedure. Every rule here is derived from empirical DAPO rollout-investigation results across multiple reward-function versions (v4 → v4.7.x, Phase 1 and Phase 2). The failure modes observed and the mitigations that worked are catalogued below.

Read this document **before** authoring or editing a reward function. Also re-read it before promoting a reward-function change from smoke test to full training — several of the biggest regressions in the historical run came from patches that satisfied a unit test in isolation but violated an invariant in this document.

---

## 1. Scope

The reward function scores one rollout `τ` from the DAPO group:

```
r(τ)  =  r_acc(τ)  +  r_fmt(τ)
```

and DAPO applies a KL anchor `β · KL(policy ‖ sft_ref)` at the loss level, not inside `r(τ)`. This document specifies:

- The **tier structure** and **numeric magnitudes** of `r_acc` and `r_fmt`.
- The **structural sub-rewards** that make up `r_fmt`.
- The **penalty catalogue** — what to detect, what magnitude to apply, and what floors to preserve.
- The **verifier-hygiene invariants** that any reward function depends on to produce a coherent gradient (a broken verifier corrupts the reward signal in ways that no penalty tuning can compensate for).
- The **design principles** that make the difference between a reward that steers the policy and a reward that inverts its own gradient.

Out of scope: the KL coefficient, curriculum construction, DAPO clip ranges, and process-reward auxiliary signals (deferred to a future extension). Those are covered elsewhere in the procedure.

---

## 2. Core reward decomposition

### 2.1 Accuracy tiers (`r_acc`)

Three tiers, produced by the deterministic verifier acting on the post-`</think>` block only:

| Tier | Condition | Value |
|---|---|---|
| **Strict** | Verifier accepts AND post-`</think>` block is clean (bare numeric / `\boxed{}` / accepted terminal-answer template) | `1.0` |
| **Lenient** | Verifier accepts AND post-`</think>` block is verbose (gold appears inside a full sentence or multi-token prose) | `0.5` |
| **Wrong** | Verifier rejects | `0` |

**Non-negotiable invariants:**

1. **Strict acceptance MUST use post-`</think>` extraction only.** Do not scan the `<think>` body for gold substrings. Scanning `<think>` creates lenient false-positives on rollouts whose post-think is symbolic / non-numeric (e.g. `\cot A` for gold `1`, `\frac{3√82}{11}` for gold `3`).
2. **Lenient acceptance also MUST use post-`</think>` extraction only**, and additionally require the gold to appear in the *last-K* window of the post-think (or as the last-occurring numeric match). This prevents lenient false-positives on multi-answer markdown dumps that happen to contain the gold as one of many.
3. **The lenient-vs-wrong boundary is the biggest single lever.** Every failure mode below either widens or shrinks the strict-vs-lenient gap; if that gap collapses below ~0.20 the gradient becomes noise (see §5).

### 2.2 Format reward (`r_fmt`)

`r_fmt` combines a structural base with subtractive penalties:

```
r_fmt(τ)  =  r_struct(τ)  −  Σ_i  penalty_i(τ)
```

with a hard floor when the rollout is otherwise well-formed (`</think>` closure present):

```
r_fmt(τ)  =  max(0.10, r_struct − Σ penalties)     if </think> present
r_fmt(τ)  =  max(0.00, r_struct − Σ penalties)     otherwise
```

The `0.10` floor is load-bearing. Its purpose is documented in §5.1.

### 2.3 Recommended magnitude budget

The following gap between the top of the strict tier and the top of the lenient tier is the design target:

```
total(strict-clean)  −  total(lenient-verbose)   ≈  0.55  …  0.65
```

Anything less than ~0.45 fails to produce a consistently-signed gradient on verbose-prose problems. Anything more than ~0.75 turns the format channel into a noise floor over the accuracy channel and induces compactness-driven capability regressions (see §4.5).

Working numeric budget (validated end-to-end):

| Component | Range | Notes |
|---|---|---|
| `r_acc` strict / lenient / wrong | `1.0 / 0.5 / 0` | Fixed by §2.1 |
| `r_struct` base | `0.10 … 0.35` | Base structural score before penalties |
| Short-clean bonus (bare-numeric / `\boxed{}` + short post-think) | `+0.10 … +0.15` | Additive on top of base; gated on `acc ≥ 0.5` (see §4.4) |
| Soft length tail penalty | `0 … −0.30` | Linear in `|τ_think|` above 8k chars, saturating at 20k chars |
| Repetition penalty (soft) | `0 … −0.20` | Capped; floor at 0.10 when `</think>` present |
| Placeholder / sentinel penalty | `0 … −0.15` | Capped (see §4.2) |
| Post-think tail-repeat penalty | `0 … −0.15` | Per-repeat scaling with a cap |
| Post-think register-leakage penalty | `−0.05` | Fires on markdown headers / numbered-list leakage into the final block |

Total worst-case-strict-with-artifacts:

```
total  =  1.0 (strict acc)  +  max(0.10, 0.35 − Σ penalties)   ≥  1.10
```

which stays comfortably above verbose-lenient (`0.5 + 0.10 = 0.60`). This margin is the design contract — if any penalty combination can drive it below `verbose-lenient + 0.30`, the penalty is over-magnitude and the gradient can invert (see §5.2).

---

## 3. Structural sub-rewards

These build `r_struct`. The base pattern that worked: reward hierarchy on presence of terminal-answer commitment structure.

| Rule | Fires when | Contribution |
|---|---|---|
| `has_closed_think` | `</think>` appears in the completion | `+0.15` (baseline for well-formed rollouts) |
| `terminal_answer_boxed` | Post-think ends with `\boxed{X}` (or the base-model's native equivalent) | `+0.10` additional |
| `terminal_answer_bare` | Post-think final line is a bare numeric / symbolic answer, no prose | `+0.10` additional |
| `short_clean_bonus` | Chunk count ≤ ~25 AND terminal answer is bare/boxed AND `acc ≥ 0.5` | `+0.10 … +0.15` additional |
| `chunk_restart_present` | `<think>` contains ≥ 2 numbered chunk restarts (`1.`, `2.`, …) | `+0.05` (register-preservation signal) |

Design notes:

- The `chunk_restart_present` reward is **only appropriate for base models whose SFT prior emits a numbered compact register**. For base models whose native compact register uses different structure (e.g. `thought\n` + trailing `\boxed{}`), replace with the equivalent structural marker in that register. Whatever the marker, reward its presence — do not accidentally penalize it (see §4.7).
- The `short_clean_bonus` MUST be gated on `acc ≥ 0.5`. An ungated bonus makes "compact wrong" outrank "verbose correct" on the format channel (§4.4).
- The `short_clean_bonus` gate on chunk count is *more robust* than a gate on think-token length, because SFT-installed rigid `1./2./3.` templates can produce low-token-length rollouts with high chunk counts (or vice versa). Prefer chunk-count gating.

---

## 4. Penalty catalogue

Each entry: **what the failure looks like → detector → magnitude → floor/cap rule**. The failure classes come from ~280 empirical rollout observations across two DAPO phases.

### 4.1 Rumination-runaway / soft length penalty

**Failure shape.** Median think tokens climb steadily during training (past ~2× the SFT baseline). Post-solution thrash: after answer is derived, 30–50% of think tokens go to "Final Answer: X" / "End of thought process" / "Output complete" sentinel chunks.

**Detector.** Compute `|τ_think|` in characters (or tokens; be consistent).

**Magnitude.** Soft linear tail:

```
length_penalty  =  min(0.30, max(0, (|τ_think| − 8000) / 40000))
r_fmt  -=  length_penalty
```

**Rules.**
- Kicks in at 8k characters (below this, no penalty).
- Saturates at −0.30 at 20k characters. **Never a cliff** — cliffs cause mode collapse.
- Combined with the strict-tier `+1.0`, `total` for a 20k-char correct rollout is still `1.0 + max(0.10, base − 0.30) ≥ 1.10`, so length penalty cannot suppress strict-correct below verbose-lenient. This is the design contract from §2.3.

### 4.2 Chunk-repetition / placeholder penalty (soft, capped)

**Failure shape.** Same short chunk (`Final Answer: X.`, `End.`, `Done.`) emitted 3–100× in a row. Two variants: (a) inside `<think>` before closure, (b) in the post-`</think>` tail.

**Detector.** Count exact-string chunk repetitions where each chunk is bounded by chunkwise separator (`|` / `\n\n` / register-specific). Distinguish the count from the number of placeholder tokens.

**Magnitude.**

```
n_excess         =  max(0, n_placeholder_chunks − 2)
placeholder_pen  =  min(0.15, 0.05 × n_excess)
r_fmt  -=  placeholder_pen
```

**Rules.**
- **Cap at 0.15.** An uncapped `0.05 × excess` scaling makes strict winners with 100+ tail-filler chunks lose more than a full point of format, dropping `total` below verbose-lenient. This inverted the gradient on ~20–25% of strict winners in the historical broken-v4 run.
- **Floor at 0.10 when `</think>` present.** Even the worst legitimate strict-with-artifacts rollout keeps `r_fmt ≥ 0.10`, preserving the strict-vs-lenient gap.
- **The "escape-once" allowance (`n_excess = n − 2`)** is deliberate. Legitimate strict-correct rollouts frequently emit one restatement of the final answer as a natural closing gesture; only sustained repetition should trigger.

### 4.3 Repetition-loop penalty (n-gram / template)

**Failure shape.** Identical or near-identical chunks emitted 10–500× times; typically `<think>` is dominated by "Maybe D is… No.", "Try x=−32, y=2, z=1. −val = −val." enumeration loops that cap-hit at 16–27k chars with zero progress.

**Detector.** Two-stage:

1. **Exact-string repetition detector.** Same chunk text ≥ 10 times consecutive → trigger. **Threshold 10, not 5.** Threshold 5 catches legitimate tail-filler in winners and inverts the gradient there.
2. **Template-normalized repetition detector.** Strip numerics, LaTeX identifiers (`\triangle X`, `$X$`, `\boxed{}`), signs, and parenthesized arguments from the first N tokens of each chunk, then check for repeated normalized-template chunks ≥ 5 times consecutive. Catches cycling-candidate paraphrase loops ("Wait, is it possible the answer is X?" with X cycling).

**Magnitude.**

```
if exact_rep_detected OR template_rep_detected:
    r_fmt  =  max(0.10, r_fmt − 0.20)     # soft, floored
```

**Rules.**
- **Soft penalty with floor.** A hard zero on the rep-detected branch (as in early versions) killed the format signal on strict-correct rollouts that happened to end in "Final Answer: X" 6 times, dropping their `total` to `1.0` — exactly equal to verbose-lenient. That is not a punishment, it is gradient noise. The soft floor preserves the strict-vs-lenient gap.
- Exclude short placeholder-vocab chunks (`Final Answer: X`, `End.`, `Done.`) from the rep counter when the *total* chunk count is ≤ ~30. Only long structural chunks are meaningful for this signal.
- **Template normalization is required** to catch the cycling-candidate loops that dominated hard problems (loop content varies enough to defeat exact matching, but the template shape ("Is it possible the answer is X? No.") is uniform).

### 4.4 Compact-wrong protection (bonus gating)

**Failure shape.** A compact rollout with bare-numeric post-think but the wrong answer earns the short-strict bonus (fmt = 0.30–0.35), while a strict-correct rollout with tail-filler drops to fmt = 0.10 from repetition penalty. Compact-wrong outranks strict-correct on the format channel.

**Mitigation.** Gate the `short_clean_bonus` and any register-preservation bonus on `acc ≥ 0.5`. A wrong rollout receives base structural score only, never the bonus.

```
if acc >= 0.5:
    r_struct  +=  short_clean_bonus     # eligible
else:
    r_struct  +=  0                     # ineligible
```

This is a **behavioural gate**, not a magnitude change. Its purpose is to prevent the format channel from ever preferring a wrong compact answer over a correct verbose one.

### 4.5 Counter-restart / monolithic-think / chunk-count anomaly

**Failure shape.** Two opposite anomalies, both symptomatic:

- **Counter-restart penalty evasion.** Model segments `<think>` into 9–13 numbered passes each starting `1.` to dodge a naive "no more than 100 numbered steps" penalty.
- **Monolithic-think collapse.** A rollout emits ZERO chunk restarts and enumerates 100–300 steps in one uninterrupted block, or produces `chunk_count = 1` while peer rollouts in the same group cluster at chunk_count ≈ 10.

**Detector.**

```
max_numbered_step_in_rollout  >  50       # any rollout stepping past step 50
OR
n_chunk_restarts  ==  0  AND  |τ_think| > 6000     # monolithic long think
OR
chunk_count_variation_across_group  >  100         # one outlier vs peers
```

**Magnitude.** `r_fmt -= 0.10` per detector trigger, capped at −0.15 total.

### 4.6 Post-`</think>` answer repetition

**Failure shape.** Rollout emits correct answer `151`, closes `</think>`, then emits `151\n\n151\n\n151\n\n…` 7+ times.

**Detector.** Regex on the post-`</think>` block: `^([\d]+|[A-Z])(\n\n\1)+$` with match count ≥ 2.

**Magnitude.** `r_fmt -= 0.05` per repeated occurrence, capped at −0.15.

### 4.7 Post-think register leakage

**Failure shape.** Compact register meant to live inside `<think>` leaks into the post-`</think>` final-answer block: markdown headers (`**Base Geometry:**`), numbered-step structure (`1. Define…`, `2. Interpret…`), or bolded prefixes.

**Detector.** Regex on post-`</think>`: `\*\*[A-Z]` OR `^[0-9]+\.\s` at line start.

**Magnitude.** `r_fmt -= 0.05`. Fires at most once per rollout regardless of detector-hit count.

### 4.8 Sentinel-phrase rumination penalty

**Failure shape.** Post-solution thrash inside `<think>` — 3+ occurrences of "Final Answer:", "End of thought process", "Output complete", "(Self-Correction Log)", "(Final Output Generation)", "(Placeholder for formatting if needed)".

**Detector.** Regex count of `(Final Answer:|End of thought process|Output complete|\(Placeholder|\(Self-Correction|\(Final Output)` ≥ 3.

**Magnitude.** `r_fmt -= 0.05`. Additive with other penalties, subject to the 0.10 floor.

**Note.** This penalty is auxiliary to the length penalty (§4.1). Length penalty cannot punish sentinel rumination in *correct* rollouts because `acc = 1.0` dominates; this rule provides a small direct signal.

### 4.9 Cap-hit handling

**Failure shape.** > 25% of rollouts in a group hit `max_completion_length` without emitting `</think>`. Zero reward across the whole group → GRPO has no advantage signal to propagate.

**Prescription.** Two changes:

1. **Small positive reward for closing `</think>`** even without any structural content. `r_fmt += 0.02` unconditionally when `</think>` present. This creates a tiny gradient toward termination even on hopeless problems.
2. **Do NOT try to fix cap-hit collapse with reward alone.** It is primarily a curriculum / context-length problem. If a group has ≥ 4/8 cap-hits, either shorten `max_completion_length` to force termination, or drop the problem from the curriculum.

### 4.10 Think/post-think numerical contradiction penalty

**Failure shape.** Two distinct sub-classes, both observed:

- **Handoff failure.** Model derives `10^7` correctly inside `<think>`, then emits `11111111` as post-think final answer (a previously-rejected candidate). Post-think numeric contradicts the last numeric conclusion inside `<think>`.
- **Redemption recap.** Model derives WRONG inside compact `<think>` chunkwise, closes `</think>`, then RE-DERIVES correctly in verbose prose post-`</think>` — and the last-numeric verifier matches the recap value. This reinforces a "do compact-wrong-then-prose-correct-recap" attractor.
- **Late-think overwrite.** Model derives correct answer at chunk 7, then off-by-one propagates through later chunks and abandons the correct value ("CRITICAL CORRECTION" chunks after the correct answer was already found).

**Detector.** Extract the last numeric expression from the final ~200 characters of `<think>`. Extract the terminal answer from post-`</think>`. If they disagree symbolically, flag.

**Magnitude.** Two options, pick one based on how aggressive you want the mitigation:

- **Weak.** `r_fmt -= 0.05` on contradiction. Provides small negative gradient without inverting winners.
- **Strict-tier gate (stronger).** Require the last-numeric-in-`<think>` to match the post-think final answer before granting strict tier; otherwise demote to lenient. This treats "compact-wrong-then-prose-correct-recap" as capped at 0.5 accuracy.

**Rule.** The strict-tier gate is preferred when the redemption-recap attractor is observed. Otherwise the weak penalty is safer — a naive contradiction detector can misfire on legitimate arithmetic rewrites in the last-line.

### 4.11 Meta-pattern / hedged-guess finalizer (soft, discretionary)

**Failure shape.** Correct-by-luck finalizers earning full strict reward:

- `Common answer for such specific angle problems is often related to the angle itself divided by 2 or 4. $120/4 = 30$` (gold=30, acc=1.0).
- `227. Or 12. 228. Let's go with 24.` (lucky-correct on gold=24).

**Detector.** Regex on `<think>` tail (last ~500 chars): `Common answer|such problems often|usually the answer|Let's go with|Or [A-Z0-9]+\. Or`.

**Magnitude.** `r_fmt -= 0.10`. Discretionary — only enable if the failure mode recurs at rate > 5% across a few checkpoints. Reward-only mitigation cannot fully solve this class (it is a capability-selection problem better addressed by process-reward audit).

---

## 5. Design principles

Six principles that separate "reward that steers" from "reward that inverts". These are derived from the historical run's failure catalog; violate any one and the same failures reappear.

### 5.1 Never zero out the format channel on well-formed rollouts

If `</think>` is present, `r_fmt ≥ 0.10`. Any penalty that can drive `r_fmt` to zero on a rollout that closed `</think>` is over-magnitude and will destroy the strict-vs-lenient gradient on legitimate strict winners. Implement the floor at the *final* aggregation step, not per-penalty.

**Why it matters.** In the historical run, hard zero-out on the rep-detected branch dropped strict-correct rollouts with tail-filler to `total = 1.0`, exactly equal to verbose-lenient's `1.0`. GRPO got zero advantage on the strict winners. The soft floor (0.10) restored the gap and let the gradient point in the right direction again.

### 5.2 Cap every penalty independently AND check worst-case composition

Individual penalties are capped in §4. But *composed* penalties can still explode if you don't check the worst case:

```
worst_case_r_fmt  =  max(0.10, r_struct_base − length_pen − placeholder_pen − rep_pen − sentinel_pen − leakage_pen)
```

Compute this on paper. If `total(strict + worst_case_r_fmt) < total(verbose-lenient) + 0.30`, penalty magnitudes are too aggressive — the format channel can outweigh the accuracy channel on adversarial-but-legitimate strict winners.

### 5.3 Gate every bonus on accuracy

Structural bonuses (short-clean, register-preservation) must be gated on `acc ≥ 0.5`. Without gating, compact-wrong outranks strict-correct on the format channel, and GRPO reinforces "compact wrong" over "verbose correct" whenever the group happens to contain both.

### 5.4 The strict-vs-lenient gap is the design target

Track this metric per training step:

```
gap  =  mean(total | strict-correct-in-group)  −  mean(total | lenient-correct-in-group)
```

Healthy: `0.55 ≤ gap ≤ 0.65`. If gap < 0.45, the format channel is not doing its job — investigate before increasing penalties. If gap > 0.75, format is drowning accuracy — reduce penalty magnitudes.

### 5.5 Soft tails, never cliffs

Every threshold-based penalty should be a soft linear (or log) function of its trigger metric, not a step function. Cliffs cause mode collapse: rollouts either bunch just below the threshold or blow past it entirely. The length penalty in §4.1, the placeholder penalty in §4.2, and the rep-penalty magnitude in §4.3 are all soft. The rep-penalty *detector* is a threshold (min_repeats ≥ 10) because it must be binary; the *magnitude* it applies is soft.

### 5.6 Verifier hygiene precedes penalty tuning

A broken verifier corrupts the reward signal in ways that no penalty can compensate for. Before iterating on penalties, confirm every §6 verifier invariant holds on your data. In the historical run, ~8% of training rollouts were mis-scored by verifier bugs before v4.6; that mis-scoring corrupted GRPO advantages more than any penalty magnitude change could fix.

---

## 6. Verifier hygiene (mandatory patches)

The verifier is upstream of the reward function. Every reward mode below is required to hold, or the reward function's gradient becomes structurally noisy.

### 6.1 Post-`</think>` extraction only

Both strict and lenient extraction operate on the post-`</think>` block. Never scan `<think>` for gold. Lenient additionally requires the gold to appear as the *last-occurring* numeric/symbolic match in post-think, or within the last-K characters (K ≈ 500).

### 6.2 Numeric normalization

Verifier must canonicalize before comparison:

- **Thousands separators.** `10,000,000` ≡ `10000000`. Bidirectional.
- **Scientific / power notation.** `10^7` ≡ `10000000`. Also handles LaTeX `10^{7}`.
- **Negatives.** `-6` ≡ `-6` under any wrapping. Handle in the same code path as positives.
- **Decimal vs fraction.** `0.5` ≡ `\frac{1}{2}` ≡ `1/2` ≡ `\dfrac{1}{2}` ≡ `\tfrac{1}{2}`.
- **Percentage.** `20%` ≡ `20` (strip `%`, `\%`).
- **LaTeX spacing artifacts.** `14{,}400`, `14\,400`, `14\!400` all equivalent to `14400`.
- **Angle units.** `155^\circ` ≡ `155` when gold specifies degrees. Gate the stripping on gold-has-degree to avoid false positives.

### 6.3 Word-boundary numeric matching

Bare integer matching MUST enforce a trailing non-`\` boundary. Otherwise `24\sqrt{2}` matches gold `24` as a prefix and produces a strict false positive.

```
if gold_is_bare_integer:
    match_regex = r"\b" + re.escape(gold) + r"(?!\\)"
```

### 6.4 LaTeX-fraction exact match

`\frac{162}{5}` must strictly match gold `\frac{162}{5}` even without `\boxed{}` wrapping. The canonicalization from §6.2 handles decimal equivalents; this rule is about accepting the LaTeX form itself when it is byte-identical to gold.

### 6.5 MCQ value ↔ letter mapping

When gold is a letter (`A`, `B`, `(C)`, `**B**`), parse the prompt's options block to build a `letter → value` dictionary and accept rollouts that emit the value form ("300" for option B whose text is "300"). Without this patch, MCQ problems where the model correctly identifies the option's value but emits the value instead of the letter get 0/8 false negatives.

### 6.6 Set / interval canonicalization

For gold values that are sets or intervals (`x < a or x > b`, `(-∞, a) ∪ (b, ∞)`, comma-lists like `-1,-2,2`):

- Strip `x \in` prefixes and `\left/\right/\frac/\infty` wrappings.
- Convert inequality-disjunction forms to interval-union.
- **For comma-list golds ("find all solutions")**: split on commas on both sides, sort, compare as sets. Order-sensitive matching rejects reordered-but-correct answers.
- **Gate against thousands-separators**: `1,000` and `1,000,000` are single numbers (3-digit comma groups), not lists. Detect list vs number by checking whether any comma group is not a 3-digit group.

### 6.7 Prose-templated answer canonicalization (strict tier)

The strict extractor MUST accept common prose-templated finalizers that carry an unambiguous bare answer, otherwise the strict gate is structurally unreachable on entire problem classes (geometry word problems, plain-decimal-gold problems). Observed variants that recur across many training steps:

- `**Answer:** X` (markdown-bold with prefix)
- `**X**` (markdown-bold bare)
- `Answer: X` (plain prefix, terminal line)
- `Final Answer: X`, `Final result is X` (plain phrase)
- `The {quantity} is {N}.` (full-sentence numeric commit — e.g. "The height of the pyramid is 4.")

**Detector.** After the primary bare-numeric / `\boxed{}` extractors fail, run a fallback regex over the terminal line of post-`</think>`:

```
r"(?:\*\*)?(?:Answer|Final\s+Answer|Final\s+result\s+is|The\s+.+?\s+is)(?:\*\*)?\s*[:=]?\s*\**\s*([^\s.*][^*]*?)\s*\**\.?\s*$"
```

If the captured group parses to a value equal to gold under §6.2 numeric normalization, accept as strict.

**Rule.** This canonicalization is required specifically for the strict tier. Without it, ~10–30% of problem classes (geometry word problems, plain-decimal, single-integer with prose finalizers) have a structurally unreachable strict ceiling — GRPO cannot generate the strict-vs-lenient gradient on those problems and the compact register cannot be learned there.

### 6.8 Cosmetic LaTeX normalization

After the primary answer comparison fails, retry with cosmetic normalization:

- `\dfrac` / `\tfrac` → `\frac`
- `\text{X}` / `\mbox{X}` unwrapped
- `%`, `\!`, `\,` stripped
- Trailing radix subscript `_N` (N=2..36) stripped when gold has the subscript, matching bare digit-string. Gate on gold-has-`_N` or the prompt containing "base N".

**Do NOT strip unit words** ("5 minutes" ≠ "5 hours"). Unit stripping produced zero validated recoveries and multiple false positives in the historical run.

### 6.9 Backslash-eating boundary fix

When the model emits `\boxed{...}` and the completion serializer eats the `\b` as backspace (`\x08`), the extraction sees `oxed{...}`. Accept `oxed{X}` as `\boxed{X}` equivalent, OR fix the upstream serializer. Either mitigation is fine; do not ignore the pattern.

### 6.10 Malformed / duplicate `\boxed{}` detection

**Failure shape.** Model emits garbage terminal boxes: `\boxed{10}{boxed{10}}`, `$$\boxed{x}$${boxed{x}}`, double-box in one final block. Accuracy extraction survives (last-box parse), so `r_acc = 1.0`, but the rollout structurally deserves a format penalty.

**Detector.** Two triggers:

- `> 1` occurrence of `\boxed{` in the post-`</think>` final block.
- Any occurrence of the literal token `{boxed{` (evidence of escape-eaten duplicate).

**Magnitude.** `r_fmt -= 0.05` per detector trigger, capped at −0.10. Soft — do not zero.

### 6.11 Boundary consistency: `r_fmt` and `r_acc` see the same "final block"

**Failure shape.** `r_fmt` defines the final block as "text after last `\n\n`" while `r_acc` defines it as "walk-back from last `\boxed{}`". A rollout that boxes the correct answer *mid-completion* (then adds prose after) gets `r_acc = 1.0` but `r_fmt` scored on a different block, mismatched to the accuracy path.

**Rule.** The final-block extractor used by `r_fmt` MUST be the same function (or equivalent boundary rule) as the one used by `r_acc`. Otherwise the two channels disagree on which bytes constitute the "answer", and structural penalties fire on bytes the accuracy check did not consider.

**Test.** Emit a rollout that boxes the answer, then continues with prose. Confirm that `r_fmt`'s final-block matches the `r_acc`'s final-block byte-for-byte.

### 6.12 Symbolic equivalence (v5.0-class, out of scope for hand-rolled patches)

The following classes require sympy or numeric evaluation and should NOT be attempted as string normalizers:

- Radical equivalence: `\sqrt{52}` ≡ `2\sqrt{13}`.
- Trigonometric equivalence: `\arccos(-1/2)` ≡ `120°`.
- Mixed-number ↔ improper-fraction: `10\frac{1}{12}` ≡ `\frac{121}{12}`.

These are genuine false-negative classes but the false-positive risk of hand-rolled normalization outweighs the recovery. Defer to a sympy-based v5.0 verifier with sign-exact numeric equivalence (compound-radical golds have sign-flipped wrong-answer siblings that a sloppy normalizer will accept).

### 6.13 Multi-subproblem prompts

Prompts containing sub-question markers (`(a)`, `(b)`, `(i)`, `R1.15`, `R2.3`, …) are a curriculum bug, not a verifier problem. Detect at training-data-prep time and either upgrade the gold to a per-sub tuple with all-match requirement, or drop the prompt from the curriculum.

---

## 7. Diagnostic metrics to track per checkpoint

The following signals let you detect reward-function pathology before it accumulates over many steps. Emit these alongside every save-step:

| Metric | Healthy range | Failure interpretation |
|---|---|---|
| `strict_vs_lenient_gap` | 0.55 – 0.65 | < 0.45 → format channel not steering; > 0.75 → format drowning accuracy |
| `strict_winners_fmt_ge_0.10` | ≥ 90% | Lower means penalty stack is zeroing legitimate winners; check §5.1 |
| `max_wrong_fmt` | ≤ 0.30 | Higher means compact-wrong is earning bonuses it shouldn't; check §4.4 |
| `median_think_tokens` | ≤ 1.5× SFT baseline | Higher means rumination-runaway; length penalty (§4.1) not biting hard enough on `acc=1` |
| `p95_think_tokens` | ≤ 3× SFT baseline | Same |
| `cap_hit_rate` | < 10% | Higher means curriculum too hard for context budget or repetition attractor active |
| `verifier_false_positive_rate` (sampled) | 0 | Any nonzero rate corrupts GRPO advantages; audit verifier (§6) |
| `verifier_false_negative_rate` (sampled) | 0 | Same |
| `zero_advantage_group_frac` (all-8-same-acc) | < 20% | Higher means curriculum saturation; rebuild from current-checkpoint K=8 |
| `think_jaccard_5gram_mean` | ≤ 0.30 | ≥ 0.65 combined with `unique_frac ≤ 0.15` = entropy collapse (see below) |
| `entropy_collapse_flag` | never fires | If fires, halt training and inspect |

**Entropy-collapse rule.** Fire only when `think_jaccard_5gram ≥ 0.60` AND `final_answer_unique_frac ≤ 0.15`. Do NOT include `chunk_count_std` as a standalone trigger — it produces false positives on SFT-installed rigid numbered templates (where `chunk_count_std = 0` with high content diversity means the register is uniform but the reasoning is not collapsed).

**Semantic mode collapse warning.** Low lexical jaccard does NOT rule out semantic collapse. On combinatorial problems the model can commit 7/8 to the same wrong intermediate lemma while producing lexically diverse traces. Add a semantic-diversity signal: extract numeric/categorical claims from `<think>` and check agreement across rollouts. If 7/8 share the same intermediate claim despite low jaccard, flag.

**Single-integer-gold benign case.** `final_answer_unique_frac = 0.25` on a single-integer-gold problem (e.g. gold = 2, and 6/8 correctly emit "2") is a *convergence* signal, not a collapse signal. The entropy-collapse rule correctly does not fire here because `think_jaccard_5gram` stays low. Do not add uniq-alone alerts.

**Shared-setup jaccard inflation.** On geometry problems where the figure setup is forced (trapezoid, pyramid, canonical convex configurations), the first 30–50 tokens of `<think>` are nearly identical across all 8 rollouts because the setup is dictated by the problem. This inflates `think_jaccard_5gram_mean` into the 0.05–0.10 band even when mid-trace algebra is genuinely diverse. To avoid false collapse alarms on this class, optionally compute jaccard on the *post-setup tail* (drop the first ~100 tokens or the first 2 numbered steps) when the prompt contains a forced-figure setup.

**Chunk-restart count as format-health proxy.** Number of numbered-chunk restarts in `<think>` is a cleaner format-health signal than raw `<think>` character length. A rollout at 25k chars with zero chunk restarts is monolithic-think collapse (bad); a rollout at 25k chars with 10 chunk restarts is legitimate long reasoning (fine). Emit `min_chunk_restarts_in_group` and `max_chunk_restarts_in_group` per step alongside the other metrics.

**Confident-wrong vs. chaotic-correct.** Length alone cannot differentiate a short-confident-wrong rollout (clean, internally consistent under wrong premise, never self-doubts) from a long-chaotic-correct rollout (oscillates across many revisions, lands right). Track the ratio of these two classes per step: `confident_wrong = (short, no hedging tokens, terminal answer ≠ gold)` vs `chaotic_correct = (long, multi-revision, terminal = gold)`. If confident-wrong grows, the SFT init's "verify before commit" prior is decaying.

---

## 8. Validation checklist for reward-function changes

Before promoting any reward-function change from smoke test to full DAPO training:

- [ ] **Worst-case-strict-with-artifacts total ≥ verbose-lenient + 0.30** (§5.2). Compute on paper.
- [ ] **Every bonus is `acc ≥ 0.5` gated** (§5.3).
- [ ] **Every penalty has a documented cap** (§4).
- [ ] **The 0.10 floor is applied at final aggregation, not per-penalty** (§5.1).
- [ ] **No threshold penalty has a cliff** (§5.5). Confirm the magnitude is soft-linear in the trigger metric.
- [ ] **All verifier hygiene invariants (§6) verified on a stratified sample** of the training pool.
- [ ] **Smoke test on ~100 archived rollouts** exercises each penalty branch: legitimate tail-filler (should keep fmt ≥ 0.10); exact rep loop 10× (should trigger, floor at 0.10); exact rep loop 100× (same, still floored); template-varying paraphrase loop (should trigger); compact-wrong bare-numeric (should get base only, no bonus); compact-correct bare-numeric (should get bonus); verbose-correct (should get lenient tier only); strict-correct with 3+ sentinel phrases (should lose 0.05); post-think markdown headers (should lose 0.05).
- [ ] **Diversity-grouper metrics from §7** confirmed instrumented and dashboarded before launch.

---

## 9. Out-of-band items (not reward-shape fixes, but must be handled)

These recurring pathologies from the historical run are NOT addressable by reward tuning. Include them in the run's operating playbook alongside the reward function.

### 9.a Curriculum audit for prompt-truncation and hint-dependency

- **Multi-subproblem prompts** (§6.13) — detect at data-prep and either upgrade gold to per-sub tuple or drop.
- **Translation-note dependencies** — problems where a translation footnote is critical to gold recovery ("silently override literal `7-个长方形` as translation typo for `一个长方形`"). Half of rollouts override and succeed; half honor literal and fail. Trains the policy to ignore stated user constraints. Filter these out or the reward function will teach the wrong lesson.
- **Asy-figure vs. text inconsistency** — problems where an asymptote figure contradicts the text, and rollouts declare "indeterminate" then guess the small-integer gold. Spurious positives. Filter.
- **Explicit hint prompts** (`Suggestion:`, `Hint:`, `Show that…`) — model may ignore the hint and pursue brute force. Either strip the hint (for pure-reasoning fairness) or verify the SFT teacher trace on that problem actually followed the hint.

### 9.b Curriculum saturation

Groups where all 8 rollouts hit the same accuracy tier produce zero within-group advantage. Track `zero_advantage_group_frac`; if it exceeds ~20–25%, rebuild the DAPO curriculum by running K=8 from the *current* checkpoint (not the previous phase's) and re-selecting the 1–7/8 band. Do NOT try to fix curriculum saturation via reward changes.

### 9.c Genuine capability regressions vs. reward-shape failures

Some observed failure classes are capability regressions caused by compactness pressure and are NOT reward-shape fixable:

- Case-split dropping on parity-constrained combinatorics.
- Mod-3 prime-constellation sieve regressions.
- Telescoping-bound off-by-one (`1 − 1/n` vs `1 − 1/(n+1)`).
- Lower-bound fence-post bug (dropping factor of 2 in `2a ≥ e ⇒ a ≥ ⌈e/2⌉`).
- Hallucinated numerical approximation as substitute for algebra.
- Second-constraint blindness on multi-bound problems.
- Verification ritual without substance (re-substituting same buggy equations, declaring "Confirmed!").

If any of these recur across ≥ 3 checkpoints on the same problem class, the correct mitigation is either targeted SFT data augmentation for the failing micro-skill or a process-reward auditor (deferred to a future extension). Do NOT tune reward magnitudes further in response — you will over-fit the reward to a capability problem it cannot solve.

### 9.d Rollout logging invariants

The rollout logger must record the full prompt (or a hash + prompt tail), not a truncated prefix. If the logger truncates before the problem statement, every problem hashes to the same prompt and diversity/entropy metrics collapse into one artificial "group". Group by `(prompt, gold)` not by prompt alone.

---

## 10. Anti-patterns (do not do these)

Failed approaches from the historical run. Each is tempting; each broke the gradient.

### 10.1 Hard-zero the format channel on any single trigger

Any single failure detector that can drop `r_fmt` to `0` (or below the 0.10 floor) on a well-formed rollout will zero out legitimate strict winners that happen to trip the detector, exactly equalising them with verbose-lenient. GRPO gets zero advantage on cleanest behaviour. Always floor at 0.10 when `</think>` is present.

### 10.2 Uncapped `k × excess` scaling on any penalty

Multiplicative scaling with the trigger count (e.g. `0.05 × n_tail_filler_chunks`) can produce penalties of ≥ 5.0 on adversarial-but-legitimate rollouts, dropping strict-correct total to zero. Always cap the *magnitude* even when the *trigger* is unbounded.

### 10.3 Structural bonus not gated on `acc`

Bonuses awarded on structural features alone (chunk count, bare-numeric post-think) without checking that the answer is at least lenient-correct will make compact-wrong outrank verbose-correct on the format channel. Always gate on `acc ≥ 0.5`.

### 10.4 Length-only penalty as sole compactness signal

Length penalty (§4.1) is *necessary* but not *sufficient*. It cannot suppress rumination inside `acc = 1.0` rollouts because accuracy dominates the total. Sentinel-phrase (§4.8) and post-think tail-repeat (§4.6) penalties provide the missing direct signal.

### 10.5 Threshold at 5 on exact-string repetition detector

Threshold ≥ 5 catches legitimate tail-filler ("Final Answer: 3." emitted 5×) in strict winners and inverts the gradient. Use threshold ≥ 10 on exact-string detection; use template-normalized detection with threshold ≥ 5 for the cycling-paraphrase class separately.

### 10.6 Lenient extraction that scans `<think>`

Any lenient acceptance path that scans `<think>` for gold substrings produces false positives on rollouts whose post-think is symbolic / non-numeric. Restrict lenient to post-`</think>` extraction with a last-K window.

### 10.7 KL β = 0 (vanilla DAPO)

Removing the KL anchor lets the policy re-discover the verbose mode. Compactness collapses within ~35 training steps; median think tokens double, cap-hit rate rises to > 90%, accuracy collapses to < 30%. Keep `β = 0.005`. This is a Stage-5 configuration parameter documented in the main procedure, but it is part of the load-bearing reward stack — noting here for completeness.

### 10.8 Reward-function iteration without verifier audit

Every attempt in the historical run to fix a training-instability symptom by tuning penalty magnitudes without first fixing an upstream verifier bug produced smaller improvement than the eventual verifier fix. The verifier is upstream — patch it first, then observe whether the penalty tuning is still needed.

---

## 11. Iteration protocol

When a new failure mode is observed via rollout-investigation:

1. **Classify.** Is it a *verifier* bug (§6) or a *reward-shape* bug (§4, §5)? Verifier bugs corrupt the reward signal; reward-shape bugs steer the policy in the wrong direction. Fix verifier bugs first.
2. **Detect.** Write a detector for the failure mode. Confirm it fires on a curated positive set and does NOT fire on curated negatives (legitimate strict winners are the most important negative class — see §5.1).
3. **Magnitude.** Pick a penalty magnitude consistent with §2.3 and §5.2. Confirm the worst-case-strict-with-artifacts total stays above verbose-lenient + 0.30.
4. **Cap.** Every penalty has a cap (§4).
5. **Floor.** Preserve the 0.10 floor at final aggregation (§5.1).
6. **Smoke test.** Exercise the detector on archived rollouts covering every branch in §8.
7. **Deploy.** Restart DAPO from the SFT init (do NOT resume from a checkpoint trained under the broken reward).
8. **Instrument.** Monitor the §7 metrics for 5 checkpoints. If `strict_vs_lenient_gap` stays in the 0.55–0.65 band, the fix is working.
9. **Log.** Append the new failure class and mitigation to the reward-function version log with date, step, observed pattern, magnitude, and validation evidence.

---

## 12. Reward-function versioning

Keep the reward function under version control and tag every checkpoint with the reward version used to train it. When rollout-investigation surfaces a critical bug, deploy the fix as a new reward version and restart from the SFT init — do not resume from a checkpoint trained under a broken reward. In the historical run, this discipline distinguished successful recoveries (v4.1, v4.2, v4.6 all clean restarts) from failed patches (attempts to hot-swap reward mid-run confused the KL anchor and degraded training).

---

## 13. Summary

The reward function is a system of interacting tiers, structural sub-rewards, and penalties, held together by three invariants:

- **The strict-vs-lenient gap** (≥ 0.20 in tier magnitudes, ≥ 0.45 empirically after penalties) is the design target.
- **The 0.10 floor** on `r_fmt` when `</think>` is present prevents penalty stacks from zeroing legitimate strict winners.
- **Every bonus gates on `acc ≥ 0.5`**, so compact-wrong can never outrank verbose-correct on the format channel.

Everything else in this document is a specific detector, magnitude, or verifier patch that serves one of those three invariants. When any invariant is violated, GRPO's gradient goes noisy or inverts and the policy degrades in exactly the ways catalogued in §4.

The correct sequence for reward-function work is: **verifier hygiene → tier magnitudes → structural sub-rewards → penalty catalogue → smoke test → deploy → monitor**. Attempting them out of order produces the historical run's iteration count.
