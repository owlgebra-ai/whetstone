<!--
  WHETSTONE v2 — REGISTER CARD  ***DRAFT — AWAITING USER REVIEW***

  Staged by packet P2 Part 4; notation spec + exemplars 1–2 DRAFTED BY CLAUDE
  (2026-08-02) at the user's request — the user reviews and ratifies. THIS FILE
  IS THE ONE HUMAN DESIGN INPUT IN THE WHOLE PIPELINE (design §1, precondition
  2): the register is *specified, not discovered*.

  P3 IS BLOCKED until the Status line below says FILLED (user flips it after
  review; exemplars 3–8 are written once the notation is ratified).

  Where this file is consumed:
    * P3  — prompted chunkwise compression puts this card in the compressor's
            context; every output record stores this file's sha1 as provenance.
    * P5  — Stage A keeps the card in the TEACHER'S CONTEXT, never its weights
            (design §3.1). The card text is literally pasted into the prompt,
            so its length is a real cost: aim for ~1 page + exemplars.
    * P4  — the structural whitelist in §2 below seeds the register-token set R
            used by the Round-0 inoculation loss mask (design §12.3).
  The STUDENT never sees this card. The register reaches the student only
  through Stage-B weights.

  All symbols below were tokenization-checked against Qwen/Qwen3-1.7B
  @ 70d244cc on 2026-08-02 (bare and space-prefixed). ⚠ and S1:-style markers
  were REJECTED on token cost (3 tokens spaced); see §1.1 notes.
-->

# Register card — WHETSTONE v2 compact reasoning register

**Status:** DRAFT — notation + exemplars 1–2 drafted by Claude; awaiting user review. NOT yet FILLED.
**Author:** Claude (draft) / ⟨user⟩ (review + ratification)
**Date:** 2026-08-02
**Target model:** Qwen3-1.7B (feasibility tier), then Qwen3-4B-Thinking-2507 / Qwen3-8B

---

## 0. The one constraint that decides whether this works

**Shorter lines, not bigger jumps.**

The register is compressed *notation*, not compressed *reasoning*. Every step
that existed in the verbose trace must still exist in the compact one — it is
just written with fewer tokens. A rewrite that fuses three derivation steps into
one asserted result is a **leap**, and leaps are exactly what the Stage-A reward
term `G_spike` punishes (design §3.2): the frozen scorer reads an unfollowable
jump as a huge top1-vs-actual logprob gap, and the teacher gets driven away from
it with `β` weight.

Concretely, when writing exemplars:

- Style must be **executable by a 1.7B model**. Clever-but-dense human shorthand
  poisons Round 0.
- **Never elide the final numeric/symbolic result of a step.** Intermediate
  prose may go; intermediate *values* may not.
- Prefer dropping: hedging, restatement of the problem, self-talk ("Let me
  think…", "Wait, actually…"), re-derivations of the same fact, verbal framing
  of an equation that is written on the next line anyway.
- Prefer keeping: each equation-manipulation step, each case split, each check.

If you are unsure whether a rewrite is a compression or a leap, ask: *could a
1.7B model reproduce the next line from the previous one alone?* If no, it is a
leap.

---

## 1. Notation spec

### 1.1 Symbol vocabulary

Every symbol below is a **single Qwen3 token both bare and space-prefixed**
unless noted. Rejected on token cost: `⚠` (3 tokens spaced → replaced by `!`),
`∴` / `⇔` / `≡` (2 tokens spaced → use `⇒` / words), `S1:` markers (3 tokens).

| Symbol | Means | Example use |
|---|---|---|
| `⇒` | therefore / it follows that (closes a derivation) | `12+14+b=50 ⇒ b=24` |
| `→` | becomes / evaluates to / rewrite as (one transformation) | `3h → 180min`; `2x+5=19 → 2x=14 → x=7` |
| `;` | micro-step separator within one line | `45+30=75; +50=125; +25=150` |
| `✓` | check passed / branch confirmed | `chk: 12+14+24=50 ✓` |
| `✗` | branch ruled out / check failed (2 tokens after a space — use sparingly, line-end) | `per≈44.4≠50 ✗` |
| `!` | caution / constraint to respect (replaces ⚠) | `! x≠0` |
| `?` | value to determine / open subgoal | `base b?` |
| `case <cond>:` | open a case branch | `case right triangle:` |
| `chk:` | verification line (exactly one per check — no repeated re-checking) | `chk: 150+30=180 ✓` |
| `goal:` | compact restatement of the target (first line, optional) | `goal: minutes left` |
| `let` | variable introduction | `let L=12, R=L+2=14` |
| `sub` | substitute | `sub x=7: y=3·7−2=19` |
| `·` `×` `≤` `≥` `≠` `±` `∈` `Δ` `\|` | standard math, all single tokens | — |

LaTeX (`\frac`, `\sqrt`, `\sum`, …) remains legal inside expressions — the model
is native in it and the verifier normalizes it. The register compresses the
**prose between the math**, not the math itself.

### 1.2 Step-marker convention

- **One step per line.** The newline is the step boundary.
- **Top-level steps are numbered `1.` `2.` `3.` at line start** (2 tokens each;
  `whetstone/reward/extract.py` treats leading `^\d+\.` as a chunk boundary, and
  the v1 reward code counts numbered steps — numeric markers are the cheapest to
  instrument). Numbering does **not** restart inside a case.
- Short auxiliary lines (`goal:`, `let`, `chk:`, `case …:`) are unnumbered.
- Micro-steps inside a line are separated by `;`.

### 1.3 Equation-manipulation shorthand

- **Transformation chains** use `→` with at most one operation per arrow:
  `2x+5=19 → 2x=14 → x=7`. Annotate the operation in parentheses only when it
  is not obvious: `x²=52 → x=√52 (x>0)`.
- **Substitution:** `sub <binding>: <resulting expression>`.
- **Case split:** one `case <cond>:` line per branch; every branch **must** end
  in either a result (`⇒ …`) or a rejection (`… ✗`). No silently dropped
  branches — branch elimination is reasoning, and it stays.
- **Sub-result naming:** tag a line-final value with `(A)`, `(B)`, … and refer
  back by the bare letter: `hw total 150 (A)` … `180−A=30`.
- **Units:** drop mid-derivation, restate in the final `⇒` line if the problem
  asks for them.

### 1.4 What may be elided vs never elided

| May be elided | Never elided |
|---|---|
| Problem restatement, "understanding the problem" prose | Final numeric/symbolic result of each step |
| Hedging and self-talk ("Wait", "Let me double-check", "Hmm") | Every intermediate value in a derivation chain |
| Repeated re-derivations of the same arithmetic (keep exactly one `chk:` line) | Case branches and their verdicts (✓ result or ✗ rejection) |
| Verbal description of an equation that is written on the next line | Variable definitions (`let` lines) |
| Transitional prose ("Now let's move on to…") | The single verification `chk:` of the final answer |
| Markdown headers, bold, display-math scaffolding | Constraint notes (`! …`) that later steps depend on |

### 1.5 Answer segment

The register governs the **think segment only**. After `</think>` the model must
still produce a normal, human-readable final answer ending in `\boxed{…}` —
Stage C holds the answer segment to the original checkpoint with a forward-KL
term, and the deterministic verifier reads post-`</think>` only. No register
symbols in the answer segment.

---

## 2. Structural whitelist for R (auto-derived, confirm here)

The Round-0 register-token set R = {types with mean surprisal > 75th pct AND
across-occurrence std < median} ∪ **structural whitelist**. P4's
`build_register_tokenset.py` tokenizes the literal strings below (bare AND
space-prefixed) against Qwen3's vocab and dumps the ids.

```
⇒  →  ;  ✓  ✗  !  ?  ·  ≤  ≥  ≠  ±  ∈  Δ
case  chk  goal  let  sub
1.  2.  3.  4.  5.  6.  7.  8.  9.
```

Reference ids (Qwen3-1.7B @ 70d244cc, bare/spaced): `⇒`=144016/58703,
`→`=51018/11397, `✓`=143617/52375, `;`=26/2587, `?`=30/937, `!`=0/753,
`case`=5638/1142, `chk`=35896/39242, `sub`=1966/1186. P4 re-derives — these are
recorded for drift detection, not consumed.

---

## 3. Exemplars

Verbose sides are the model's own verifier-correct traces, staged in
[`register_card_exemplars_staged.md`](register_card_exemplars_staged.md).
Exemplars 1–2 below are **drafts for review**; 3–8 are written after the
notation is ratified (candidates 3–8 in the staged file: levels 3–6, algebra /
number theory / combinatorics / other).

### Exemplar 1 (staged candidate 2) — level 1, arithmetic

- **_uid:** `gsm8k:97f4db57` — **gold:** `30`
- **problem:** Porche has 3 hours for homework: math 45 min, English 30, science
  50, history 25, plus a special project. How much time is left for the project?
- **verbose trace:** 1,600 chars (staged file, candidate 2)

**Compact-register rewrite:**

    goal: minutes left for project
    let total=3h → 180
    1. hw: 45+30=75; +50=125; +25=150 (A)
    2. left: 180−A=30
    chk: 150+30=180 ✓
    ⇒ 30

Every intermediate sum (75, 125, 150) survives; the double- and triple-checks in
the verbose trace collapse to one `chk:` line; all self-talk is gone.

### Exemplar 2 (staged candidate 1) — level 1, geometry

- **_uid:** `gsm8k:87f4cb6f` — **gold:** `24`
- **problem:** Triangle perimeter 50; right side 2 cm longer than left; left =
  12 cm. Find the base.
- **verbose trace:** 7,719 chars (staged file, candidate 1 — mostly spent
  resolving whether "right side" implies a right triangle)

**Compact-register rewrite:**

    goal: base b?
    let L=12, R=L+2=14, L+R+b=50
    1. "right side" = name of third side, or right angle? test both
    2. case plain triangle: b=50−12−14=24
    3. case right triangle, legs 12,14: hyp=√(144+196)=√340≈18.4; per≈44.4≠50 ✗
    4. case hyp=R=14: b²=14²−12²=52; b=√52≈7.2; per≈33.2≠50 ✗
    ⇒ b=24
    chk: 12+14+24=50 ✓; R−L=2 ✓

Note what was **kept**: the interpretation ambiguity and both rejected branches
(that *is* the reasoning in this trace — eliding them would be a leap). What was
dropped: five rounds of re-litigating the same three cases in prose. 7,719 chars
→ ~340 chars with zero steps fused.

### Exemplars 3–8

⟨PENDING — written after user ratifies §1. Sources: staged candidates 3 (L3
algebra), 4 (L4), 5 (L5 algebra), 6 (L5 number theory), 7 (L6 combinatorics),
8 (L6). Note: no level 7–9 candidate exists under the 12k-char staging cap.⟩

---

## 4. Self-check before flipping Status to FILLED

- [ ] User has reviewed §1 (symbols, markers, shorthand, elision rules) and
      edited to taste.
- [ ] Exemplars 3–8 written in the ratified notation; FORMAT DEMO removed (done).
- [ ] 5–10 exemplars, spanning difficulty bands; ≥1 algebra, ≥1 combinatorics,
      ≥1 geometry (1–2 give geometry+arithmetic; 3/5 algebra; 7 combinatorics).
- [ ] Re-read each compact rewrite asking "could a 1.7B model produce the next
      line from the previous one alone?" — no leaps.
- [ ] No step's final value was dropped.
- [ ] §1.1 symbol table matches the symbols actually used in the exemplars.
- [ ] §2 whitelist matches §1.1.
- [ ] Card is ~1 page of spec + exemplars.
