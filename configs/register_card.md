<!--
  WHETSTONE v2 — REGISTER CARD  ***FILLED — RATIFIED 2026-08-02***

  Staged by packet P2 Part 4; drafted by Claude, validated by the P3a bake-off
  (activity 004, arm A), ratified by the user. THIS FILE IS THE ONE HUMAN
  DESIGN INPUT IN THE WHOLE PIPELINE (design §1, precondition 2): the register
  is *specified, not discovered*.

  Any future edit must preserve: zero literal think-tag strings (§1.6), no
  letter-tag sub-result naming (activity 004 runaway class), un-indented
  exemplar blocks, and identical elision rules to any variant card being
  compared against.

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

**Status:** FILLED — ratified by user 2026-08-02 (bake-off winner, activity 004; required edits applied; tokenizer-audited §1.6; 5 exemplars). Known gap: no true combinatorics exemplar (staged candidate 7 was mislabeled — no verifier-correct combinatorics trace existed under the 12k-char staging cap).
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
- **Sub-result naming:** refer back by **step number** (`from 2:`) or simply
  restate the value — values are short in this register. (The earlier `(A)`,
  `(B)` letter-tag scheme is **banned**: the bake-off showed the model exhausts
  letters and rolls into `AAA`/`BBB` runaway loops — activity 004, 10–18% of
  traces.)
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

The register governs the **think segment only**. After the think segment closes,
the model must still produce a normal, human-readable final answer ending in
`\boxed{…}` — Stage C holds the answer segment to the original checkpoint with a
forward-KL term, and the deterministic verifier reads only post-think text. No
register symbols in the answer segment.

(Deliberately not written with the literal closing tag here: the tag strings
tokenize as the REAL boundary tokens even in prose — a card containing them
injects boundary tokens into every prompt it rides in. See §1.6.)

### 1.6 Tokenizer audit (Qwen3-1.7B @ 70d244cc, 2026-08-02)

- Whole-card encode → **zero special/boundary-token injections** (after the
  §1.5 rewording; before it, the two literal closing-tag strings each encoded
  as real token 151668). **Rule: no literal think-tag strings anywhere in this
  file, ever.** Any pipeline step that pastes this card into a prompt should
  assert the rendered prompt contains no boundary-token ids outside their
  structural positions.
- Encode→decode round-trip is lossless — no UTF-8 normalization gotchas.
- Every §1.1 symbol is **1 token bare, line-start, and after a digit**; ✗ ≈ ² √
  cost 2 when space-prefixed but occur attached in practice (`b²`, `≈18.4`,
  `√(340)`). Unicode math ≤ ASCII everywhere (`²`=1 vs `^2`=2). ⚠ (3 tokens
  spaced) remains banned.

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

```
goal: minutes left for project
let total=3h → 180
1. hw: 45+30=75; +50=125; +25=150
2. left: 180−150=30
chk: 150+30=180 ✓
⇒ 30
```

Every intermediate sum (75, 125, 150) survives; the double- and triple-checks in
the verbose trace collapse to one `chk:` line; all self-talk is gone.

### Exemplar 2 (staged candidate 1) — level 1, geometry

- **_uid:** `gsm8k:87f4cb6f` — **gold:** `24`
- **problem:** Triangle perimeter 50; right side 2 cm longer than left; left =
  12 cm. Find the base.
- **verbose trace:** 7,719 chars (staged file, candidate 1 — mostly spent
  resolving whether "right side" implies a right triangle)

**Compact-register rewrite:**

```
goal: base b?
let L=12, R=L+2=14, L+R+b=50
1. "right side" = name of third side, or right angle? test both
2. case plain triangle: b=50−12−14=24
3. case right triangle, legs 12,14: hyp=√(144+196)=√340≈18.4; per≈44.4≠50 ✗
4. case hyp=R=14: b²=14²−12²=52; b=√52≈7.2; per≈33.2≠50 ✗
⇒ b=24
chk: 12+14+24=50 ✓; R−L=2 ✓
```

Note what was **kept**: the interpretation ambiguity and both rejected branches
(that *is* the reasoning in this trace — eliding them would be a leap). What was
dropped: five rounds of re-litigating the same three cases in prose. 7,719 chars
→ ~340 chars with zero steps fused.

### Exemplar 3 (staged candidate 3) — level 3, algebra

- **_uid:** `deepmath:5ddfa38c` — **gold:** `3`
- **problem:** How many integers satisfy $(x+3)^2 \leq 1$?
- **verbose trace:** 8,675 chars — contains a genuine wrong turn (a botched
  interval) caught by a test value; the register **keeps** that correction.

**Compact-register rewrite:**

```
goal: # integers with (x+3)²≤1
1. (x+3)²−1≤0 → (x+2)(x+4)≤0
2. roots −4, −2; opens up ⇒ −4≤x≤−2
3. |x+3|≤1 route: −1≤x+3≤1 → −4≤x≤2? chk x=0: (0+3)²=9>1 ✗; fix: subtract 3 both ends → −4≤x≤−2 ✓ agrees with 2
4. integers in [−4,−2]: −4,−3,−2
chk: x=−4→1≤1 ✓; x=−3→0 ✓; x=−2→1 ✓; x=−1→4>1 outside ✓
⇒ 3
```

### Exemplar 4 (staged candidate 6) — level 5, coordinate geometry / number theory

- **_uid:** `deepmath:1cd7da92` — **gold:** `67`
- **problem:** R=(8,6) is the midpoint of PQ with P on 8y=15x, Q on 10y=3x;
  |PQ| = m/n in lowest terms; find m+n.
- **verbose trace:** 5,790 chars

**Compact-register rewrite:**

```
goal: |PQ|=m/n, find m+n
let R=(8,6) midpoint; P=(x1, 15x1/8) on 8y=15x; Q=(x2, 3x2/10) on 10y=3x
1. x1+x2=16
2. 15x1/8 + 3x2/10 = 12 → ×40: 75x1+12x2=480
3. sub x2=16−x1: 75x1+192−12x1=480 → 63x1=288 → x1=288/63=32/7
4. x2=16−32/7=80/7
5. y1=(15/8)(32/7)=60/7; y2=(3/10)(80/7)=24/7
6. Δx=80/7−32/7=48/7; Δy=24/7−60/7=−36/7
7. |PQ|=√(48²+36²)/7; 2304+1296=3600; √3600=60 → 60/7
8. gcd(60,7)=1 ⇒ m=60, n=7
chk: 48²=2304, 36²=1296, sum 3600 ✓
⇒ 67
```

### Exemplar 5 (staged candidate 7) — level 6, integral (MCQ)

- **_uid:** `deepmath:697274a5` — **gold:** `B`
- **problem:** Evaluate $\int_3^4 \sqrt{(x-3)(4-x)}\,dx$ using the x→7−x
  symmetry; options (a) π/4, (b) π/8, (c) π/2, (d) none.
- **verbose trace:** 11,724 chars — shows the register handling a dead-end
  hint exploration, a mini-correction inside a route, two independent
  cross-checks, and a **letter** answer. (Staged topic label said
  "combinatorics" — the keyword heuristic misfired; it is an integral.)

**Compact-register rewrite:**

```
goal: I=∫₃⁴ √((x−3)(4−x)) dx; options: a π/4, b π/8, c π/2, d none
1. hint y=7−x: limits 3↔4, dx=−dy → I=∫₃⁴ √((4−y)(y−3)) dy = I ⇒ symmetric about x=3.5; I+I=2I gives nothing direct
2. sub t=x−3: I=∫₀¹ √(t(1−t)) dt
3. Beta route: not B(2,2) (that is ∫t(1−t)); exponents 1/2 → B(3/2,3/2)=Γ(3/2)²/Γ(3)=(√π/2)²/2=π/8
4. chk trig route: t=sin²θ → I=2∫₀^{π/2} sin²θcos²θ dθ=(1/4)∫₀^{π/2}(1−cos4θ)dθ=(1/4)(π/2)=π/8 ✓
5. chk circle route: u=x−3.5 → I=∫ √(1/4−u²) du over [−1/2,1/2] = half-disc r=1/2 = π(1/4)/2=π/8 ✓
6. π/8 = option b
⇒ B
```

---

## 4. Self-check (completed at FILLED, 2026-08-02)

- [x] User reviewed and ratified §1 (symbols, markers, shorthand, elision rules).
- [x] 5 exemplars spanning levels 1–6: arithmetic (1), geometry (2), algebra
      (3), coordinate geometry (4), integral/MCQ (5). **Combinatorics: none —
      known gap**, no verifier-correct combinatorics trace under the staging
      cap; the seed corpus will cover the topic at scale.
- [x] Each compact rewrite re-read for leaps — every step's value present,
      self-corrections preserved (ex. 3 step 3, ex. 5 step 3), dead ends kept
      where they are reasoning (ex. 2 cases 3–4, ex. 5 step 1).
- [x] §1.1 symbol table matches symbols used in exemplars; letter-tag naming
      absent (banned, activity 004).
- [x] §2 whitelist matches §1.1.
- [x] Tokenizer audit §1.6: zero boundary-token injections, lossless round-trip.
- [x] Card ≈ 1 page spec + 5 exemplars (rides in the teacher's Stage-A context).
