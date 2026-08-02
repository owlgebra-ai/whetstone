<!--
  WHETSTONE v2 — REGISTER CARD, VARIANT B ("caveman" / telegraphic English)
  ***BAKE-OFF CANDIDATE — see activity/packets/P3a-register-bakeoff.md***

  Same elision rules and structural skeleton as the symbolic card
  (configs/register_card.md, variant A). The ONLY difference: connectives are
  common English words instead of symbols, and derivations may be written as
  terse sentence fragments instead of one-op-per-arrow chains. The bake-off
  measures which variant the model handles more organically (style-tax size and
  concentration, entropy profile, Δlogp faithfulness); the winner (or a hybrid)
  becomes the ratified card. Do not edit one variant without mirroring the
  shared parts in the other — the comparison is only valid if elision rules
  and skeleton stay identical.
-->

# Register card (variant B) — telegraphic-English compact register

**Status:** BAKE-OFF CANDIDATE — not ratified.
**Target model:** Qwen3-1.7B (feasibility tier)

---

## 0. The one constraint that decides whether this works

**Shorter lines, not bigger jumps.** The register is compressed *notation*, not
compressed *reasoning*. Every step in the verbose trace still exists in the
compact one. Never fuse derivation steps into an asserted result. Never drop an
intermediate value. If a 1.7B model could not produce the next line from the
previous one alone, it is a leap — rewrite it.

Telegraphic style compresses the **words**, never the **steps**. `"compute.
add. done."` is the canonical failure (v1 audit), because the values vanished.
`"total 180. hw 150. left 30."` is correct caveman: terse, every value present.

---

## 1. Notation spec

### 1.1 Style

- Sentence fragments. Drop articles, auxiliaries, pronouns, pleasantries:
  "Now I need to find the total" → `find total.`
- Keep all mathematical content verbatim: numbers, variables, equations, LaTeX.
- Connectives are plain words, each a single token:

| Word | Means | Example |
|---|---|---|
| `so` | therefore (closes a derivation) | `12+14+b=50 so b=24.` |
| `check:` | verification line (exactly one per check) | `check: 12+14+24=50 ok.` |
| `ok` | check passed / branch confirmed | `per = 50 ok.` |
| `no` | branch ruled out / check failed | `per ≈ 44.4, not 50. no.` |
| `note` | caution / constraint later steps depend on | `note x≠0.` |
| `find` | states the unknown / subgoal | `find base b.` |
| `case <cond>:` | opens a case branch | `case right triangle:` |
| `goal:` / `let` / `sub` | as variant A | `let L=12.` |

### 1.2 Step-marker convention

Identical to variant A: one step per line; top-level steps numbered `1.` `2.`
at line start; `goal:`/`let`/`check:`/`case` lines unnumbered; `;` separates
micro-steps within a line.

### 1.3 Derivation style

Short declarative fragments, one operation per fragment:
`2x+5=19. 2x=14. x=7.` — the equation states themselves are the steps; no
symbol chain required. Substitution: `sub x=7: y=3·7−2=19.` Every case ends in
a result (`so …`) or a rejection (`… no.`).

### 1.4 What may be elided vs never elided

**Identical to variant A §1.4.** Elide: restatement, hedging, self-talk,
repeated re-derivations (one `check:` line survives), transitional prose,
markdown scaffolding. Never elide: any step's final value, intermediate values
in a chain, case branches and verdicts, `let` definitions, the final check,
constraint notes.

### 1.5 Answer segment

Identical to variant A: register governs the think segment only; normal prose +
`\boxed{…}` after `</think>`.

---

## 2. Structural whitelist for R

```
so  ok  no  note  find  case  check  goal  let  sub
1.  2.  3.  4.  5.  6.  7.  8.  9.
```

(All common English tokens — the bake-off's M3 measures whether a usable R-set
exists for this variant at all; that is one of the things being decided.)

---

## 3. Exemplars (same two problems as variant A, for direct comparison)

### Exemplar 1 — `gsm8k:97f4db57` (gold 30)

    goal: minutes left for project.
    let total = 3h = 180 min.
    1. hw: 45+30=75; +50=125; +25=150. (A)
    2. left: 180−A=30.
    check: 150+30=180 ok.
    so 30.

### Exemplar 2 — `gsm8k:87f4cb6f` (gold 24)

    goal: find base b.
    let L=12, R=L+2=14, L+R+b=50.
    1. "right side" = third side name, or right angle? test both.
    2. case plain triangle: b=50−12−14=24.
    3. case right triangle, legs 12,14: hyp=√(144+196)=√340≈18.4. per≈44.4, not 50. no.
    4. case hyp=R=14: b²=196−144=52. b≈7.2. per≈33.2, not 50. no.
    so b=24.
    check: 12+14+24=50 ok. R−L=2 ok.
