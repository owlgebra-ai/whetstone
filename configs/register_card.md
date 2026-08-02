<!--
  WHETSTONE v2 — REGISTER CARD  ***TEMPLATE — NOT YET FILLED IN***

  Staged by packet P2 Part 4. THIS FILE IS THE ONE HUMAN DESIGN INPUT IN THE
  WHOLE PIPELINE (design §1, precondition 2): the register is *specified, not
  discovered*. No downstream component is asked to invent it.

  P3 (seed register corpus) IS BLOCKED UNTIL THE TODOs BELOW ARE FILLED IN.

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

  HOW TO FILL IT IN: replace every ⟨TODO …⟩ marker. Delete the FORMAT DEMO
  block once your own exemplars are in — it exists to show the shape of an
  entry, not to propose a notation. Real pool problems with real verbose
  traces are staged for you in configs/register_card_exemplars_staged.md.
-->

# Register card — WHETSTONE v2 compact reasoning register

**Status:** ⟨TODO: change to FILLED when done — P3 checks this line⟩
**Author:** ⟨TODO⟩
**Date:** ⟨TODO⟩
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

Concretely, when writing the exemplars in §3:

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

⟨TODO: ~1 page. This is the notation the model is being taught. Be prescriptive
and exhaustive — ambiguity here shows up as register-internal variance later.⟩

### 1.1 Symbol vocabulary

Fill in the table. Candidate symbols (design §12.3's structural whitelist starts
from these — add or remove freely, but record the final set here because it
becomes the token set R):

| Symbol | Means | Example use |
|---|---|---|
| `⇒` | ⟨TODO: e.g. "therefore / implies"⟩ | ⟨TODO⟩ |
| `→` | ⟨TODO: e.g. "rewrite as / substitute"⟩ | ⟨TODO⟩ |
| `;` | ⟨TODO: e.g. "step separator within a line"⟩ | ⟨TODO⟩ |
| `✓` | ⟨TODO: e.g. "checked / verified"⟩ | ⟨TODO⟩ |
| `⚠` | ⟨TODO: e.g. "case needs care / constraint"⟩ | ⟨TODO⟩ |
| `?` | ⟨TODO: e.g. "unknown / to determine"⟩ | ⟨TODO⟩ |
| ⟨TODO: add rows⟩ | | |

### 1.2 Step-marker convention

⟨TODO: how is a step opened? `1.` / `S1:` / bare newline? Do steps renumber per
case? Note: `whetstone/reward/extract.py` already treats a leading `^\d+\.` as a
chunk restart, and the v1 reward code counts numbered steps — a numeric marker
is the cheapest to instrument.⟩

### 1.3 Equation-manipulation shorthand

⟨TODO: how is "multiply both sides by 3" written? How is substitution written?
How is a case split opened and closed? How is a sub-result named and referred
back to?⟩

### 1.4 What may be elided vs never elided

| May be elided | Never elided |
|---|---|
| ⟨TODO⟩ | Final numeric/symbolic result of each step |
| ⟨TODO⟩ | ⟨TODO⟩ |

### 1.5 Answer segment

The register governs the **think segment only**. After `</think>` the model must
still produce a normal, human-readable final answer ending in `\boxed{…}` —
Stage C holds the answer segment to the original checkpoint with a forward-KL
term, and the deterministic verifier reads post-`</think>` only.

⟨TODO: confirm / add any answer-segment convention you want.⟩

---

## 2. Structural whitelist for R (auto-derived, confirm here)

The Round-0 register-token set R = {types with mean surprisal > 75th pct AND
across-occurrence std < median} ∪ **structural whitelist**. The whitelist is the
symbols from §1.1 plus step markers.

⟨TODO: after §1.1 is final, list the exact literal strings here. P4 tokenizes
this list against Qwen3's vocab and dumps the ids.⟩

---

## 3. Exemplars

5–10 pairs, one per difficulty band, **at least one algebra, one combinatorics,
one geometry**. Real problems from the pool.

Real pool problems with the model's own real verbose think traces are staged in
[`register_card_exemplars_staged.md`](register_card_exemplars_staged.md) — they
are the raw material for these slots. Use them (or pick your own from
`/data/whetstone/data/pool/train_30k.jsonl`), and write the COMPACT side.

<!-- ============ FORMAT DEMO — DELETE ONCE YOUR EXEMPLARS ARE IN ============
     This shows the SHAPE of an entry. The notation used is a placeholder,
     NOT a design proposal.

### Exemplar 0 — FORMAT DEMO (delete me)

- **_uid:** `gsm8k:00000000`
- **level:** 1
- **topic:** arithmetic
- **problem:** A shop sells pens at $3 each. Ann buys 7 and pays with a $50 note.
  How much change does she get?

**Verbose think trace (as emitted by the model):**

    Okay, so Ann is buying pens. Each pen costs $3, and she buys 7 of them.
    So first I need to find the total cost. That would be 3 times 7. Let me
    compute that: 3 * 7 = 21. So the pens cost $21 in total. Now, she pays
    with a $50 note, so the change is 50 minus 21. Let me compute: 50 - 21 = 29.
    So she gets $29 in change. Let me double check: 7 pens at $3 is 21, and
    50 - 21 is 29. Yes, that's right.

**Compact-register rewrite:**

    1. cost = 3·7 = 21
    2. change = 50 − 21 = 29 ✓

Note what happened: every *value* survived (21, 29); only the prose around them
was dropped. No step was fused.
============================ END FORMAT DEMO ============================= -->

### Exemplar 1

- **_uid:** ⟨TODO⟩
- **level:** ⟨TODO⟩
- **topic:** ⟨TODO — algebra / combinatorics / geometry / number theory / …⟩
- **problem:** ⟨TODO⟩

**Verbose think trace:**

    ⟨TODO⟩

**Compact-register rewrite:**

    ⟨TODO⟩

### Exemplar 2

- **_uid:** ⟨TODO⟩
- **level:** ⟨TODO⟩
- **topic:** ⟨TODO⟩
- **problem:** ⟨TODO⟩

**Verbose think trace:**

    ⟨TODO⟩

**Compact-register rewrite:**

    ⟨TODO⟩

### Exemplar 3

⟨TODO — repeat the block. Aim for 5–10 total, spanning the difficulty bands
present in the pool (levels 1, 4–5, 6–7, 8–9; note levels 2–3 and 10 are nearly
empty in DeepMath — see activity 002 note 1).⟩

---

## 4. Self-check before handing this back

- [ ] Every ⟨TODO⟩ replaced; FORMAT DEMO block deleted.
- [ ] 5–10 exemplars, spanning difficulty bands.
- [ ] At least one algebra, one combinatorics, one geometry.
- [ ] Re-read each compact rewrite asking "could a 1.7B model produce the next
      line from the previous one alone?" — no leaps.
- [ ] No step's final value was dropped.
- [ ] Symbol table in §1.1 matches the symbols actually used in the exemplars
      (a symbol that appears only in the exemplars will not be in R).
- [ ] Card is ~1 page of spec + exemplars — it rides in the teacher's context
      on every Stage-A rollout.
