# 004 — P3a: register bake-off, symbolic (A) vs telegraphic/caveman (B)

> **CORRECTION (activity 005, 2026-08-02).** The M5 section below reports the
> `**Final Answer** … \boxed{…}`-inside-`<think>` violation as an **arm-B**
> failure and does not report it for arm A. **Arm A has it at 20/50 = 40%**
> (B: 41/50 = 82%), so `final_A.jsonl` is contaminated and the "§1.5 violation"
> row is not an A-vs-B discriminator — only a 2× difference on a defect both
> arms share. **The verdict is unaffected** (decided by register adoption, 15×).
> Cause: v1's `clean_oneshot` only stripped a code fence at the very end of the
> text, so it removed no trailer in either arm. Fixed in activity 005 finding 5;
> the same 50 traces recompressed now measure 0/50. Do not reuse
> `final_{A,B}.jsonl` as clean register corpora.

- **Packet:** [packets/P3a-register-bakeoff.md](packets/P3a-register-bakeoff.md)
- **Status:** done — awaiting user ratification of the card
- **Machine(s):** mac (code), turing (compression, entropy), spark (Δlogp, style tax)
- **Code commit(s):** `b5bfa96` → `<this commit>`
- **Started / finished:** 2026-08-02 → 2026-08-02

## Goal

Decide the register form empirically before P3 commits the seed corpus, against
the user's criterion: *whichever is more organic and natural for the model to
handle, for training dynamics and entropy.* Five measurements (M1–M5) over the
same 50 traces compressed under two cards.

---

## VERDICT

> **Arm A (symbolic) wins. Ratify `configs/register_card.md`, with two required
> card edits (§ "Required card edits") before P3 Part 2 builds the seed corpus.**

**Deciding rubric line: 3 — "If B's tax is small *and* usable-R exists for it →
B wins" — fails on both halves, and it fails for a reason that sits underneath
the whole rubric: arm B never installed its register at all.**

Under a notation-neutral prompt, card B produced **0.24 register markers per 100
think tokens** against card A's **3.68** — a 15× gap that held at every
temperature tested and under both prompt variants. What B actually produced was
the model's *native* markdown-LaTeX solution write-up: `**bold headers**`,
`$$…$$` display blocks, `---` rules, and — in many traces — a literal
`**Final Answer** … \boxed{…}` **inside the think segment**, which card §1.5
forbids outright. B's plain-English connectives (`so`, `ok`, `no`, `check:`) are
too close to ordinary prose to displace that default style; A's symbols are
distinctive enough to switch the model into a different mode.

A register the model ignores cannot be the ratified register, so B is eliminated
before the style-tax comparison decides anything. Rubric 2's *direction* also
holds (A's excess surprisal is the more concentrated, and it sits on register
markers while B's sits on markdown scaffolding), though neither arm reaches the
literal 70% / 40% thresholds. **Rubric 4 (hybrid) is rejected on the evidence:**
B's word connectives are precisely the component that failed, so grafting them
onto A's skeleton imports the failure.

A's size is **not** extreme in the Risk-1 sense: p95 `d_t` gap is 2.375 nats,
below the τ_leap ≈ 4 scale that would make the register indistinguishable from a
genuine reasoning leap. No escalation needed on that axis.

---

## The finding that outranks the bake-off

**v1's chunkwise prompted compression (`compress_local_versionB.py`, v1 §3.4) is
broken on Qwen3-1.7B under a notation-neutral prompt, and is retired as the
default.** This is a P3 Part 2 blocker that the bake-off happened to surface.

The chunkwise loop shows the model, at depth k, ORIGINAL chunks 1..k plus the
COMPACT chunks 1..k−1 it already emitted. That cumulative context is a
**repetition attractor**: once two consecutive compacts agree, the model copies
that block at every remaining depth. Measured on the first bulk run:

| | arm A | arm B |
|---|---|---|
| stalled-chunk rate (median) | **0.59** | 0.00 |
| traces ≥50% stalled | **54%** | 32% |
| register markers /100 think tokens | 0.39 | 0.12 |

Arm B emitted byte-identical chunks 2, 3 and 4 on the very first alignment
check. Adding an explicit chunk-scoping rule to the scaffold fixed arm B and did
not fix arm A.

**The decisive diagnostic** was a one-shot probe — whole think segment in, whole
compact rewrite out, same scaffold, same cards. It separates "the model cannot
execute this register" from "the loop is the defect", and the answer was
unambiguous:

```
ARM A | gsm8k:97f4db57 | verbose 491 tok -> one-shot 90 tok (ratio 0.183)
goal: minutes left for project
    let total=3h → 180
    1. hw: 45+30=75; +50=125; +25=150 (A)
    2. left: 180−A=30
    chk: 150+30=180 ✓
    ⇒ 30
```

That is card A's exemplar style reproduced essentially exactly, from a prompt
that never names a single symbol. Register-marker density rose **10×** when the
chunkwise loop was removed (0.39 → 3.68 per 100 tokens).

`--mode oneshot` is now the default; `--mode chunkwise` remains reachable by
flag. **P3 Part 2 must build the seed register corpus one-shot.**

---

## Runs

### Run 1 — subset selection (turing, CPU)

```bash
python -u scripts/select_bakeoff_subset.py \
  --out_dir /data/whetstone/runs/register_bakeoff --n 50
```

200 audit rollouts → **139 eligible** (18 segment-parser gate failures, all 16k
cap-hits; 43 verifier-wrong; 0 missing gold) → proportional level-stratified 50,
seed 0. Levels 3 and 9 have 2 eligible traces each and round to 0 selected —
activity 002 note 1 again. Median verbose think **5,404 tokens**.

Outputs: `/data/whetstone/runs/register_bakeoff/{subset.jsonl,subset_uids.json}`.

The audit rollouts predate `entropy_audit.py` carrying `ground_truth`, so golds
are joined back from `probe.jsonl` by `_uid`. The think/answer split comes from
the **token-level parser** on vLLM's own `completion_token_ids`.

### Run 2 — chunkwise compression (turing) — superseded, kept as evidence

`bakeoff_{A,B}.jsonl`, T=0.4, `--max-chunk-tokens 800 --max-chunks 24`. Results
in the table above. `--max-chunks 24` (not v1's 16) because 7 of 50 traces
exceed 16 chunks and v1's `merge_and_cap` is **round-robin**, which scrambles
chunk order — a silent corruption of long traces. Max observed: 20 chunks.

Chunk-size and temperature variants (5 traces/arm) confirmed the loop, not the
settings, was at fault: at 250-token chunks the stalled-chunk rate rose to
0.94/0.69 and compression collapsed to ratio 0.555; T=0.7 and T=1.0 did not
rescue it.

### Run 3 — one-shot compression, temperature sweep (turing)

```bash
python -u scripts/compress_local_versionB.py --mode oneshot \
  --input  /data/whetstone/runs/register_bakeoff/subset.jsonl \
  --output /data/whetstone/runs/register_bakeoff/final_A.jsonl \
  --model Qwen/Qwen3-1.7B --card configs/register_card.md --arm A \
  --tp 1 --gpu-mem 0.85 --max-tokens-oneshot 2048 \
  --temperature 0.4 --top-p 0.95 --seed 0
```

**Temperature was swept at the user's request (T = 0.4 / 0.7 / 1.0, both arms,
all 50 traces).** It does not change the verdict:

| T | A markers/100 | A ratio | A cap-hit | B markers/100 | B ratio | B cap-hit |
|---|---|---|---|---|---|---|
| 0.4 | 4.74 | 0.040 | 18% | 0.31 | 0.077 | 0% |
| 0.7 | 4.38 | 0.041 | 12% | 0.26 | 0.073 | 0% |
| 1.0 | 4.35 | 0.037 | 8% | 0.27 | 0.076 | 4% |

Register adoption, compression and the A-vs-B ordering are flat in T. The one
real temperature effect is that **arm A's runaway rate falls as T rises**
(18% → 12% → 8%), which is consistent with those runaways being a
low-temperature repetition lock rather than a content problem. **T=0.4 stays
pinned** for P3 (design wants mild register-internal variance); raising T is the
recorded mitigation if runaways persist after the card fix below.

The primary corpora `final_{A,B}.jsonl` were regenerated after making the
scaffold's chunk-scoping rule chunkwise-only — it is meaningless in one-shot
mode. That single change cut arm A's runaway rate 18% → 10%. Both prompt
variants are byte-identical across arms, so the A-vs-B comparison is valid in
either; `final_*` is primary because its prompt is the correct one.

---

## M1–M5 — the side-by-side table

Primary corpora: `final_{A,B}.jsonl`, one-shot, T=0.4, n=50, seed 0.

| | verbose orig. | **arm A** (symbolic) | **arm B** (caveman) |
|---|---|---|---|
| **M1** think tokens, median | 5,404 | **176** | 376 |
| M1 IQR | 2,771–7,846 | 122–477 | 143–594 |
| M1 compression ratio (median) | 1.000 | **0.043** | 0.077 |
| M1 % under B_target = 600 | 2% | **80%** | 76% |
| register markers /100 think tok | — | **3.68** | 0.24 |
| runaway (cap-hit at 2,048) | — | 10% | **0%** |
| **M2** Δlogp pass rate | — | **66%** | 56% |
| **M3** mean surprisal (nats) | 0.240 | 0.561 | **0.481** |
| M3 p95 surprisal | 1.388 | 2.862 | **2.489** |
| M3 mean `d_t` gap | 0.093 | 0.467 | **0.369** |
| M3 p95 `d_t` gap (τ_leap ≈ 4) | 0.750 | 2.375 | **2.000** |
| M3 excess nats / think token | 0.165 | 0.515 | **0.423** |
| M3 **top-20 excess share** | 33.2% | **52.6%** | 46.7% |
| M3 types ≥ 10 occurrences | 1,224 | 182 | 226 |
| M3 proto-R size | 4 | 1 | 1 |
| **M4** think median entropy | 0.0278† | 0.0002 | **0.0013** |
| M4 think p80 (H_pivot preview) | 0.6923† | 0.2276 | **0.4288** |
| M4 collapse mass (<0.1) | 56.8%† | 76.1% | **69.6%** |
| M4 fork mass (>1.5) | 2.8%† | 4.1% | 3.8% |
| `verify_response` | 50/50 | **50/50** | **50/50** |

† native-trace baseline from activity 003 (16k rollouts), not this subset.
Bold marks the better value on that row; note that A does *not* win every row.

**M3's top excess-surprisal types are the whole story of this bake-off:**

| rank | arm A | share | arm B | share |
|---|---|---|---|---|
| 1 | `goal` | 13.2% | `1` | 14.7% |
| 2 | `'   '` (indent) | 8.2% | `.\n` | 4.1% |
| 3 | ` ⇒` | 4.4% | `**` | 2.9% |
| 4 | ` chk` | 3.5% | ` $` | 2.8% |
| 5 | `\n` | 3.3% | ` ok` | 2.0% |
| 6 | ` let` | 1.9% | `'  '` | 1.8% |
| 7 | ` ✓` | 1.7% | `---\n` | 1.8% |

A pays its style tax on **register markers** — exactly the token set Round 0's
inoculation mask is built from (design §12.3). B pays its tax on **step digits,
markdown bold, LaTeX `$`, and horizontal rules** — generic formatting the mask
has no principled claim on, with its own connective `ok` a distant 5th at 2.0%.
That is the concrete meaning of "Round 0 has a clean handle" vs "there is
nothing to mask".

**Caveat, stated plainly:** the top-20 concentration numbers are *not* stable
across the two prompt variants (A 65.4% / B 45.8% under the earlier prompt;
52.6% / 46.7% under the final one), and A's *size* is the larger of the two in
the primary set. The verdict therefore does not rest on M3's size axis. It rests
on register adoption (15×, stable everywhere) and on *what* the excess sits on.

### M5 — faithfulness eyeball

10 traces per corpus reviewed. Two failure classes, one per arm.

**Arm A — the 10% runaway class is a card defect, not a register defect.** The
longest arm-A traces degenerate into a labelling loop:

```
let EEE = π ln(b)
let FFF = π ln(b)
let GGG = π ln(b)      ← deepmath:8dca3f9b, 4,077 → 2,049 tok (cap-hit)
```

The cause is card §1.3's **sub-result naming** convention ("tag a line-final
value with `(A)`, `(B)`, …"): the model exhausts single letters and rolls over
to `AAA`/`BBB`/`CCC`. A second instance (`deepmath:33d0f47d`) enumerates
`arctan(2/(n²+n+4))` to n = 47. Outside this class, arm A's rewrites are clean
and faithful — every intermediate value present, no fused steps:

```
goal: integral value
    let f(z) = z e^{3/z}
    1. residue of f(z) at z=0: coefficient of 1/z in Laurent series
    2. Laurent series of e^{3/z} = sum_{n=0}^\infty 3^n / n! z^{-n}
    3. multiply by z: sum_{n=0}^\infty 3^n / n! z^{1 - n}
    4. coefficient of 1/z is when 1 - n = -1 ⇒ n = 2
    5. residue = 3^2 / 2! = 9 / 2
    6. integral = 2πi * 9 / 2 = 9πi
    chk: 9πi ✓
    ⇒ 9πi
```
(15,451 verbose tokens → 189, with the verbose side's sign flip-flopping dropped
and the actual derivation intact.)

**Arm B — no loops, but no register either, and a §1.5 violation.** B's traces
terminate cleanly (0% runaway — genuinely B's best result) but they are
markdown solution write-ups, and many end with `**Final Answer** … \boxed{…}`
*inside* `<think>`. The register governs the think segment only; a boxed answer
there is exactly the contamination Stage C's answer-segment KL exists to
prevent. **B's predicted failure mode (value-dropping caveman) did not appear** —
because disciplined caveman never appeared either.

Neither arm was disqualified under rubric 1: M2 is 66% / 56% (v1 expects ~70%;
the script's failure line is 50%), and neither M5 class is a
faithfulness-of-content failure.

---

## Required card edits before P3 Part 2

1. **Retire or bound §1.3's `(A)`, `(B)`, … sub-result naming.** It is the
   direct, reproducible cause of the 10–18% runaway class. Replace with a
   back-reference by step number (`step 3`), or cap the scheme at two labels.
2. **Un-indent the exemplars.** They are 4-space markdown code blocks; the model
   copies the indentation verbatim, and pure indentation whitespace is **8.2% of
   arm A's total excess surprisal** — the second-largest single contributor,
   ahead of `⇒`. This is paid on every line of every trace, forever, for
   nothing.

Neither is a change to the notation, so neither invalidates this bake-off.

## Flag for the user — H_pivot is going to land low

A's compact-register think p80 is **0.2276 nats**, against the native-trace
0.6923 — a 3× drop; think median entropy is 0.0002 vs 0.0278, and collapse mass
rises 56.8% → 76.1%. Compact register is far more deterministic text than native
chain-of-thought, which is not surprising, but it means **H_pivot pinned off the
seed corpus (design §12.6) will be small**, and it interacts with two things
activity 003 already flagged: the restoration-mode Δ_max = 0.7, and TEA's
τ_c = 1.0 sitting above this checkpoint's real second mode. Arm B's p80 (0.4288)
is closer to native, which is B's one genuine advantage and is recorded as such.

This is M4 acting as a *flag*, not a winner-adjustment: the rubric's
adjustment clause requires "similar compression", and A compresses 2.1× harder
than B (0.043 vs 0.077).

---

## Deviations from the packet (all deliberate, all logged)

1. **`--mode oneshot` replaces the chunkwise loop as the default.** Justified
   above; the packet assumed v1's machinery worked on this model.
2. **The pinned scaffold gained one rule** — "rewrite ONLY the last original
   chunk" — after the alignment check found the repetition attractor. It is
   notation-neutral and byte-identical across arms, and it is now
   **chunkwise-only** (meaningless in one-shot mode).
3. **Card rendering strips non-notation sections** before pasting: the HTML
   provenance header, "Structural whitelist for R" (a P4 token-set artifact
   carrying raw Qwen3 token ids), "Self-check before flipping" (an A-only human
   checklist) and A's `⟨PENDING⟩` exemplar stub. Without this the two prompts
   would differ by ~40 lines of asymmetric non-register text. The rendered
   prompt's sha1 is recorded per arm in each output's `.meta.json`.
4. **Temperature swept beyond the pinned 0.4** at the user's request. T=0.4
   remains the pinned value; the sweep is recorded above.
5. **`--max-chunks 24`** rather than v1's 16, to avoid `merge_and_cap`'s
   round-robin order-scrambling on the 7 traces exceeding 16 chunks.
6. `perplexity_score.py` now reads `compact_think` (v2 field) and defaults to
   `sdpa` — `flash_attention_2` is installed on neither box.

Prompt provenance (recorded in every `.meta.json`):

| arm | card blob | rendered prompt sha1 | chars |
|---|---|---|---|
| A | `7e0923b25d04` | `f698e4afdc28` (chunkwise) / one-shot variant per sidecar | 8,221 |
| B | `53ae708feff7` | `fbf093d20189` (chunkwise) / one-shot variant per sidecar | 4,177 |

Note the cards are **not** the same length — A's prompt is 2.1× B's. This is an
acknowledged asymmetry inherent to the cards as staged (A carries a symbol table
and exemplar commentary), and it is a real Stage-A cost since the card lives in
the teacher's context (design §3.1).

## Gotchas

1. **The orphaned `VLLM::EngineCore` bit again**, exactly as activity 003 gotcha
   1 describes — this time caused by my own `| head -75` SIGPIPE-ing a probe.
   28.6 GB held, next vLLM start died with "Engine core initialization failed".
   `nvidia-smi --query-compute-apps=pid,used_memory` → `kill -9 <pid>`. Do not
   pipe a vLLM script into `head`.
2. **`style_tax.py` exits 134** ("terminate called without an active exception")
   during vLLM teardown on spark *after* writing its JSON. Results are complete;
   check for the output file before treating the exit code as failure.
3. The Mac cannot reach github (sandboxed network). Sync to turing/spark was
   done with `git bundle` over scp + `git fetch <bundle>` + `git merge --ff-only`
   — commits and shas are preserved, so the journal's shas are real.

---

## Conclusion

**Arm A (symbolic) is the recommended register**, decided by arm B's failure to
install its register at all (0.24 vs 3.68 markers per 100 think tokens) rather
than by the style-tax margins, which are mixed. A also compresses 2.1× harder
(ratio 0.043 vs 0.077), passes Δlogp more often (66% vs 56%), and pays its
excess surprisal on the register markers that Round 0's mask is built from,
whereas B pays it on markdown and digits.

Established for downstream packets:

1. **v1 chunkwise compression is retired for Qwen3-1.7B** — repetition
   attractor, 10× lower register adoption. `--mode oneshot` is the default and
   **P3 Part 2 must use it**.
2. **The register is reachable from a notation-neutral prompt.** The model
   reproduces card A's exemplar style one-shot without the prompt naming a
   single symbol — v1's notation-prescribing SYSTEM_PROMPT is not needed and
   stays retired.
3. **Compression ratio ~0.043 (176 think tokens median)** for arm A, against a
   5,404-token verbose median — well inside `B_target = 600` for 80% of traces,
   so G_budget's B₀ will start far lower than v1's numbers suggested.
4. **H_pivot preview = 0.2276 nats** (arm A p80). P3 pins the real value; expect
   it low.
5. **p95 `d_t` gap = 2.375 nats** for a clean symbolic register, below
   τ_leap ≈ 4 — the Round-0 band-existence check (Risk 1) is not pre-empted by
   the register's own accent.
6. Two card edits are required before the seed corpus is built (above).

**Next:** user ratifies (or edits) `configs/register_card.md`, flips its Status
to FILLED; P3 Part 2 then builds the seed register corpus one-shot with the
ratified card. P3 Part 1 (seed harvest) was never blocked by this packet.
