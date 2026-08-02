# 003 — P2: preconditions (segment parser, entropy audit, calibration probe, register card)

- **Packet:** [packets/P2-preconditions-audit.md](packets/P2-preconditions-audit.md)
- **Status:** done
- **Machine(s):** mac (code), turing (all GPU work)
- **Code commit(s):** `896db9e` → `<this commit>`
- **Started / finished:** 2026-08-02 → 2026-08-02

## Goal

The four design-§1 preconditions that must hold before any corpus is built:
a segment parser proven against Qwen3-1.7B's *actual* chat template, the entropy
audit that picks SED's preservation-vs-restoration mode, the v1 Step-2.1
calibration probe, and the register card staged for the user (the one human
design input, and the thing P3 is blocked on).

---

## Part 1 — Segment parser

### What the real tokenizer says

Everything below was dumped from the shipped tokenizer, not taken from docs
(`/tmp/inspect_template.py` on turing; output reproduced in the test docstring).

    model      Qwen/Qwen3-1.7B
    revision   70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    class      Qwen2Tokenizer     (transformers 5.14.1)
    vocab      151643  (len(tok) = 151669)

| token | id | single token inline? |
|---|---|---|
| `<think>` | **151667** | yes |
| `</think>` | **151668** | yes |
| `<|im_start|>` | 151644 | yes |
| `<|im_end|>` | 151645 | yes (also `eos_token`) |
| `<|endoftext|>` | 151643 | yes (also `pad_token`) |
| `assistant` | 77091 | yes |

All three template behaviours the packet asked me to verify **hold**:

1. `enable_thinking=True` →
   `'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n'`.
   It does **not** pre-fill `<think>`; the model emits it. Confirmed on real
   rollouts — all three sampled fixtures start with token 151667.
2. `enable_thinking=False` → `'…<|im_start|>assistant\n<think>\n\n</think>\n\n'`.
   Parsed as an *empty think segment*, not malformed, as required.
3. Multi-turn strips previous-turn think blocks
   (`<think>reasoning one</think>\n\nA1` renders as just `A1`).

Two extra facts worth recording:

- **The default (kwarg omitted) equals `enable_thinking=True`** on this
  revision. That is *why* the ROADMAP rule to always pass it explicitly matters:
  the eval was accidentally correct, not deliberately correct, and a template
  bump would have flipped it silently.
- `<think>` / `</think>` are added tokens with **`special=False`**, so
  `decode(..., skip_special_tokens=True)` does **not** strip them.

### Design decisions in `whetstone/segments.py`

- **Masks exclude the boundary tokens and the terminating EOS.** `think_len` is
  the body only, so `G_budget`'s annealed budget B means what the schedule says
  rather than being 1–2 tokens off.
- **`prompt_len` parameter** so the same call serves both the rollout-only case
  and the §12.2 one-pass `(q, τ)` scorer prefill; prompt positions are 0 in both
  masks.
- **`think_opened_by_prompt`** for v1-style seeded prompts (see the harvest bug
  in Part 3). Off by default.
- **`blank_token_ids`** (optional): without it the empty-answer rule is
  token-count based, and a rollout that emitted `</think>\n\n` then hit the cap
  passes the gate on one whitespace token. Both behaviours are asserted in the
  tests so the difference is visible to whoever wires up training.
- **The g=0 list was not widened.** The packet names four rules (missing
  `</think>`, duplicated boundary, empty answer, close-before-open); I kept
  exactly those. Two anomalies that are *not* in the list surface as
  **warnings** instead: `preamble_before_think` (tokens before `<think>` belong
  to neither mask, so they receive no loss routing in Stage C — a silent hole
  worth seeing) and `trailing_tokens_after_eos`. **Decision for the next agent:**
  if Stage C wants those gated out, tighten there, not here.
- No third-party imports, so the module and its tests run on the Mac.

### Tests

`tests/test_segments.py`, **24/24 green on both machines**. Runs under pytest or
as a plain script (`python tests/test_segments.py`) — no test framework needed.

Covered: well-formed short/long/no-EOS, cap-hit, duplicated open, duplicated
close, close-before-open, missing open, empty answer (zero-token and
whitespace-only), empty think (both the `enable_thinking=False` pre-fill shape
and zero tokens), `<think>` mid-text, prompt-prefixed sequences, out-of-range
`prompt_len`, `think_opened_by_prompt`, multi-turn rejection, batch counting,
plus the real rollouts.

`test_recorded_ids_match_live_tokenizer` re-derives every recorded id and both
template shapes from the live tokenizer — it skips on the Mac (no transformers)
and **executes on turing**, where it passes. That is the guard against a
tokenizer revision bump silently invalidating the constants.

**Real-rollout fixtures** (`tests/fixtures/real_rollouts_qwen3_1p7b.json`) are
vLLM's own `output.token_ids`, never a re-tokenization of the decoded text:

| fixture | tokens | finish | shape |
|---|---|---|---|
| `well_formed` | 1956 | stop | g=1, think 1526 / answer 427 |
| `cap_hit` | 48 | length | g=0 `missing_think_close` |
| `pool_problem` | 4096 | length | g=0 `missing_think_close` |

`pool_problem` was *meant* to be a second well-formed case; at a 4096-token
budget a real val_2k problem ran out of room mid-think. Kept as-is — a real
cap-hit is better test material than a synthetic one, and the fixture test is
data-driven off each record's own boundary count so new fixtures drop in without
editing assertions. It is also the first datapoint that **4k is far too small a
think budget for this pool**, which Part 3 confirms.

---

## Part 3 (code) — template fixes
*(the probe run itself is in "Part 3 (run)" below)*

### `run_eval.py` — the fix P1 handed forward (activity 002 note 4)

- `_build_prompt` now passes **`enable_thinking=True`** (flag-controlled).
- Defaults moved to the design-§12.7 SCA-matched protocol: **K=8, T=0.7,
  top_p=0.95, max_tokens=32768**, `max_model_len` 36864 (was 32768, which could
  not hold prompt + 32k completion). v1's cheap settings stay reachable by flag —
  the probe and the continuity dashboard both want them.
- `--no_system_prompt` added; `sys_prompt=""` now sends no system message at all
  rather than an empty one.
- The run summary records `top_p`, `max_tokens`, `enable_thinking` and the
  system prompt, so a run's eval protocol is recoverable from its own output.

### `harvest.py` — a live bug, not just a missing flag

`--prefill_think` **defaulted to `True`** (a Gemma-era default). On Qwen3 with
`enable_thinking=True` the rendered prompt ends at the assistant header, so the
old code appended `<think>\n` to the prompt — which moves the opener *into the
prompt*. Every completion would then have no `<think>` token and parse as
`missing_think_open` (g=0), i.e. the P3 seed harvest would have gated out
**100% of its own rollouts**.

Fixed: `prefill_think` now defaults to **False**, `enable_thinking` is passed
and defaults True, and `--no_system_prompt` was added.

**Gemma scrub (P3 Part 1 asks P2 to verify):** `harvest.py` and `run_eval.py`
import **no** `whetstone.patches.*` and build prompts purely through
`apply_chat_template`. The only remaining Gemma references are docstring
examples. Scrub confirmed.

---

## Part 2 — Entropy audit

### Runs

Generation and scoring are **separate process invocations** on purpose — see
gotcha 1 below.

```bash
# on turing, cd ~/workspace/whetstone && source .venv/bin/activate
python -u scripts/entropy_audit.py \
  --pool /data/whetstone/data/pool/val_2k.jsonl \
  --out_dir /data/whetstone/runs/entropy_audit \
  --n 200 --generate_only                      # 200 rollouts, T=0.9, top_p=0.95,
                                               # max_tokens=16384, enable_thinking=True

python -u scripts/entropy_audit.py \
  --pool /data/whetstone/data/pool/val_2k.jsonl \
  --out_dir /data/whetstone/runs/entropy_audit \
  --n 200 --compare_model Qwen/Qwen3-1.7B-Base   # resumes from rollouts.jsonl
```

Outputs in `/data/whetstone/runs/entropy_audit/`: `probe.jsonl` (200 problems),
`rollouts.jsonl`, `per_token_entropy.npz` (11.6 MB — raw per-token arrays, not
just histograms, per gotcha 4), `audit.json`, 4 PNGs (copied to
[assets/003/](assets/003/)).

### Rollout health (before any entropy number is trusted)

| | |
|---|---|
| traces | 200 |
| segment-parser gate pass | **91.0%** (182/200) |
| gate failures | 18, **all** `missing_think_close` |
| cap-hit rate at 16k | **10.0%** |
| median think length | **6,099 tokens** |
| median answer length | **679 tokens** |

Every gate failure is a 16k cap-hit — there is not a single duplicated boundary,
empty answer or inverted boundary in 200 real rollouts. The parser and the model
agree on structure.

**Median think = 6,099 tokens is the headline number for the whole project.**
That is the verbosity WHETSTONE exists to compress, measured on the actual pool
with the actual checkpoint. `G_budget`'s B_target of 600 (design §12.6) is a
~10× reduction from here.

### Entropy (top-512, nats)

| segment | n tokens | p10 | p25 | **p50** | p75 | **p80** | p95 | mean |
|---|---|---|---|---|---|---|---|---|
| think | 1,406,480 | 0.0000 | 0.0000 | **0.0278** | 0.5819 | **0.6923** | 1.3114 | 0.3176 |
| answer | 126,013 | 0.0000 | 0.0000 | **0.0012** | 0.3092 | 0.5119 | 1.3325 | 0.2522 |
| think — **Qwen3-1.7B-Base** | 1,406,480 | 0.0003 | 0.0032 | **0.1163** | 0.7298 | 0.8824 | 1.6417 | 0.4306 |
| answer — **Base** | 126,013 | 0.0001 | 0.0008 | 0.0184 | 0.4260 | 0.6153 | 1.4810 | 0.3094 |

| segment | collapse mass (<0.1) | fork mass (>1.5) | bimodality coeff. |
|---|---|---|---|
| think | 56.8% | **2.8%** | 0.705 |
| answer | 67.8% | 3.6% | 0.739 |
| think — Base | 48.7% | 6.8% | 0.621 |
| answer — Base | 62.2% | 4.8% | 0.612 |

The reference column is the *same 200 traces* teacher-forced through
`Qwen3-1.7B-Base`, which is what makes "median comparable to a base model"
(design §1) an actual measurement rather than a guess.

**Position-alignment sanity check passed decisively:** median entropy of the
`</think>` token itself = **6.6e-05 nats**. The packet predicted `</think>`
after a terse line should be low-entropy; near-zero is as low as it gets. An
off-by-one would have smeared this with neighbouring think tokens and produced
something O(0.1). Alignment is right.

### Verdict — RESTORATION (Stage-B `Δ_max = 0.7`)

Three of four collapse checks fire:

| check | fired | evidence |
|---|---|---|
| think median < 0.15 nats | **yes** | 0.0278 |
| collapse mass > 0.60 | no | 0.568 (just under) |
| fork mass < 0.10 | **yes** | 0.028 |
| think median < ½ of reference | **yes** | 0.0278 vs 0.1163 → **4.2× lower** |

The argument (the packet asks for one, since there is no published threshold):
the checkpoint's think-segment median entropy is **4.2× below the same-family
base model on identical text**, and its high-entropy mass is **2.4× thinner**
(2.8% vs 6.8%). More than half of all think tokens are effectively deterministic
(<0.1 nats), and over a quarter are numerically indistinguishable from zero
(p25 = 0.0000). This is a post-RL checkpoint that arrived entropy-collapsed —
exactly the case design §4.2 introduces restoration mode for. **Δ_max = 0.7.**

Caveat, stated plainly: the sub-checks are a heuristic I wrote, not a published
rule. The base-model comparison is the load-bearing evidence; the other two
thresholds mainly corroborate it. Δ_max is revisited at Stage B anyway.

### Two findings the next agent must not miss

1. **The design's expected "80/20 fork structure" is not what this checkpoint
   has.** The histogram *is* formally bimodal (BC 0.705 > Sarle's 0.555, and
   *more* bimodal than the base model's 0.621), but the second mode sits at
   **≈0.65–0.70 nats**, not above 1.5. Fork mass above 1.5 nats is 2.8%, nowhere
   near 20%. Read the think histogram PNG: a huge spike at 0, then a small but
   distinct secondary bump right around 0.7, then a thin tail.

   Consequences to carry forward:
   - **The 1.5-nat "fork" threshold is the wrong knife for this model.** Any
     later component that hard-codes it (dashboards, the S3 entropy floor)
     should use ~0.7 instead, or measure rather than assume.
   - **TEA's `τ_c = 1.0`** (design §12.6) sits *above* this checkpoint's second
     mode. At face value that would treat almost every genuine decision token as
     low-entropy. Flagged for Stage C (P7) — `τ_c` is not on the run-1 sweep
     list (β, H_pivot, λ_TEA), and on this evidence it probably should be.
   - Encouragingly, **the H_pivot recipe lands on the real structure**: p80 of
     the think histogram = 0.6923, essentially exactly the secondary mode. The
     design's "80th percentile" rule is picking out something real. (This p80 is
     from *native* traces and is **not** H_pivot — see below.)

2. **Per-level think-entropy is U-shaped, not monotonic.**

   | level | 1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
   |---|---|---|---|---|---|---|---|---|
   | median | 0.0266 | 0.0392 | 0.0173 | 0.0099 | 0.0152 | 0.0287 | 0.0763 | 0.0979 |

   The most collapsed band is the **middle** (level 5, 0.0099) and the hardest
   bands retain 4–10× more entropy (level 9, 0.0979). Mid-difficulty is where
   the model is most rigidly confident, which is precisely where over-confident
   compression can go unnoticed.

### H_pivot is NOT pinned by this run

Per gotcha 3, H_pivot is the 80th percentile of the **compact-register**
histogram, which needs the P3 seed register corpus. The script supports this
directly:

```bash
python scripts/entropy_audit.py --traces <seed_register.jsonl> \
  --completion_field completion --out_dir /data/whetstone/runs/entropy_audit_compact
```

For reference only, p80 of the *native* think histogram is 0.6923.

### Gotchas found while running this

1. **vLLM's `EngineCore` is a separate process and outlives its parent.** When
   the first fixture run was killed, its `EngineCore` was reparented to init and
   sat on **28.6 GB of the 32 GB card indefinitely**; the next vLLM start died
   with `RuntimeError: Engine core initialization failed` whose real cause (an
   OOM) was ~200 lines up the log — the same shape of buried error as activity
   001 gotcha 1. `entropy_audit.py` now tears the engine down explicitly, and
   `--generate_only` splits generation from scoring into separate processes so
   the HF pass gets the card to itself. **If a vLLM start fails, check
   `nvidia-smi --query-compute-apps` for an orphaned `VLLM::EngineCore` before
   debugging anything else.**
2. **`pkill -f "VLLM::EngineCore"` matches its own command line** and kills the
   shell running it (ssh returns 255, nothing else happens). Kill by PID from
   `nvidia-smi --query-compute-apps=pid`.
3. **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in
   transformers 5.x**, not a list of ids — `list(enc["input_ids"])`. Any v1-era
   snippet that indexes the return value directly will raise a confusing
   `TypeError` inside `decode`.
4. **matplotlib was not installed** on turing; added to `pyproject.toml` core
   deps (dashboards are first-class stage deliverables, design §7).

---

## Part 3 (run) — calibration probe

`scripts/calibration_probe.py` (new) runs v1 Step 2.1 unchanged, with the exact
sampling config the P3 seed harvest will use (**K=2, T=0.9, top_p=0.95,
max_tokens=32768, max_model_len=34816, enable_thinking=True**) on a 50-problem
proportional level-stratified slice of `train_30k.jsonl`.

Two things it does differently from a literal v1 re-run, both inside the probe's
stated purpose:

- Format compliance is measured with the **token-level parser**, not a string
  search, so the probe validates the same masks Stages A–C will route on.
- It compares **two prompt variants in one run** — `sys` (v1's system prompt)
  vs `nosys` (no system message). Prompts are built by importing
  `harvest._build_prompt`, so the probe exercises the real harvest path rather
  than a copy of it.

```bash
python -u scripts/calibration_probe.py \
  --pool /data/whetstone/data/pool/train_30k.jsonl \
  --out_dir /data/whetstone/runs/calibration_probe --n 50 --K 2
```

### Results — 100 rollouts per variant

| metric | threshold | `sys` (v1 prompt) | `nosys` (no system msg) |
|---|---|---|---|
| M1 format compliance | ≥ 80% | 94.0% PASS | **100.0% PASS** |
| M3 pass rate overall | U-shaped | 65.0% | **73.0%** |
| M4 median think grows with level | yes | PASS | PASS |
| M5 cap-hit rate | < 10% | **0.0%** PASS | **0.0%** PASS |
| gate failures | — | **6 × `duplicated_think_close`** | **none** |
| median think / answer tokens | — | 4,698 / 666 | 6,531 / 751 |

**All five metrics pass on both variants**, so by v1's decision rule the probe
clears P3 to proceed. But the comparison found a real fault:

### The template fix the probe forced: drop the v1 system prompt

v1's system prompt is *"Place all your step-by-step reasoning between `<think>`
and `</think>` tags. After `</think>`, give the final answer."* On Qwen3 that
instruction is not just redundant — it is **harmful**:

- **6% of rollouts emit a duplicated `</think>`** with it, and **0%** without.
  Naming the tags in the prompt is what makes the model re-emit one. Every one
  of those is `g = 0` and would be dropped from all structural rewards.
- **Accuracy is 8 points lower** with it (65% vs 73%).

Applied: `run_eval.py --system_prompt` now defaults to `""`, and
`harvest._load_system_prompt(None)` now returns `""`. The v1 text is retained in
both files as `SYS_PROMPT_V1` with the measurement in the comment, and a system
prompt can still be supplied deliberately. **`sft_train.py` still carries its own
v1 `SYS_PROMPT`** — deliberately untouched, since Stage B replaces that script
outright (CLAUDE.md code map).

### 32k is the right harvest budget; 16k is not

Cap-hit rate is **0.0% at 32,768 tokens** across both variants, versus **10.0%
at 16,384** in the entropy audit on the same model. P3's `max_tokens 32768` is
correct and must not be lowered. Per-level median think length grows from ~600
(level 1) to ~13,600 (level 9) in the `sys` variant — metric 4 is healthy, and
the growth is steep enough that a smaller budget would truncate the top bands
first, exactly where yield is already lowest.

### Per-level pass rate (metric 3)

| level | 1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| n | 22 | 2 | 4 | 16 | 26 | 12 | 14 | 4 |
| `sys` | 72.7% | 0.0% | 100% | 50.0% | 69.2% | 58.3% | 78.6% | 25.0% |
| `nosys` | 86.4% | 0.0% | 100% | 56.2% | 80.8% | 66.7% | 71.4% | 50.0% |

Broadly U-shaped and not uniformly low, which is what metric 3 asks for. Levels
3, 4 and 9 have n ≤ 4 — activity 002 note 1 again (there is no level-2/3/10 mass
in this pool), so read those cells as noise. `nosys` is at or above `sys` in
every band except level 8.

### Metric 2 — verifier acceptance shape (10 hand-checked decisions)

The spot-check found the failure mode v1 warns about explicitly:

| extracted | gold | verdict | assessment |
|---|---|---|---|
| `290 tomatoes` | `290` | **False** | **verifier miss — unit suffix**, answer is right |
| `$$` | `e^{\frac{\pi}{2}}` | False | **extraction failure** on a `$$…$$` display block |
| `\frac{191}{7}` | `\dfrac{285}{7}` | False | genuinely wrong |
| `1` | `\dfrac{e}{5}` | False | genuinely wrong |
| `** Yes` | `No` | False | correct verdict; note markdown leaking into extraction |
| `Yes` / `False` / `\infty` / `3x^2 y - y^3` / `66` | match | True | correct, incl. LaTeX and boolean golds |

Quantified over all 200 rollouts (both variants) by checking cases where the
gold is a substring of the extraction or vice versa: **6% (`sys`) / 4% (`nosys`)
near-misses**, of which the clearly-recoverable classes are the unit-suffix case
(`290 tomatoes`) and the `$$` display-block extraction failure (1 in `sys`, 3 in
`nosys`). So on the order of **2–4% of yield is lost to extraction/verification
shape, not to reasoning.**

**Not fixed here, deliberately.** CLAUDE.md's first invariant is that
`verify.py` stays deterministic and that leniency belongs in `whetstone/reward/`.
The `$$` case is arguably a plain extraction bug rather than leniency, but
changing `verify.py` would shift measured yield across every v1 comparison and
is out of P2's scope. **Handed forward:** P8 (eval hardening) should decide, and
P3 should watch for it — if seed-harvest yield comes in ~3 points under the P2
probe numbers, this is the reason, not a template fault.

---

## Part 4 — Register card (⚠ BLOCKS P3)

Two files staged for the user:

1. **[`configs/register_card.md`](../configs/register_card.md)** — the card
   template. Every section is a marked `⟨TODO⟩`: notation spec (symbol table,
   step markers, equation shorthand, elide-vs-never-elide), the structural
   whitelist that seeds the Round-0 token set R, and 5–10 exemplar slots. It
   opens with the one constraint that decides whether the whole approach works —
   *shorter lines, not bigger jumps* — with the "could a 1.7B model reproduce
   the next line from the previous one alone?" test, plus a self-check list. A
   clearly-labelled FORMAT DEMO block shows the shape of an entry and is marked
   for deletion so it cannot be mistaken for a notation proposal.

2. **[`configs/register_card_exemplars_staged.md`](../configs/register_card_exemplars_staged.md)**
   — generated by `scripts/stage_register_exemplars.py` (new) from the audit's
   own 200 rollouts. 8 real pool problems with the model's **real** verbose think
   traces and an empty compact slot under each, so the user only writes register
   notation. Verifier-correct traces only — an exemplar built on a wrong trace
   would teach the register on reasoning that never reaches the answer.

   | # | level | topic | trace chars | _uid |
   |---|---|---|---|---|
   | 1 | 1 | geometry | 7,719 | `gsm8k:87f4cb6f` |
   | 2 | 1 | other | 1,600 | `gsm8k:97f4db57` |
   | 3 | 3 | algebra | 8,675 | `deepmath:5ddfa38c` |
   | 4 | 4 | other | 5,113 | `deepmath:4bf08255` |
   | 5 | 5 | algebra | 8,287 | `deepmath:be20883e` |
   | 6 | 5 | number_theory | 5,790 | `deepmath:1cd7da92` |
   | 7 | 6 | combinatorics | 11,724 | `deepmath:697274a5` |
   | 8 | 6 | other | 8,200 | `deepmath:8c2036a6` |

   The packet's three required topics (algebra / combinatorics / geometry) are
   all covered. Topic labels come from a keyword heuristic and are printed so a
   wrong call is easy to spot and swap; candidate 1 was hand-checked and is a
   genuine triangle problem.

   **Only 58 of 200 traces were eligible at all**, and at the initially chosen
   4,000-char cap only **one** candidate survived — the native traces are simply
   too long to hand-rewrite. The cap is now 12,000 chars, and levels 7–9 have no
   candidate under it. If the user wants a hard-band exemplar they will have to
   accept a very long verbose side.

### ⚠ P3 IS BLOCKED UNTIL THE CARD IS FILLED IN

The register is *specified, not discovered* (design §1 precondition 2) — no
downstream component invents it. P3's prompted compression puts the card in the
compressor's context and records its sha as provenance; P4's inoculation loss
masks on the token set R derived from its symbol table; Stage A carries it in
the teacher's context on every rollout. Nothing downstream can start without it.

---

## Conclusion

**P2 is complete; all four Definition-of-Done items pass.** The parser is proven
against the real template rather than the documentation, the entropy audit has a
verdict backed by a same-family base-model comparison, the calibration probe
clears all five v1 metrics *and* caught a live prompt fault, and the register
card is staged with real material.

Established for downstream packets:

1. **`<think>`=151667, `</think>`=151668**, single tokens; `enable_thinking=True`
   does not pre-fill `<think>` — the model emits it. Parser and model agree on
   structure in 100% of 200 real rollouts (all 18 gate failures were 16k
   cap-hits, none structural).
2. **SED runs in RESTORATION mode → Stage-B `Δ_max = 0.7`.** Think-segment
   median entropy is 0.0278 nats, **4.2× below Qwen3-1.7B-Base on identical
   text**, with 2.4× thinner high-entropy mass.
3. **The expected 80/20 fork structure is not present.** The second mode sits at
   **≈0.7 nats, not >1.5**; fork mass above 1.5 is 2.8%. **TEA's `τ_c = 1.0`
   sits above this checkpoint's second mode** and should probably join the run-1
   sweep (design §12.6 currently sweeps only β, H_pivot, λ_TEA). Flagged for P7.
4. **Median think length is 6,099 tokens** — the verbosity the project exists to
   compress, and the number `G_budget`'s B_target of 600 is measured against.
5. **Drop the v1 system prompt.** It causes 6% duplicated-`</think>` gate
   failures and costs 8 points of accuracy on Qwen3. Both scripts now default to
   no system message.
6. **`harvest.py --prefill_think` defaulted to True and would have gated out
   100% of the P3 seed harvest.** Now False.
7. **32,768 is the right harvest budget** — 0% cap-hit vs 10% at 16,384.
8. **H_pivot is NOT pinned.** P3 must re-run
   `entropy_audit.py --traces <seed_register.jsonl>` on the compact corpus to
   set it. Native-trace p80 is 0.6923, for reference only.
9. **~2–4% of yield is lost to extraction/verification shape**, not reasoning
   (unit suffixes, `$$` display blocks). Not fixed — `verify.py` stays
   deterministic. Handed to P8; P3 should expect its yield ~3 points under the
   probe numbers for this reason.

**Next:** P3 is unblocked on everything except the register card, which is the
user's to write. P4 inherits the entropy baseline (`per_token_entropy.npz`) for
its S3 entropy-floor stop.
