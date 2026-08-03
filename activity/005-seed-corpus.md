# 005 — P3: seed harvest + seed register corpus

- **Packet:** [packets/P3-seed-corpus.md](packets/P3-seed-corpus.md)
- **Status:** done
- **Machine(s):** mac (code), turing (harvest + compression, 4 model servers), spark (scoring, entropy, judge)
- **Code commit(s):** `ed5cd8e` → `<this commit>`
- **Started / finished:** 2026-08-02 → 2026-08-03

## Goal

Build the only two corpora that exist before the teacher is trained (design §1,
preconditions 3–4): a blind verifier-filtered seed harvest of native verbose
traces, and the seed register corpus produced by one prompted-compression pass
under the ratified card. Then pin **H_pivot** from the compact-register entropy
histogram and write the Round-0 measurement sets P4 consumes.

---

## Infrastructure changes made for this packet

Three changes to how generation runs, all driven by the scale jump from the
bake-off's 50 traces to this packet's 9,000 rollouts.

### 1. vLLM server/client instead of in-process batches

`harvest.py` and `compress_local_versionB.py` both gained a `--server` mode that
issues **one `/v1/completions` request per unit of work** against a resident
`vllm serve`, with a bounded in-flight window.

The offline `llm.generate(batch)` API is a **barrier**: the call does not return
until the batch's slowest member finishes, so one 32k-token rollout idles every
other slot in its batch. Independent requests let continuous batching refill a
slot the moment it frees. The server also outlives the client, so a resume costs
no model load — which matters when the run is expected to be interrupted.

`/v1/completions` is used, never `/v1/chat/completions`: prompts are rendered by
`apply_chat_template` in the calling script, which is what keeps the blindness
contract (v1 §2) and `enable_thinking` (ROADMAP rule 4) auditable and identical
to the offline path. vLLM 0.26.0 supports `return_token_ids: true`, so
`completion_token_ids` survive the HTTP hop — necessary because
`whetstone.segments` is token-level by design and re-tokenizing decoded text
does not round-trip at the `<think>` boundary (design §12.1).

Server command (turing):

```bash
source .venv/bin/activate && vllm serve Qwen/Qwen3-1.7B --port 8000 \
  --max-model-len 34816 --max-num-seqs 64 --gpu-memory-utilization 0.90
```

Reported **GPU KV cache size: 226,800 tokens**, max concurrency 6.51× at full
34,816-token requests.

### 2. Crash-safe resume (`whetstone/runio.py`)

Factored out of `harvest.py` and shared with the compressor:

- **`repair_tail()`** truncates a torn trailing line *before* the file is
  reopened for append. Skipping an unparseable line on read is **not**
  sufficient: the next append lands on that same line, fusing garbage with a
  good record and silently losing the good one on every future pass. This was a
  latent bug in v1's resume machinery.
- **`checkpoint()`** flushes + `fsync`s the corpus, then atomically replaces a
  `<output>.progress.json` sidecar. Data is synced before the sidecar so the
  sidecar can never advertise records that are not on disk. Runs on every exit
  path, including `KeyboardInterrupt`.
- Failed requests go to `<output>.failed.jsonl` as an audit trail and are
  deliberately kept **out** of the corpus, so resume retries them.

### 3. Per-rollout seeds

A single `SamplingParams.seed` shared across a K-sample group makes every
candidate for a problem byte-identical — the group collapses to K copies of one
trace. Seeds are now derived as `sha1(uid:k:seed)[:8]` in both execution paths:
candidates stay independent, and a resumed run regenerates exactly what the
interrupted one would have.

---

## Runs

### Run 1 — seed subset selection (turing, CPU) — 2026-08-02

```bash
.venv/bin/python scripts/select_seed_subset.py \
  --pool /data/whetstone/data/pool/train_30k.jsonl \
  --out_dir /data/whetstone/corpora/seed --frac 0.15 --seed 0
```

4,500 of 29,998 problems, proportional level strata preserved end to end
(including level 10's single row):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| pool | 6002 | 38 | 767 | 1510 | 4833 | 7488 | 3788 | 4303 | 1256 | 13 |
| selected | 901 | 5 | 115 | 226 | 725 | 1124 | 569 | 646 | 188 | 1 |

Sources: deepmath 3,600 / gsm8k 900. Outputs
`/data/whetstone/corpora/seed/{subset_uids.json,subset.jsonl}`. The uid list is
written first and is the resume contract (v1 §2.5).

### Run 2 — blind seed harvest (turing) — 2026-08-02

```bash
source .venv/bin/activate && python -u scripts/harvest.py \
  --input  /data/whetstone/corpora/seed/subset.jsonl \
  --output /data/whetstone/corpora/seed/seed_harvest.jsonl \
  --model Qwen/Qwen3-1.7B --server http://127.0.0.1:8000/v1 \
  --K 2 --temperature 0.9 --top_p 0.95 --max_tokens 32768 \
  --concurrency 96 --flush_every 20 --no_system_prompt \
  --seed 0 --shuffle --worker_id 0 --n_workers 1
```

No system prompt and `--prefill_think` False, both now defaults (activity 003).
Gemma scrub confirmed: `harvest.py` imports no `whetstone.patches.*` and builds
prompts via `apply_chat_template`; the only Gemma references left are docstrings
and the unused `--assistant_model` flag.

**Resume test (packet-mandated), performed twice:**

1. `kill -9` of the client at 124 records. The kill landed between writes, so the
   file was already newline-terminated — the common case, and not a test of
   anything. So:
2. A partial record was appended by hand to simulate a kill *mid-`write()`*, and
   the client restarted. It reported
   `[resume] repaired torn tail: dropped 67 B of a partial record from the
   previous run`, then `[resume] 124 (uid, k) pairs already done` and
   `[load] 8876 rollouts to generate` — 9000 − 124, no duplicated and no skipped
   work. The server drained to `Running: 0, Waiting: 0` when the client died and
   was healthy for the restart.

**Deviation — restarted once more at 2,059 records to add `--shuffle`.** See
finding 2 below.

Note: SIGINT does not stop this client promptly (asyncio blocked in
`as_completed` with ~7k pending tasks); it needed SIGKILL after 40 s. Not a
correctness problem — the file is line-buffered and `repair_tail` covers the
torn-write case — but use `kill -9` and expect the in-flight window to be lost.

---

## Findings

### 1. `max_tokens 32768` nearly eliminates cap-hits

Activity 003 measured **9%** of rollouts hitting the 16,384-token cap. At 32,768
the cap-hit rate is **0.4%**, and the segment-parser gate passes **99.6%** of
rollouts (the only failures are `missing_think_close`, i.e. the cap-hits). The
generous budget converts what was ~9% pure waste — traces with no answer segment,
useless to both the verifier and Part 2 — into usable traces.

### 2. Pool file order is level-clustered; the harvest must shuffle

Pool JSONLs are sorted by `_uid`, so every `deepmath:*` problem is submitted
before any `gsm8k:*` one — and gsm8k **is** the entire level-1 stratum. The
partial yield report at 2,043 rollouts covered levels 2–10 and contained
**zero** level-1 rollouts.

Nothing is lost in a run that completes, but a level-stratified subset harvested
in level-clustered order has an unrepresentative prefix, so an interrupted run
is far less usable than its record count suggests. `harvest.py --shuffle`
(seeded, applied to the remaining work — resume is a set difference on
`(uid, k)`, not positional) fixes it.

It also *helped throughput*: 8.1 → 12.9 rollouts/min, because short level-1
traces now interleave with long deepmath ones and free KV slots faster.

### 3. Card rendering silently re-admitted two non-notation sections

`render_card()` dropped sections by **title substring**. Ratification renamed
the headings — `Self-check before flipping` → `4. Self-check (completed at
FILLED…)`, and the `Exemplars 3–8` ⟨PENDING⟩ stub became real exemplars — so the
bake-off's drop list stopped matching, and **§1.6 (tokenizer audit) and §4
(self-check checklist) leaked into the compressor's system prompt**.

Both are meta about the card, not register spec; §4 even states "Combinatorics:
none — known gap", which is a note to the card's authors and nonsense as an
instruction to a compressor. Drop patterns are now anchored on section
**numbers**, which are stable, and `render_card()` returns what it dropped so
the sidecar records it.

This is the class of failure that produces no error and degrades the corpus
quietly. Worth a standing rule: **config selected by prose title is config that
will drift.**

### 4. Compression prompt, pinned for this packet

`scripts/compress_local_versionB.py --dump-prompt` (CPU-only pre-flight, run on
spark):

| field | value |
|---|---|
| card blob sha | `de176e8044ab398465e7fd330f98b7c70bd399b0` |
| card raw sha1 | `fc143757ef103f21874821a67e5c11b9b164b04b` |
| card rendered sha1 | `e20ce28e111c646358fb745d2699e6a8e2f3804e` |
| **rendered prompt sha1** | `c6656806ba8de84da2b1eb5e543bd006db7b286c` |
| rendered prompt | 10,867 chars / **3,839 tokens** |
| **boundary-token hits** | **0** |
| dropped | §1.6 tokenizer audit, §2 structural whitelist, §4 self-check |
| kept | §0, §1.1–1.5, §3 + exemplars 1–5 |

**P3 gotcha 1 is enforced in code, not by inspection:**
`assert_no_boundary_tokens()` refuses to run the compressor if the rendered
scaffold+card encodes `<think>` / `</think>` / `<|im_start|>` / `<|im_end|>` /
`<|endoftext|>`. It is checked on the *system prompt* rather than the templated
prompt, because the chat template legitimately emits `im_start`/`im_end` at its
own structural positions — from text we wrote the budget is exactly zero.

### 5. The ratified card made the model put `\boxed{}` **inside** the think segment

Found by a 20-trace dry run of Part 2 against the live server, before the real
compression pass. **7 of 16 (44%) compact rewrites carried a
`**Final Answer** \boxed{...}` trailer inside `compact_think`** — a card §1.5
violation, the contamination Stage C's answer-segment KL exists to prevent, and
precisely the failure that disqualified **arm B** in the bake-off, now appearing
in arm A.

**The trailer is the model's native habit and it predates ratification.**
Measured across all three corpora:

| corpus | card | `\boxed{}` in compact think |
|---|---|---|
| bake-off `final_A.jsonl` (**accepted by activity 004**) | old, indented exemplars | **20/50 = 40%** |
| bake-off `final_B.jsonl` | arm B | 41/50 = 82% |
| same 50 traces, ratified card + fixed cleaner | new | **0/50 = 0%** |

**This is a correction to activity 004.** Its M5 review reported the
`**Final Answer** … \boxed{…}`-inside-think violation as an **arm-B** failure
("B's traces … many end with `**Final Answer** … \boxed{…}` *inside* `<think>`,
which card §1.5 forbids outright"), and did not report it for arm A. Arm A had
it at 40%. B is genuinely twice as bad, and the bake-off **verdict is
unaffected** — that was decided by register adoption (15×), not by this axis —
but the specific claim that A was clean here was wrong, and **the arm-A corpus
activity 004 shipped is contaminated at 40%**.

What ratification changed is the trailer's *shape*, not its existence. With
indented exemplars the model appended the trailer as bare text; with fenced
exemplars it closes the fence first and then appends. v1's `clean_oneshot`
removed neither — it stripped a fence only at the very end of the text, which
matches neither case:

```
⇒ Yes
```                 ← model closes the fence …
**Final Answer**    ← … and keeps going
\boxed{Yes}
```

The closing fence is now treated as a hard end-of-register marker wherever it
appears, with a fallback cut at an un-fenced final-answer flourish — which is
the form the bake-off corpus carries (1 of the 7 P3 cases had no fence either,
just `### **Final Answer**` + `$$\boxed{a}$$`).

Re-run on the same traces with the fix: **0/10 `think_has_boxed`**, 5 trailers
cut, and register marker density **4.23 per 100 compact think tokens** against
the bake-off's 4.74 — the cleaning removes the trailer without damaging the
register. `_finalize` now hard-fails on any surviving `\boxed{}` in think, the
same standing as the pre-existing answer-segment leak check, and the per-record
`clean_flags` make the trailer rate visible as card feedback instead of
disappearing into the cleaner.

Same function also had a latent dedent bug: `text.strip()` removed the *first*
line's indentation, so min-indent computed as 0 and uniformly-indented blocks
were never dedented — silently negating half of activity 004's un-indent fix,
whose whole point was that indentation whitespace was 8.2% of arm A's excess
surprisal.

**Two findings in this packet (3 and 5) are both "the card was edited, and a
downstream text-processing rule that keyed on its old shape stopped working
silently."** Neither produced an error. Worth treating as a standing hazard for
every future card edit.

### 6. The letter-tag ban worked — runaways are gone

Bake-off arm A had a **10% runaway rate** (cap-hit at `max_tokens_oneshot=2048`),
traced to card §1.3's `(A)`, `(B)`, … sub-result naming: the model exhausted
single letters and rolled into `AAA`/`BBB` loops. Ratification banned the
scheme. On the 20-trace dry run: **cap-hit 0.0%**, stalled-chunk rate 0.00, 0%
of traces ≥50% stalled. Activity 004's required edit 1 is confirmed effective.

### 7. Register adoption vs the bake-off — resolved: input length, not the card

Measured with the bake-off's own metric code (`bakeoff_metrics.py`, so the
numbers are directly comparable to activity 004):

| | bake-off arm A (T=0.4) | P3 dry run (n=20) |
|---|---|---|
| verbose think tokens, median | 5,404 | 8,796 |
| compact think tokens, median | 176 | 209 |
| compression ratio, median | 0.043 | **0.025** |
| % under `B_target = 600` | 80% | **95%** |
| **markers /100 think tok** | **4.74** | **2.16** |
| cap-hit (runaway) | 10% | **0%** |
| `verify_response` | 50/50 | 20/20 |

Per-marker, over 20 traces: `goal:` in 19, `let ` in 16, `⇒` in 18 — the
*structural* register is installed. What is thin is the **verification and
case-split vocabulary**: `chk:` in 2/20, `✓` in 2/20, `case ` in 0/20, `✗` in
0/20. Step bodies also carry more English connective prose ("Since x₁ is real
and between 1 and 2", "Therefore …") than the card's exemplars do.

Two candidate causes, and they are separable:

1. **Input distribution.** These traces are 8,796 verbose tokens at the median
   against the bake-off's 5,404 (a 32k cap vs 16k), and they compress 1.7×
   harder (0.025 vs 0.043). Harder summarization of longer input plausibly
   produces prose-ier step bodies.
2. **The card itself changed.** Ratification added exemplars 3–5, and those
   have measurably lower marker density than the two the bake-off card carried:

   | exemplars | markers / 100 chars |
   |---|---|
   | 1–2 (bake-off card) | **5.67** |
   | 3–5 (added at ratification) | **4.14** |

   Exemplar 5 alone is 2.44 — close to what the model now produces.

**Controlled run — the bake-off's own 50 traces, recompressed with the ratified
card.** Same inputs, different card, isolating the card from the input change:

| on the SAME 50 bake-off traces | bake-off card | ratified card |
|---|---|---|
| markers /100 think tok | 3.68 | **3.49** |
| compact think tokens, median | 176 | **150** |
| compression ratio, median | 0.043 | **0.035** |
| % under `B_target = 600` | 80% | **96%** |
| cap-hit (runaway) | 10% | **6%** |
| `verify_response` | 50/50 | 50/50 |

**Cause 2 is refuted. The card costs ~5% of register density (3.68 → 3.49) and
buys harder compression and half the runaways.** A 27% lower exemplar density
moved the output by 5%, so the exemplar-density hypothesis does not carry.

The drop to 2.16 is therefore **the input distribution**: 3.49 on 5,404-token
verbose traces vs 2.16 on 8,796-token ones. The 32k harvest cap that bought
finding 1's 0.4% cap-hit rate also produces longer source traces, which the
model summarizes into wordier step bodies.

**And "adoption" is the wrong word for what changed.** Structural marker
*presence per trace* is essentially identical across the two input sets:

| marker | bake-off traces (5.4k) | P3 traces (8.8k) |
|---|---|---|
| `goal:` | 50/50 (100%) | 19/20 (95%) |
| `⇒` | 48/50 (96%) | 18/20 (90%) |
| `let ` | 41/50 (82%) | 16/20 (80%) |
| `chk:` | 21/50 (42%) | 2/20 (10%) |
| `✓` | 21/50 (42%) | 2/20 (10%) |
| `case ` | **0/50** | **0/20** |
| `✗` | 1/50 | 0/20 |

The register skeleton is installed just as reliably. What falls with input
length is the **verification vocabulary** (`chk:`, `✓`: 42% → 10%), and the
density figure falls mostly because compact traces get longer (209 vs 150
tokens) with more English prose inside each numbered step.

**`case ` is absent from both corpora**, including the bake-off corpus activity
004 accepted — so the missing case-split vocabulary is a pre-existing property
of the register as installed, not something ratification or this packet caused.

**Verdict: no card edit proposed, build proceeds.** Reported as register-card
feedback (packet P3 Part 2) with two items for the user:

1. Long source traces yield compact traces that keep the skeleton but drop the
   `chk:`/`✓` verification lines. If Stage A's teacher is meant to imitate
   self-checking, the seed corpus under-supplies that exemplar.
2. `case ` / `✗` are effectively absent corpus-wide, so branch elimination —
   which card §1.3 says must never be silently dropped — has almost no exemplar
   in the conditioning data.

### 8. Activity 004's "H_pivot will land low" flag was mostly corpus contamination

Activity 004 flagged, as a thing for the user to worry about, that arm A's
compact-register think p80 was **0.2276** against the native-trace 0.6923 — a 3×
drop — and warned it would interact badly with restoration-mode `Δ_max = 0.7`
and TEA's `τ_c = 1.0`.

That number was measured on `final_A.jsonl`, which finding 5 has since shown
carries **10% runaway repetition loops, 40% boxed-answer trailers, and copied
4-space indentation**. All three are near-zero-entropy filler, and all three
inflate the token count they are averaged over.

Re-measured on **the same 50 source traces**, recompressed with the ratified
card and the fixed cleaner:

| entropy audit, same 50 traces | `final_A` (bake-off) | ratified + cleaned |
|---|---|---|
| think tokens, total | 21,624 | **11,791** (−45%) |
| think tokens, median/trace | 176 | 150 |
| think entropy median | 0.0002 | **0.0022** |
| **think p80 (H_pivot recipe)** | **0.2276** | **0.5067** |
| cap-hit (runaway) | 10% | 0% |
| `</think>` sanity entropy | — | 0.0198 (low ✓, no off-by-one) |

**H_pivot on a clean corpus is 2.2× the preview**, and 45% of the think tokens
the old measurement averaged over were filler that no longer exists. Against the
native-trace 0.6923, the compact register is *not* the dramatically more
deterministic text activity 004 concluded it was — most of that gap was the
contamination, not the register.

A 12-trace preview on P3's own (longer, 8.8k-verbose) inputs gives 0.667, so the
real value is likely to land in **≈0.5–0.7**, near native. The binding number is
still whatever step 8 measures on the full seed corpus; what is established here
is that **the 0.2276 preview should not be carried forward**, and the design
concerns activity 004 raised about `Δ_max` and `τ_c` are much weaker than it
thought.

### 9. Faithfulness audit — the Δlogp gate cannot see the failure that matters

Every automated check in the compression pipeline is orthogonal to
faithfulness. `verify_response` grades the **answer segment**, which is copied
through untouched — it would pass on an empty `compact_think`. Marker density,
compression ratio and entropy p80 measure style, size and predictability. Δlogp
is the only content signal and it is a low bar: `delta > 0` means "better than
no trace at all".

So `scripts/faithfulness_audit.py` was built (external LLM judge, GLM-5.2 via an
Anthropic-compatible endpoint — a deliberate, logged exception to the
central-model principle; **evaluation only, no judge output enters the corpus**).
Pilot: 49 of the 50 bake-off traces recompressed with the ratified card.

**Judge sanity check first.** It independently flagged both exemplar-leaked
traces found by hand (`deepmath:420f2cf4`, `deepmath:5f402961`) *and* the
repetition-degenerate one — with correct explanations ("solves an entirely
different optimization/coordinate geometry problem; verbose trace is about
hexagon area sectors"). It is detecting, not confabulating.

**Headline: 49% faithful, 29% lossy, 22% wrong**, and strongly level-dependent:

| level | n | faithful | lossy | wrong |
|---|---|---|---|---|
| 1 | 15 | **87%** | 7% | 7% |
| 4 | 3 | 67% | 33% | 0% |
| 5 | 7 | 43% | 14% | 43% |
| 6 | 13 | 31% | 31% | 38% |
| 7 | 5 | **0%** | 100% | 0% |
| 8 | 6 | 33% | 33% | 33% |

Compression is reliable on level-1 gsm8k and degrades badly from level 5 up.

**Cross-tabulated against the Δlogp gate on the same 50 traces:**

| Δlogp | n | faithful | lossy | wrong |
|---|---|---|---|---|
| **pass** | 34 | 58% | 23% | **17%** |
| fail | 15 | 26% | 40% | 33% |

Flag rates **among gate-passing traces**: `dropped_branch` 56%,
`dropped_values` 21%, `invented_content` 15%.

**The gate is real but structurally blind to one failure mode.** It catches the
catastrophic class — both off-topic exemplar-leak traces failed it — and
gate-passing traces are 2.2× more likely to be faithful (58% vs 26%). But the
six unfaithful traces that *pass* are all the same shape:

> "asserts step 5 without the generating-function derivation that justifies it"
> "compact starts from x=2,y=4 with no derivation"
> "all modular constraints, rejected values and bounding logic are absent; only
> the verification remains"

Δlogp asks whether the compact trace **helps predict the gold answer**. A trace
that states the *conclusion* and drops the *derivation* helps predict the answer
perfectly well — better, if anything. **The sufficiency gate optimizes for
exactly the property that makes this class of trace unfaithful**, so no
threshold on it can separate them. This is a structural limit, not a tuning
problem.

That matters beyond P3: a corpus that teaches "assert the conclusion, skip the
derivation" is seeding the pipeline with the *unfollowable leap* that Stage-A's
`G_spike` (β high) exists to penalize, and `dropped_branch` at 56% is a direct
card §1.4 violation ("branch elimination is reasoning, and it stays") —
consistent with `case `/`✗` appearing in ~0% of traces (finding 7).

**Caveats, stated plainly.** n=49, one corpus, one judge. The judge's
false-positive rate is unmeasured — the card permits dropping "repeated
re-derivations of the same result", and some `dropped_branch` flags may be
counting those. And these are the *bake-off's* 5.4k-token traces; P3's own
inputs are 8.8k and compress 1.7× harder, so the real corpus is likely worse,
not better.

**Resolved by the user**, in two steps: authorising an external compressor for a
bootstrap corpus (finding 10), and then — once finding 15 showed the underlying
capability is not promptable at this scale — decoupling teacher from student
([006](006-teacher-student-decoupling.md)). Δlogp itself was retired outright
(finding 11).

### 10. GLM-5.2 bootstrap corpus — attested central-model deviation

**User-authorised deviation from v1 §3** (CLAUDE.md: "the compressor is the SAME
base model that produced the harvest; no external teacher"), taken because
finding 9 measured Qwen3-1.7B's own compressions at 49% faithful / 22% wrong and
0% faithful at level 7 — a seed corpus built from those demonstrates the
register badly.

`scripts/glm_compress.py`, 989/1000 traces, ~50/min, **zero GPU** (API-bound, ran
alongside the harvest and the card A/B). 11 rate-limit failures, retryable on
resume. **0 verify failures, 0 boxed-in-think** after cleaning. Diversity
sampling is round-robin over (level, topic-family) cells rather than
proportional, so the hard levels are represented rather than buried: 161 at L7,
166 at L8, 43 at L9.

GLM produces the same answer-trailer artifact as Qwen3 but at **6% vs 54–65%**,
and `clean_oneshot` removed all of it.

**Deterministic comparison — no judge, so no self-evaluation bias:**

| corpus | n | lines (med) | ratio (med) | **branch kept** | **verification kept** | mark/100ch |
|---|---|---|---|---|---|---|
| **GLM-5.2** | 989 | 11 | 0.0298 | **39%** | **95%** | 3.15 |
| Qwen3 (pilot 50) | 50 | 7 | 0.0185 | 2% | 44% | 1.60 |
| Qwen3 (A_control) | 88 | 8 | 0.0149 | **1%** | **18%** | 0.93 |

"branch kept" = contains `case ` or `✗`; "verification kept" = contains `chk:` or
`✓`. These are string counts, not judgements — and they land on exactly the two
failure modes finding 9 identified (`dropped_branch` 56%, `case ` at ~0%).
GLM also compresses **2× less aggressively** (0.0298 vs 0.0149), which is direct
support for over-compression being the mechanism rather than the card's rules.

A level-8 example (`deepmath:60adefee`, 12,755 verbose tokens → 18 lines) keeps
three case splits with `✗` rejections, every intermediate field-degree
computation, and a `chk:` line — the register as specified, on a hard problem.

**Where this corpus may be consumed** — stamped on every record
(`compressor_model`, `central_model_deviation: true`) and in the sidecar:

| consumer | verdict | why |
|---|---|---|
| Stage-A teacher conditioning | **safe, arguably better** | in-context demonstrations; Stage-A RL pulls the teacher toward what earns reward under the frozen student scorer regardless |
| Round-0 scorer inoculation | **unsafe** | Round 0 calibrates the scorer on the *student's own* register statistics; inoculating on an external distribution is the silently-inverted-meter failure CLAUDE.md calls the project's largest risk |
| H_pivot | **unsafe** | p80 would measure "how surprising is GLM text to Qwen3", not "how surprising is Qwen3's register to Qwen3" — mis-sets the SED gate |
| Stage-B assimilation | **wasteful** | the ZPD band-pass masks tokens outside the student's reachable zone, so much of it is gated off rather than learned |

**Recommendation: GLM corpus → teacher conditioning; Qwen3 corpus → Round 0 and
H_pivot.** Both are cheap to keep; the split costs nothing and protects the two
measurements that are specifically about the student's own distribution.

**What this does not yet establish.** Marker presence is a necessary, not
sufficient, condition for faithfulness — 39% branch-retention beats 1% but is
not 100%. Δlogp on the GLM corpus is running. And **reachability is still
open**: a corpus Qwen3-1.7B cannot imitate is a bad teacher-conditioning target
no matter how good it looks, which is what the `F_glm_exemplars` A/B arm exists
to measure.

### 11. Δlogp retired; structural gate replaces it

**The whole Δlogp family is the wrong instrument here, not just its threshold.**
Three independent measurements:

1. 17% of gate-**passing** traces judged unfaithful, all in one shape —
   conclusion asserted, derivation dropped (finding 9);
2. the more faithful GLM corpus passes **less** often than the terser Qwen3 one
   (58.4% vs 70%) — the gate mildly selects *against* faithfulness;
3. **the obvious fix does not work.** `--mask-conclusion` (strip trailing `⇒`
   lines, then score) was implemented and tested: GLM 56% vs Qwen3 63% — same
   ordering. Diagnosis: the gold string survives masking in **54–61%** of
   traces, because a correct derivation *ends by producing the answer*
   (`2. left: 180−150=30` both derives and states it).

So any metric of the form `P(gold | q, compact)` is dominated by whether the
answer is literally in context and cannot separate "derived 30 and wrote 30"
from "just wrote 30". The flag survives in `perplexity_score.py` for the record,
but Δlogp is no longer P3's gate.

**Replacement: `scripts/structural_gate.py`** — card §1.4's "never elided"
column, measured directly against each trace's own verbose source. Deterministic
string ops: no model, no external judge, no distributional bias, free. Each
check fires **only when the source warrants it**, which is what makes it a
faithfulness measure and not a style measure.

Calibration on the two corpora:

| check | GLM (n=989) | Qwen3 (n=200) |
|---|---|---|
| `branch_kept` (`case `/`✗` when source branched) | **39.9%** | 3.0% |
| `verify_kept` (`chk:`/`✓` when source verified) | **95.9%** | 37.5% |
| `value_coverage` median | **0.667** | 0.500 |
| `value_coverage` p10 | **0.40** | 0.111 |
| `invented_frac` p90 | 0.318 | **0.077** |

**A bug in the gate's own first draft, worth recording** because it is the third
time this confound has bitten: counting line-leading step numbers as content
made the *better* corpus look like it invented more (0.25 vs 0.083 median). The
register numbers its steps, so a longer — i.e. more faithful — rewrite
mechanically introduces more integers. After stripping step markers both medians
are 0.0 and the ordering is sane.

**`invented_frac` is a diagnostic, not a gate** (default disabled). An earlier
reading of it — p90 0.318 for GLM vs 0.077 for Qwen3 — was reported as evidence
GLM confabulates numerals, and does not survive audit. A third of it was
comma-fusion of the register's own step back-references (`from 4,5:` parsed as
the numeral **45**), now fixed. The residual is not confabulation: the worst
survivors are an abstract-algebra trace whose only numerals are exponent
notation (`S^{-1}A`) and step cross-references, and a trace that **builds a
concrete counterexample to check itself** (`use (0,0.49),(0.51,1) → Σμ=0.98>0.5 ✓`)
— work we want. The check penalises abstract problems, cross-references, and
self-checking by instance, so the gate's real checks are `branch_kept`,
`verify_kept` and `value_coverage`.

**Four times in this packet a numeral-extraction confound produced a false
finding** (37% "invented" in the ad-hoc probe → step numbers; the gate's first
draft → step numbers again; then comma-fusion; then notation-vs-content). Each
one initially read as a substantive result. Worth treating any numeral-derived
metric here as guilty until audited.

**Thresholds are not pinned yet.** `branch_kept` at 39.9% even on the best
corpus means requiring it would reject 60% of it — so it is reported, not gated.
A workable starting set for the *conditioning* corpus is `verify_kept` required,
`value_coverage ≥ 0.5`, `invented_frac ≤ 0.5`. The binding calibration comes
from the paired run (below), not from these two non-paired corpora.

**Caveat on this table:** the two corpora were compressed from *different* input
sets (GLM from the 1,200-problem file, Qwen3 A_control from a 200-problem file),
so this is suggestive, not paired. The Qwen3 register corpus is queued on **the
same 1,200 inputs as the GLM corpus**, which makes every later comparison
like-for-like.

### 12. Pivot — what the GPU is actually for now

`A_control` (200/200, Qwen3 + ratified card) **is** the start of the Round-0
corpus, which needs ~1,000. So:

* **card A/B killed** at `F_glm_exemplars` 2/200. Its question — can Qwen3
  imitate GLM-style compression — is a Stage-A question that Stage-A RL answers
  directly, and it was consuming GPU that P3 needs. **This is a deferral, not a
  resolution: reachability is now unmeasured.**
* **Qwen3 register corpus queued** to start automatically when the harvest
  client exits, at concurrency 64 on the whole GPU, over the same 1,200 inputs
  as the GLM corpus.

Corpus roles, settled:

| corpus | compressor | consumed by | gate |
|---|---|---|---|
| `seed_register_qwen` | Qwen3-1.7B | Round-0 inoculation, H_pivot, verbose control | structural checks **reported, not filtered** — representativeness is the point |
| `seed_register_glm` | GLM-5.2 | Stage-A teacher conditioning | filter on `structural_pass` |

### 13. Qwen3 corpus faithfulness at n=200 — pilot confirmed, gradient sharpened

GLM judging **Qwen3's** output (not self-evaluation), 200 traces on the real P3
inputs rather than the pilot's 49 bake-off traces.

| | pilot (n=49) | n=200 |
|---|---|---|
| faithful | 49% | **40%** |
| lossy | 29% | 40% |
| **wrong** | 22% | **21%** |

`dropped_branch` 46.3%, `fused_steps` 34.5%, `dropped_values` 18.6%,
`invented_content` 16.4%, `off_topic` 3.4% (the exemplar-leak class persists at
roughly the rate the string-matching audit found).

| level | 1 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| faithful | 76% | 36% | 26% | 37% | 31% | **21%** | 29% |
| wrong | 11% | 9% | 26% | 15% | 23% | **39%** | 29% |

**The pilot's "0% faithful at level 7" was n=5 noise** — the real shape is a
steady decline bottoming at level 8, not a cliff. Worth correcting because that
number was quoted as the headline motivation for the whole card A/B.

The failure mode is worse than omission. In several cases the compact trace
**adopts what the verbose trace rejected**:

> "Verbose reaches [P(8,k)]² and answer 8; compact uses C(8,k)²×k!, a rejected
> formula, yielding 6."
> "Verbose self-corrects and rejects the binary strategy as invalid; compact
> adopts it, inventing the final answer."

That is `dropped_branch` at its most harmful: not losing the rejection, but
keeping the rejected thing and building on it. It also explains why
`invented_content` and `dropped_branch` co-occur.

**What this does and does not change.** It does **not** disqualify the Qwen3
corpus for Round 0 / H_pivot: those consume register-*token* statistics and need
the student's own distribution, and a 21%-unfaithful corpus is an accurate
picture of what this student produces — which is the point. It **does** sharpen
the Stage-A concern: the v2 teacher *is* Qwen3, so Stage A starts from a
compressor that drops a branch on ~46% of traces, and `G_spike` has to do that
work through RL.

---

### 14. Hand inspection — a record whose think and answer disagree

The packet's 5-example review, done on the **paired** corpora (same problem,
both compressors). The first example is the finding:

`gsm8k:95e3c9c6`, gold **6600**. GLM's rewrite reaches 6600 and shows two
rejected readings of "a month" (`chk 28-day route: … ≠ 6600 ✗`,
`chk 30-day fractional weeks: … ✗`). Qwen3's rewrite reaches **6200** — the
wrong answer — by silently choosing one reading and never testing it.

**That record passes every automated check in the pipeline**, including
`verify_response`, because the verifier grades the **answer segment**, which was
copied through from the original correct rollout. The record therefore contains
a flat contradiction: think says 6200, answer says 6600. This is finding 9's
argument made concrete in one trace, and it is the shape of thing a corpus can
carry indefinitely without any gate noticing.

**Why a "does the conclusion match gold" check is not the easy fix.** The
obvious response is to compare the compact trace's final `⇒` against the gold
answer. Attempted, and it does not work off the shelf: the register writes
**Unicode math by design** (card §1.1 chose `²` over `^2` for token cost), while
`whetstone.verify` expects LaTeX. So `4√2` vs `4\sqrt{2}`, `π/4` vs
`\dfrac{\pi}{4}`, `a³+a²` vs `a^3 + a^2` all read as mismatches. Measured
"agreement" came out 67.6% for GLM and 82.8% for Qwen3 — *inverted*, because GLM
uses more Unicode math and is punished more by the normaliser.

Two consequences worth carrying forward:

* **The deterministic verifier cannot grade compact-register conclusions.** It
  is fine where the project actually uses it — Stage A/C grade the *answer*
  segment, which stays LaTeX — but a register-aware normaliser (Unicode math ↔
  LaTeX) is a prerequisite for any think-segment grading.
* **Fifth measurement artifact in this packet.** Step numbers (twice),
  comma-fusion, notation-vs-content, and now Unicode-vs-LaTeX have each produced
  a confident, inverted result on this corpus. Every one looked like a finding
  first. The standing lesson: on this data, a naive string comparison between
  corpora is guilty until audited.

---

### 15. Branch preservation is a scale capability that prompting cannot transfer

Four compressors, four prompting channels, all measured on the same inputs. This
consolidates a sequence of runs whose intermediate readings were misleading —
see the correction note at the end.

**As compressors** — paired, 989 shared problems, identical inputs and ratified
card:

| compressor | branch | verify | val_cov | lines | mark/100ch |
|---|---|---|---|---|---|
| Qwen3-1.7B | 3.1% | 26.2% | 0.500 | 8 | 1.22 |
| Qwen3-14B-FP8 | 5.9% | 60.1% | 0.625 | 10 | 0.82 |
| **Qwen3-32B-NVFP4** | **13.9%** | **70.6%** | 0.600 | 9 | **2.10** |
| GLM-5.2 | 39.9% | 95.9% | 0.667 | 11 | 3.15 |

Branch retention scales **3.1 → 5.9 → 13.9%**, roughly a doubling per ~2.3×
parameters. Extrapolated, matching GLM's 39.9% needs ~2 further doublings
(order 10²B) — consistent with GLM-5.2 being frontier-scale, so the gap is scale,
not a mysterious training difference.

Per level, one qualitative difference persists: **32B is flat at 13–18% across
L3–L9 while GLM rises 39% → 51%** with difficulty. Scale raises the floor; only
GLM adapts to how much case analysis a problem contains. At level 1 all four sit
at 6–9%, which is evidence the metric tracks something real rather than style —
trivial arithmetic has no branches to preserve.

**As demonstrations — nothing works.** Same 200 traces, 1.7B compressor:

| channel | branch | verify | mark/100ch |
|---|---|---|---|
| ratified card (no demos) | 2% | 35% | 1.26 |
| GLM long-trace exemplars in the card (+517 tok/rollout) | 1% | 42% | 1.19 |
| level-matched k=4 demos, GLM pool | 2% | 44% | 1.10 |
| level-matched k=4 demos, 14B pool | 1% | 49% | 0.85 |
| level-matched k=4 demos, 32B pool | 2% | 40% | 1.17 |

**Branch retention is 1–2% under every configuration.** The `verify` spread
(35–49%) is within noise at n=199 (SE ≈ 3.5pp) and shows no ordering by pool
quality or family proximity. Level-matching *did* beat the static card on length
(8 → 9 lines, ratio +15%), so the channel is not inert — it transfers surface
behaviour and not structure. Emitting a `chk:` line is formatting and copies
from a demonstration; noticing that a 6,500-token trace explored and discarded an
approach is comprehension, and no prompt moved it.

**Design consequence.** v2 makes teacher and student the same checkpoint, and
design §3 argues in-context conditioning is what puts the target inside the
sampling support so GRPO can rank it — *"a group-relative reward … cannot jump
to a register it never samples"*. At the 1.7B tier that premise fails for branch
preservation under every channel tried, so Stage-A rollouts would not contain the
behaviour and `G_spike` could not select for it. Interpolating the curve, the
**4B/8B tier gate does not resolve it either** (~4%). The remaining levers are
architectural — recorded and decided in
[006](006-teacher-student-decoupling.md).

**Serving note.** Qwen3-32B-NVFP4 runs on one RTX 5090:
`quantization=modelopt_fp4`, `kv_cache_dtype=fp8_e4m3` from NVIDIA's checkpoint,
64,224-token KV cache, ~22 traces/min at concurrency 8,
`--gpu-memory-utilization 0.93`. 14B-FP8: 76,768-token KV, ~30/min at
concurrency 12.

**Caveats.** `branch_kept` is the `case `/`✗` marker proxy; 14B and 32B write
more prose ("either case", "this fails") and may be undercounted, which would
flatter GLM. 14B's raw output triggered `boxed_in_think` on **65.6%** of traces
(GLM 3%, 32B 7.8%, 1.7B 2.4%) — all cleaned, none surviving, but its unprompted
§1.5 compliance is markedly worse. FP8/NVFP4 are weight quantizations on
compressor-only paths and never touch the student or scorer.

**Correction note — what the intermediate runs got wrong.** Reading 14B alone,
this was written up as "scale does not buy branch preservation" and "demo
quality and reachability are anti-correlated"; the 32B arm falsified both — 14B
was simply a weak point on two axes at once (branch barely moved, marker density
dipped to 0.82). Two further single-point claims were retracted the same way: a
same-family transfer advantage (14B demos 49% vs GLM 44% — 32B gives 40%, within
noise) and student marker density tracking its pool (holds for 14B at 0.85 vs
0.92, fails for 32B at 1.17 vs 2.19). The superseded text is not preserved here;
the pattern is recorded in the Method note below because it recurred.

---

## Part 1 result — full seed harvest, 9,000 / 9,000 rollouts, 0 failures

Completed 2026-08-03 after ~7h and three deliberate restarts (one mandated
resume test, one to add `--shuffle`, one to raise concurrency). Zero failed
requests across the whole run; `seed_harvest.jsonl.failed.jsonl` is empty.

| level | problems | rollouts | verify | solve@K | gate | usable | cap-hit | think med | answer med |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 901 | 1802 | 92.8% | 94.8% | 100.0% | 92.8% | 0.0% | 1344 | 281 |
| 2 | 5 | 10 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4115 | 476 |
| 3 | 115 | 230 | 77.8% | 80.0% | 100.0% | 77.8% | 0.0% | 3465 | 584 |
| 4 | 226 | 452 | 78.8% | 83.6% | 99.6% | 78.8% | 0.4% | 5566 | 746 |
| 5 | 725 | 1450 | 77.0% | 84.1% | 99.9% | 77.0% | 0.3% | 6565 | 809 |
| 6 | 1124 | 2248 | 77.3% | 85.1% | 99.5% | 77.3% | 0.6% | 7655 | 848 |
| 7 | 569 | 1138 | 74.4% | 82.8% | 99.6% | 74.3% | 0.4% | 8010 | 843 |
| 8 | 646 | 1292 | 64.2% | 74.5% | 99.9% | 64.2% | 0.1% | 9285 | 836 |
| 9 | 188 | 376 | 50.5% | 66.5% | 100.0% | 50.5% | 0.0% | 9290 | 838 |
| 10 | 1 | 2 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 7677 | 866 |
| **ALL** | **4500** | **9000** | **77.1%** | **84.1%** | **99.8%** | **77.1%** | **0.3%** | **6363** | **748** |

Gate failures: 22 in 9,000, all `missing_think_close` (the cap-hits).

**The packet's expectation was wrong in the good direction, and the reason is
the token budget.** It predicted the bulk harvest would land ~3 points *under*
the activity-003 probe's 73%; it landed **4 points above**. Per level against
that probe: L1 92.8% vs 86%, L5 **77.0% vs ~56%**, L9 50.5% vs 50%.

The probe ran at a 16,384-token cap with **9% cap-hits**; this harvest ran at
32,768 with **0.3%**. A cap-hit is a trace that never reaches an answer, so it
is scored wrong *and* has no answer segment — recovering those is worth several
points of yield on its own, and it is concentrated in the mid-levels where
traces are long enough to truncate but not hard enough to fail. Finding 1's
claim that the generous budget "converts waste into usable traces" is now
quantified: **+4 points of verify rate, not just +9 points of gate rate.**

`solve@K` (84.1%) exceeds `verify` (77.1%) by 7 points — that is the K=2 second
candidate rescuing problems whose first rollout failed, and it is the whole
justification for K=2 in this packet.

## Hand-inspected examples (packet deliverable, 5 problems, paired)

### gsm8k:95e3c9c6 — level 1 — gold `6600`
**problem:** Reese joined a Pinterest group where members contributed an average of 10 pins per day. The group owner deleted older pins at the rate of 5 pins per week per person. If the group has 20 people and the total number of pin

**verbose think (7040 tok) — first 400 chars:**
```
Okay, let's try to figure out this Pinterest group problem. So, Reese joined the group, and there are 20 people in total. The average number of pins contributed per day by members is 10. The group owner deletes older pins at a rate of 5 pins per week per person. The total number of pins is 1000, and we need to find out how many pins there will be after Reese has been a member for a month. 

First,…
```

**GLM compact (7 lines):**
```
goal: pins after 1 month; initial=1000, 20 people
let pins/day = 20·10 = 200; deletes/week = 20·5 = 100; month = 4 weeks
1. 4-week interpretation: pins_added = 200·30 = 6000; pins_deleted = 100·4 = 400
2. net = 1000 + 6000 − 400 = 6600
3. chk 28-day route: 200·28 − 100·4 = 5200 → 1000+5200 = 6200 ≠ 6600 ✗
4. chk 30-day fractional weeks: 100·(30/7) ≈ 428.6 → non-integer pins ✗
⇒ 6600
```

**Qwen3 compact (10 lines):**
```
goal: pins after 1 month
let total=1000
1. pins added per day: 20 people * 10 pins/day = 200
2. pins added per week: 200 * 7 = 1400
3. pins deleted per week: 20 people * 5 pins/week/person = 100
4. net pins per week: 1400 - 100 = 1300
5. pins added in 4 weeks: 1300 * 4 = 5200
6. total pins: 1000 + 5200 = 6200
chk: 1000 + 5200 = 6200 ✓
⇒ 6200
```

---

### deepmath:910457f1 — level 4 — gold `2 \sqrt{3}`
**problem:** Find the maximum value of \[f(x) = \sqrt{8x - x^2} - \sqrt{14x - x^2 - 48}.\]

**verbose think (8302 tok) — first 400 chars:**
```
Okay, so I need to find the maximum value of the function $ f(x) = \sqrt{8x - x^2} - \sqrt{14x - x^2 - 48} $. Hmm, let me start by understanding what this function looks like and where it's defined.

First, I know that square roots require their arguments to be non-negative. So, for $ \sqrt{8x - x^2} $, the expression inside the square root must be greater than or equal to zero. Let me write that…
```

**GLM compact (17 lines):**
```
goal: max f(x)=√(8x−x²)−√(14x−x²−48)
1. domain: 8x−x²≥0 → x(x−8)≤0 → x∈[0,8]
2. domain: 14x−x²−48≥0 → x²−14x+48≤0 → (x−6)(x−8)≤0 → x∈[6,8]
3. domain of f: [0,8]∩[6,8]=[6,8]
4. f'(x)=(4−x)/√(8x−x²) − (7−x)/√(14x−x²−48)
5. f'(x)=0: (4−x)/√(8x−x²) = (7−x)/√(14x−x²−48); square both sides
6. cross-multiply: (4−x)²(14x−x²−48)=(7−x)²(8x−x²)
7. expand LHS: (16−8x+x²)(−x²+14x−48) = −x⁴+22x³−176x²+608x−768
8. expand RHS: (49−14x+x²)(−x²+8x) = −x⁴+22x³−161x²+392x
9. set LHS=RHS, subtract: −15x²+216x−768=0 → 5x²−72x+256=0
10. quadratic: x=[72±√(5184−5120)]/10=[72±8]/10 → x=8 or x=6.4
11. chk x=8: (4−8)/√(0) → undefined ✗
12. chk x=6.4: LHS=(4−6.4)/√(10.24)=−2.4/3.2=−0.75; RHS=(7−6.4)/√(0.64)=0.6/0.8=0.75; LHS≠RHS ✗
13. both candidates extraneous ⇒ no critical pts in (6,8) ⇒ f decreasing on [6,8]
14. f(6)=√(48−36)−√(0)=√12=2√3
chk: f(8)=0−0=0 < 2√3 ✓
⇒ max=2√3
```

**Qwen3 compact (8 lines):**
```
goal: max value of f(x) = sqrt(8x - x²) - sqrt(14x - x² - 48)
let domain = [6, 8]
1. domain = [6, 8]
2. f(x) = sqrt(8x - x²) - sqrt(14x - x² - 48)
3. f(6) = sqrt(12) - 0 = 2√3
4. f(x) is decreasing on [6, 8] (derivative < 0)
5. max value at x=6
6. ⇒ 2√3
```

---

### deepmath:a98092b7 — level 6 — gold `24`
**problem:** Evaluate the expression \( \sin(5^{\circ})\sin(55^{\circ})\sin(65^{\circ}) = \frac{A\sqrt{B}+C\sqrt{D}}{E} \), where \( B \) and \( D \) are natural numbers not divisible by the square of any prime, \( A \) and \( C \) a

**verbose think (4465 tok) — first 400 chars:**
```
Okay, so I need to evaluate the expression $ \sin(5^{\circ})\sin(55^{\circ})\sin(65^{\circ}) $ and express it in the form $ \frac{A\sqrt{B}+C\sqrt{D}}{E} $, where B and D are natural numbers not divisible by the square of any prime, A and C are integers, and E is a natural number. Then find A + B + C + D + E.

First, I remember that there are some trigonometric identities that can help simplify pr…
```

**GLM compact (10 lines):**
```
goal: A+B+C+D+E from sin5°sin55°sin65°=(A√B+C√D)/E
1. sin5°sin55° = ½[cos(5−55)−cos(5+55)] = ½[cos(−50°)−cos60°] = ½[cos50°−½] = ½cos50°−¼
2. ×sin65°: ½cos50°sin65° − ¼sin65°
3. cos50°sin65° = ½[sin(50+65)+sin(65−50)] = ½[sin115°+sin15°]
4. expr = ¼[sin115°+sin15°] − ¼sin65° = ¼[sin115°+sin15°−sin65°]
5. sin115°=sin(180°−65°)=sin65° → ¼[sin65°+sin15°−sin65°] = ¼sin15°
6. sin15°=sin(45°−30°)=(√2/2)(√3/2)−(√2/2)(½)=(√6−√2)/4
7. expr=(√6−√2)/16 → A=1,B=6,C=−1,D=2,E=16
chk: √6≈2.449,√2≈1.414 → (2.449−1.414)/16≈1.035/16≈0.0647; sin5°≈0.0872,sin55°≈0.819,sin65°≈0.906 → prod≈0.0647 ✓
⇒ 1+6+(−1)+2+16=24
```

**Qwen3 compact (8 lines):**
```
goal: sin(5°)sin(55°)sin(65°) = (A√B + C√D)/E
1. sin(5°)sin(55°)sin(65°) = (1/4) sin(15°)
2. sin(15°) = (√6 - √2)/4
3. (1/4) sin(15°) = (√6 - √2)/16
4. Expression = (-√2 + √6)/16
5. A = -1, B = 2, C = 1, D = 6, E = 16
6. A + B + C + D + E = -1 + 2 + 1 + 6 + 16 = 24
⇒ 24
```

---

### deepmath:1bfd31b7 — level 8 — gold `-4\pi i`
**problem:** Evaluate the contour integral \( \oint_{|z|=2} \tan{z}\,dz \). Here, \( |z|=2 \) represents a circle of radius 2 centered at the origin in the complex plane.

**verbose think (5600 tok) — first 400 chars:**
```
Okay, so I need to evaluate the contour integral of tan(z) around the circle |z| = 2. Hmm, let me start by recalling some complex analysis. I know that integrating tan(z) over a closed contour can be tricky because tan(z) has singularities where cos(z) = 0, right? So first, I should figure out where these singularities are inside the contour |z| = 2.

The function tan(z) is defined as sin(z)/cos(z…
```

**GLM compact (11 lines):**
```
goal: ∮_{|z|=2} tan z dz
let tan z = sin z/cos z; poles where cos z=0 → z=π/2+πk
1. k=0: z=π/2≈1.57<2 ✓ inside; k=1: 3π/2≈4.71>2 ✗; k=−1: −π/2≈−1.57, |−1.57|<2 ✓ inside; k=±2: outside ✗
2. two poles inside: z=π/2, z=−π/2
3. residue formula for g/h at simple zero a of h: Res = g(a)/h'(a); here g=sin, h=cos, h'=−sin
4. Res at z=π/2: sin(π/2)/(−sin(π/2)) = 1/(−1) = −1
5. Res at z=−π/2: sin(−π/2)/(−sin(−π/2)) = (−1)/(1) = −1
6. Σ Res = −1+(−1) = −2
7. ∮ = 2πi·(Σ Res) = 2πi·(−2) = −4πi
chk: residue via direct limit at π/2: let w=z−π/2; sin(π/2+w)=cos w; cos(π/2+w)=−sin w; w·cos w/(−sin w)→−1 ✓ (same at −π/2)
⇒ −4πi
```

**Qwen3 compact (6 lines):**
```
goal: ∮_{|z|=2} tan(z) dz
1. find poles inside |z|=2: z=±π/2
2. residues at z=±π/2: -1 each
3. sum of residues: -1 + (-1) = -2
4. integral: 2πi * (-2) = -4πi
⇒ -4πi
```

---

### deepmath:917c8621 — level 9 — gold `Yes`
**problem:** Determine whether the cardinality of the set \( A = \{ D \subset [0,1] : |D| \leq |\mathbb{N}| \} \) is less than or equal to the cardinality of the real numbers \( \mathbb{R} \).

**verbose think (4667 tok) — first 400 chars:**
```
Okay, so I need to figure out whether the cardinality of the set A, which is all subsets D of [0,1] where the size of D is less than or equal to the cardinality of the natural numbers, is less than or equal to the cardinality of the real numbers. Let me start by recalling some basic concepts.

First, the cardinality of the natural numbers, ℵ₀, is the smallest infinite cardinality, and it's also th…
```

**GLM compact (10 lines):**
```
goal: |A| ≤ |ℝ|? where A={D⊂[0,1] : |D|≤|ℕ|}
let |ℕ|=ℵ₀, |ℝ|=2^{ℵ₀}
1. |D|≤ℵ₀ ⇒ D finite or countably infinite
2. finite subsets of [0,1]: |[0,1]_{<ω}| = ℵ₀
3. countably infinite subsets: each is image of an injection ℕ→[0,1]; |[0,1]^ℕ|=(2^{ℵ₀})^{ℵ₀}=2^{ℵ₀} ⇒ upper bound 2^{ℵ₀}
4. lower bound: each x∈[0,1] maps to countable set Q∪{x} ⇒ ≥2^{ℵ₀}
5. ⇒ |countably infinite subsets|=2^{ℵ₀}
6. |A|=ℵ₀∪2^{ℵ₀}=2^{ℵ₀}=|ℝ|
chk: A⊂𝒫([0,1]) ⇒ |A|≤2^{ℵ₀} ✓
⇒ Yes
```

**Qwen3 compact (9 lines):**
```
goal: cardinality of A ≤ cardinality of ℝ?
let A = {D ⊂ [0,1] : |D| ≤ |ℕ|}
1. |ℕ| = ℵ₀, so |D| ≤ ℵ₀ ⇒ D is finite or countably infinite
2. Set of all finite subsets of [0,1] is countable (ℵ₀)
3. Set of all countably infinite subsets of [0,1] is 2^ℵ₀
4. A = union of countable and 2^ℵ₀ ⇒ A has cardinality 2^ℵ₀
5. Cardinality of ℝ is 2^ℵ₀
6. Therefore, |A| = |ℝ|
⇒ Yes
```

---


---

## Conclusion

**P3 is done, in a materially different shape than the packet specified**, and
the packet's Part 2 work opened a design question that P3 could not contain —
recorded separately as [006](006-teacher-student-decoupling.md).

### Pinned for downstream packets

| quantity | value | source |
|---|---|---|
| **H_pivot** | **0.6707 nats** | p80 of think entropy, 1,200 **Qwen3-1.7B** register traces, 243,190 tokens |
| harvest verify rate | 77.1% (solve@K 84.1%) | 9,000/9,000 rollouts, 0 failures |
| parser gate / cap-hit | 99.8% / 0.3% | at `max_tokens = 32768` |
| Round-0 splits | train 960 / heldout_register 120 / probe_pool 120 | fixed seed, level-stratified, **do not re-split** |
| verbose control | 200 | disjoint from the register corpus |

**H_pivot overturns activity 004's flag.** 004 measured 0.2276 on a contaminated
corpus and warned of a ~3× drop from native CoT, interacting badly with
`Δ_max = 0.7` and TEA's `τ_c = 1.0`. The real value is within 3% of the
native-trace 0.6923; those concerns dissolve. (An intermediate 50-trace estimate
of 0.5067 was still 32% low — these percentiles need scale.)

### Corpora produced — four, paired on the same 1,200 inputs

| corpus | compressor | n (gated) | branch | verify | role |
|---|---|---|---|---|---|
| `seed_register_qwen` | Qwen3-1.7B | 1,200 | 3.1% | 26.2% | **Round 0, H_pivot** — unfiltered, must stay representative |
| `seed_register_qwen14b` | Qwen3-14B-FP8 | 1,200 (623) | 5.9% | 60.1% | scale reference |
| `seed_register_qwen32b` | Qwen3-32B-NVFP4 | 1,200 (699) | **13.9%** | 70.6% | **teacher corpus** (activity 006) |
| `seed_register_glm` | GLM-5.2 | 989 (806) | 39.9% | 95.9% | ceiling reference; carries `central_model_deviation` |

All four are `verify_response`-clean (0 failures) with 0 boxed-in-think after
cleaning, and structurally annotated.

### What changed from the packet

1. **Δlogp retired**, not re-thresholded. Any `P(gold | q, compact)` metric is
   dominated by whether the answer is literally in context, so it cannot
   separate deriving from asserting — measured three ways, including a fix
   (`--mask-conclusion`) that was implemented, tested and failed (findings 9, 11).
2. **`structural_gate.py` replaces it** — card §1.4's "never elided" column
   measured against each trace's own source, deterministic and model-free.
3. **Multiple corpora, not one**, with different compressors and consumers.

### What P3 discovered that P3 could not fix

Branch preservation — keeping the case splits and rejected alternatives the
verbose trace contains — is a **capability that appears with scale and cannot be
transferred by prompting**:

* it scales 3.1% → 5.9% → 13.9% across 1.7B/14B/32B, roughly a doubling per
  ~2.3× parameters (finding 15);
* **no prompting channel moves it**: static card exemplars, level-matched
  retrieval, and demo pools from 14B, 32B and frontier scale all leave the 1.7B
  at **1–2%** (finding 15);
* Qwen3-1.7B's own compressions are **40% faithful / 21% wrong**, dropping a
  branch on 46% of traces (finding 13, n=200, external judge).

This is not a P3 blocker — Round 0 consumes register-*token* statistics and
wants a corpus representative of what the student actually produces, which this
is. It is a **Stage-A** problem, and it is why activity 006 decouples the
teacher from the student.

### Open

* **The `G_spike` × branch-retention check** (activity 006) — `G_spike` rewards
  followability, and branch-preserving traces are longer and harder, so
  best-of-N may select *against* the property the 32B teacher exists to provide.
  **Gates P5.**
* **Structural-gate thresholds are provisional.** `branch_kept` is a corpus
  diagnostic, not a per-record gate: its source detector fires on 99.5% of
  traces, so gating on it becomes an unconditional notation requirement.
  Tightening the detector is prerequisite.
* **Think-segment grading needs a register-aware normaliser** (Unicode math ↔
  LaTeX). The deterministic verifier cannot grade compact-register conclusions
  because the register writes `4√2` where the verifier expects `4\sqrt{2}`
  (finding 14). It remains correct where the project uses it — the answer
  segment, which stays LaTeX.
* **Stage-B masking fraction is unmeasured** on a teacher corpus wider than the
  student.

### For P4

`H_pivot = 0.6707`. Use
`seed_register_qwen/{train,heldout_register,probe_pool,verbose_control}.jsonl`
and **do not re-split** — a re-split under a different seed moves probe traces
into training and silently invalidates the corrupted-trace probe. Do **not**
inoculate the scorer on any teacher corpus (14B/32B/GLM): Round 0 calibrates on
the *student's own* register statistics, and H_pivot reads 0.9119 on GLM text
against 0.6707 on the student's. P4's τ_spike correction in its packet header
still stands — the 4-nat design placeholder is dead on arrival for this
checkpoint.

### Method note — five measurement artifacts, each initially convincing

In order: step numbers counted as content (twice — an ad-hoc probe, then the
gate's first draft), comma-fusion of step back-references (`from 4,5:` → the
numeral 45), notation counted as invention, and Unicode-vs-LaTeX in conclusion
matching. **Every one produced a confident, inverted result**, and two were
reported to the user as findings before being caught. Separately, two
single-point over-reads inside the finding-15 sequence (a same-family transfer
advantage; a marker-density-tracks-pool effect) were retracted when the 32B arm
broke both.

The standing lesson: on this data, any metric derived from naive string
comparison between corpora is guilty until audited, and any pattern resting on
one arm should wait for a second.
