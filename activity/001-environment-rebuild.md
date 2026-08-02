# 001 — environment rebuild: turing (trainer) + spark (reward server)

- **Packet:** [packets/P0-environment.md](packets/P0-environment.md)
- **Status:** done
- **Machine(s):** turing, spark, mac
- **Code commit(s):** see "Commits" below
- **Started / finished:** 2026-08-01 → 2026-08-01

## Goal

Rebuild both GPU boxes from scratch for the Qwen3-1.7B feasibility tier, replacing the
stale v1/Gemma-era stack (`vllm==0.23.0`, liger-kernel, cu130-pinned torch). Establish
and commit a verified pin, and prove the two mechanisms every later packet rests on:
hybrid-Qwen3 generation **with thinking enabled** on turing, and teacher-forced
**prompt-logprob scoring** (d_t) on spark.

## Verified stack (identical on both machines)

| Package | Version |
|---|---|
| vllm | **0.26.0** |
| torch | **2.11.0+cu130** (CUDA 13.0) |
| transformers | 5.14.1 |
| triton | 3.6.0 |
| trl | 1.9.2 (turing only) |
| peft | 0.20.0 (turing only) |
| accelerate | 1.14.0 (turing only) |
| datasets | 5.0.1 (turing only) |
| safetensors | 0.8.0 |
| numpy | 2.3.5 |
| ninja | 1.13.0 |
| Python | CPython 3.12.12 (uv-managed, both) |

Both boxes install vLLM from **plain PyPI wheels** — no NGC container needed, including
on ARM. vllm 0.26.0's own metadata resolves torch 2.11.0+cu130 on aarch64, so the
container fallback in the packet was not exercised.

- turing: `torch.cuda.get_arch_list()` = `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`,
  device capability **(12, 0)** — RTX 5090 sm_120 is covered.
- spark: same arch list plus `sm_110`; device capability **(12, 1)** — GB10 is sm_**121**,
  which is *not* in the arch list but runs fine on the sm_120 cubins. See gotcha 3.

## Runs

### Run 1 — turing: layout + clone (2026-08-01 22:03)

```bash
ssh bajajra@192.168.1.220
mkdir -p /data/whetstone/{data,corpora,runs,ckpt,logs,eval}
git clone https://github.com/owlgebra-ai/whetstone.git ~/workspace/whetstone
cd ~/workspace/whetstone && rm -rf data logs
ln -s /data/whetstone/data data && ln -s /data/whetstone/logs logs
```

- **Deviation from packet:** the GitHub clone worked **anonymously** (`git ls-remote` on
  turing succeeded with no credentials) — the bare-repo fallback in the packet was not
  needed. turing's `origin` is the real GitHub remote; the Mac is not a sync hub.
- **Deviation from activity 000:** `~/workspace/whetstone` still existed (9.8 GB). Activity
  000's delete had partially failed: a **root-owned** `.venv` (~9.1 GB) survived, dragging
  `logs/` (658 MB of v1 gemma/sft smoke output) and `whetstone/patches/__pycache__` with it.
  Rather than `sudo rm` unattended, it was **moved aside** to
  `~/workspace/whetstone-v1-leftover` (reversible, no sudo needed for the rename since the
  parent is user-owned). **Still present — needs `sudo rm -rf` to reclaim 9.8 GB.**
- `.gitignore` covered `.venv` but **not** `data`/`logs`. Fixed on the Mac in this commit —
  without it, git tracks the two new symlinks.

### Run 2 — turing: venv + stack (2026-08-01 22:05)

```bash
cd ~/workspace/whetstone
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python vllm
uv pip install -p .venv/bin/python transformers accelerate datasets safetensors trl peft anthropic fire loguru pyyaml
uv pip install -p .venv/bin/python ninja      # NOT in the packet — see gotcha 1
```

Per packet: **no** liger-kernel, **no** `fla` extra, **no** separate flash-attn. Confirmed
absent from both venvs.

### Run 3 — turing: verification gauntlet (2026-08-01 22:10)

```bash
cd ~/workspace/whetstone && source .venv/bin/activate
python scripts/smoke_verify.py
python scripts/smoke_qwen3_thinking.py
```

- `smoke_verify.py` → **10/10 pass**, `extract_answer(post-think): 99`.
- `smoke_qwen3_thinking.py` (new, committed) → **PASS**:
  ```
  think_tokens=1412  answer_tokens=201
  extracted_answer='391'  gold='391'
  PASS: thinking honored + answer correct
  ```
  Closed `<think>…</think>` block present, `\boxed{391}` after it, verified through the
  deterministic verifier rather than a string match.

Segment lengths are printed separately (think vs answer) per the CLAUDE.md invariant.

### Run 4 — spark: venv + stack (2026-08-01 22:08)

```bash
ssh bajajra@192.168.1.253
mkdir -p ~/workspace/whetstone-scorer && cd ~/workspace/whetstone-scorer
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python vllm ninja
```

- aarch64 CUDA wheels resolved from PyPI on the first try (vllm 0.26.0 / torch 2.11.0+cu130).
  **Container route not needed.**
- **HF cache:** Qwen3-1.7B was already in spark's *local* cache at
  `/srv/cache/hf/hub/models--Qwen--Qwen3-1.7B` (spark's `HF_HOME` is `/srv/cache/hf`), so
  packet option (b) holds with no download. NFS option (a) not used — no lock-timeout risk.

### Run 5 — spark: prompt-logprob scoring check (2026-08-01 22:16)

```bash
cd ~/workspace/whetstone-scorer && source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/smoke_scorer_logprobs.py
```

First attempt **failed** — see gotcha 2. With the flag, **PASS**:

```
pos token        rank  lp_actual   lp_top1      d_t
  1 ' capital'   12951   -11.3579   -2.4829   8.8750
  2 ' of'           1    -0.4130   -0.4130   0.0000
  3 ' France'       5    -4.2801   -0.6551   3.6250
  4 ' is'           1    -0.0518   -0.0518   0.0000
  5 ' Paris'        1    -0.6331   -0.6331   0.0000
  6 '.'             1    -0.7144   -0.7144   0.0000

PASS: d_t computable at 6/6 positions; 4 rank-1 positions all had d_t == 0
```

Both contract requirements hold:

1. **The actual token is always returned even far outside top-k.** Position 1 (` capital`)
   ranked **12951** with `prompt_logprobs=2` and still came back with its logprob. This is
   the property G_spike depends on; without it, large gaps — exactly the interesting ones —
   would be unmeasurable.
2. **d_t == 0 exactly at rank-1 positions** (4/4), and d_t ≥ 0 everywhere.

`prompt_logprobs` returns one entry per prompt token with `pl[0] is None` (nothing
conditions position 0); the script asserts `len(pl) == len(input_ids)`.

### Run 6 — cross-machine reachability (2026-08-01 22:19)

Reward-server launch command (**record this — later packets need it verbatim**):

```bash
# on spark
cd ~/workspace/whetstone-scorer && source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-1.7B \
  --port 8100 --host 0.0.0.0 \
  --gpu-memory-utilization 0.35 --max-model-len 8192 \
  --served-model-name whetstone-scorer
```

- **Port changed 8000 → 8100.** Port 8000 on spark is **already owned by an unrelated
  `llama-swap` service** (serving `judge`, `qwen2b`, `qwen35`, `qwen35-spark`, `qwen4b`, …).
  The packet's `vllm serve --port 8000` died with `OSError: [Errno 98] Address already in
  use`, and the packet's `curl …:8000/v1/models` check **passes against llama-swap** — a
  false green. The pre-existing service was left untouched. **Every later packet must use
  8100.**
- Reachable from turing on **both** addresses:
  - LAN: `http://192.168.1.253:8100`
  - direct link: `http://198.18.0.1:8100` (turing is `198.18.0.2`, spark is `198.18.0.1`
    on `198.18.0.0/24` — the same link `/data` is NFS-mounted over)
- **d_t verified over HTTP too**, not just in-process — the trainer's actual call path:

  ```bash
  curl -s http://198.18.0.1:8100/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"whetstone-scorer","prompt":"The capital of France is Paris.","max_tokens":1,"temperature":0,"prompt_logprobs":2}'
  ```

  Returns `prompt_logprobs` with 7 entries (`[0] = null`), each a dict keyed by token id
  with `logprob` / `rank` / `decoded_token`. Values match the in-process run exactly
  (` capital` → `-11.358`, rank `12951`). Scoring can therefore be done remotely over
  HTTP with no local model copy on turing.
- The server was **shut down** after the check; the box is left clean. Restart with the
  command above.

## Gotchas for the next agent

1. **Activate the venv — never call `.venv/bin/python` directly.** vLLM's engine startup
   shells out to `ninja` **by name** for torch.compile codegen. `ninja` is a venv-installed
   *binary*; invoking `.venv/bin/python` without activation leaves `.venv/bin` off `PATH`,
   and the engine dies with `FileNotFoundError: [Errno 2] No such file or directory:
   'ninja'` wrapped in a `RuntimeError: Engine core initialization failed` whose real cause
   is ~200 lines up the log. Use `source .venv/bin/activate` (the packet's own commands use
   the direct-path form and hit this). `ninja` is now a hard dependency in `pyproject.toml`.
2. **spark needs `VLLM_USE_FLASHINFER_SAMPLER=0`.** On GB10, FlashInfer's `check_cuda_arch()`
   finds no eligible arch in `current_compilation_context.TARGET_CUDA_ARCHS` and raises the
   *misleading* `RuntimeError: FlashInfer requires GPUs with sm75 or higher` — from the
   **sampler** JIT path (`flashinfer/jit/sampling.py`), not attention. Attention is fine.
   Disabling the FlashInfer sampler falls back to vLLM's native top-k/top-p and costs
   nothing here: the scorer is **prefill-only** (`max_tokens=1`), so it never really samples.
   Alternative (untested) fix: `export TORCH_CUDA_ARCH_LIST=12.1`. turing does **not** need
   the flag.
3. **GB10 is sm_121, and the wheels only ship up to sm_120** — it works anyway (forward
   compatibility within Blackwell), but expect the occasional `sm121`-shaped warning and a
   Triton JIT spike on first use (`_topk_log_softmax_kernel` compiled during inference).
   Extend warmup if scorer latency matters.
4. **Port 8000 on spark is taken** by `llama-swap`. Use 8100. Do not kill llama-swap.
5. **9.8 GB is still reclaimable on turing** at `~/workspace/whetstone-v1-leftover` — needs
   `sudo rm -rf` because of the root-owned `.venv`. Deliberately left in place (user
   decision, 2026-08-01); reclaim with
   `sudo rm -rf ~/workspace/whetstone-v1-leftover` when convenient.
6. **turing's checkout is behind the Mac.** The two commits below are **local to the Mac —
   not pushed** (user decision, 2026-08-01). turing's clone sits at `6191564` with the new
   `pyproject.toml` and `scripts/smoke_qwen3_thinking.py` **hand-copied in via scp**, so its
   working tree is dirty relative to its own HEAD. Before P1 does anything, either
   `git push origin main` from the Mac and `git pull` on turing, or re-scp — do not assume
   turing has the committed state. (`scripts/smoke_scorer_logprobs.py` lives only on spark
   at `~/workspace/whetstone-scorer/scripts/` and on the Mac; it was never copied to turing.)

## Commits

- `pyproject.toml` — pinned to the verified stack: `vllm==0.26.0`, `transformers>=5.14.1`,
  `trl>=1.9.2`, `peft>=0.20.0`, `datasets>=5.0.1`, `safetensors>=0.8`, added `ninja>=1.13`,
  `requires-python >= 3.12`. `liger-kernel` **moved out of core deps** into a `gemma-v1`
  extra (kept, not deleted — v1 historical reference).
- `.gitignore` — added `/data` and `/logs`.
- `scripts/smoke_qwen3_thinking.py` — new (turing gate).
- `scripts/smoke_scorer_logprobs.py` — new (spark gate; the d_t contract test).

## Conclusion

**P0 is done — all Part-4 checks pass.** Both machines run the same verified stack
(vllm 0.26.0 / torch 2.11.0+cu130 / transformers 5.14.1 on CPython 3.12.12) from plain
PyPI wheels, x86_64 and aarch64 alike. Blackwell kernels work on both. Hybrid-Qwen3
thinking is honored end-to-end on turing, and the d_t primitive underneath G_spike, the
Round-0 meter and the ZPD gates is proven both in-process and over HTTP from turing.

P1–P4 are unblocked. Two facts they must carry: the scorer lives on **port 8100** (8000 is
occupied), and it must be launched with **`VLLM_USE_FLASHINFER_SAMPLER=0`**.
