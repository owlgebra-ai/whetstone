# P0 — Environment rebuild: turing (trainer) + spark (reward server)

STATUS: done (activity 001)
MACHINES: turing (192.168.1.220, x86_64, RTX 5090 32GB), spark (192.168.1.253, aarch64, GB10)
DEPENDS ON: activity 000 (done)
BLOCKS: P1, P2, P3, P4
DELIVERABLES: working venvs on both machines, verified Qwen3-1.7B generation with thinking on turing, verified prompt-logprob scoring on spark, updated pyproject pin committed, activity journal.

## Objective

Rebuild both machines from scratch for the Qwen3-1.7B feasibility tier. The old environment is gone (activity 000). The v1 pyproject pins (`vllm==0.23.0`, Gemma-era stack) are stale — the user has asked for a vLLM upgrade; this packet establishes the new verified pin.

## Read first

- [CLAUDE.md](../../CLAUDE.md) — Environment section
- [activity/000-turing-reset.md](../000-turing-reset.md) — machine survey
- Design doc §11 (models), §12.2 (scoring passes), §12.5 (topology)

## Part 1 — Directory layout (turing)

All big artifacts live on `/data` (shared with spark over NFS). Root disk holds only code + venv.

```bash
ssh bajajra@192.168.1.220
mkdir -p /data/whetstone/{data,corpora,runs,ckpt,logs,eval}
```

Clone the repo to `~/workspace/whetstone`:

```bash
git clone https://github.com/owlgebra-ai/whetstone.git ~/workspace/whetstone
```

**Gotcha — auth:** the repo is under the `owlgebra-ai` org and may be private. If the clone fails with auth errors, do NOT paste tokens into shell commands. Fallback that needs no GitHub credentials on turing: create a bare repo and push from the Mac —

```bash
# on turing
git init --bare /data/git/whetstone.git
# on the Mac
git -C ~/git/whetstone remote add turing ssh://bajajra@192.168.1.220/data/git/whetstone.git
git -C ~/git/whetstone push turing main
# on turing
git clone /data/git/whetstone.git ~/workspace/whetstone
```

If you use the fallback, record in your activity file that turing's `origin` is the bare repo, and that the Mac is the sync hub (Mac pushes to both GitHub and turing).

Inside the clone, symlink the artifact dirs so v1 scripts' relative paths (`data/`, `logs/`) land on `/data`:

```bash
cd ~/workspace/whetstone
rm -rf data logs
ln -s /data/whetstone/data data
ln -s /data/whetstone/logs logs
```

Verify `.gitignore` covers `data`, `logs`, `.venv` (it should — check, don't assume).

## Part 2 — Python env (turing)

System python is **3.14.4 — do not use it**; the ML stack won't have wheels. Use uv-managed CPython 3.12:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv absent; check `which uv` first
cd ~/workspace/whetstone
uv python install 3.12
uv venv --python 3.12 .venv
```

Install order matters. vLLM pins its own torch; install vLLM first and let it bring torch, then layer the training stack:

```bash
uv pip install -p .venv/bin/python vllm            # latest stable — this IS the upgrade
uv pip install -p .venv/bin/python transformers accelerate datasets safetensors trl peft anthropic fire loguru pyyaml
```

**Gotchas:**

1. **Blackwell (sm_120).** The RTX 5090 needs CUDA ≥ 12.8 kernels. Any 2026-era vLLM release ships them, but *verify* (Part 4) before declaring success. If you hit `no kernel image is available for execution on the device`, the wheel's torch is too old — install torch from the `cu130` index first (see `[tool.uv.sources]` in pyproject) and vLLM second with `--no-deps` resolution care, and write down exactly what worked.
2. **Do not install `liger-kernel` or the `fla` extra.** Both are v1/Gemma-era or Qwen3-Next-only. `whetstone/patches/gemma4_*` must not be imported anywhere in the Qwen3 path.
3. **Do not install flash-attn separately.** vLLM vendors its attention; the HF-side forward passes (entropy audit, SED) run fine with sdpa at 1.7B.
4. **Pin what you verified.** After Part 4 passes, update `pyproject.toml`: bump `vllm==<verified>`, `transformers>=<installed>`, drop `liger-kernel` from core deps (move to a `gemma-v1` extra if you want to preserve it), and commit with the activity file. The next agent must be able to `uv pip install -e .` and get your exact working stack.

## Part 3 — Python env (spark) — ARM, different rules

```bash
ssh bajajra@192.168.1.253
mkdir -p ~/workspace/whetstone-scorer && cd ~/workspace/whetstone-scorer
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python vllm
```

**Gotchas:**

1. **aarch64 wheels.** Recent vLLM publishes CUDA aarch64 wheels (GB200/DGX-Spark era). If `uv pip install vllm` resolves to a CPU-only or missing wheel, use NVIDIA's container instead: `docker run --gpus all -v /data:/data nvcr.io/nvidia/vllm:<latest>` (the DGX Spark playbooks route). Record which path worked; the reward-server launch script must match it.
2. **HF cache on spark.** Spark's local cache lacks Qwen3-1.7B. Two options: (a) `export HF_HOME=/data/cache/huggingface` to reuse turing's cache over NFS — but HF file locking over NFS can wedge; if you see lock timeouts, fall back to (b) local download (`hf download Qwen/Qwen3-1.7B`, ~4 GB, fine on the 1.4 TB root). Prefer (b) for the always-on reward server; NFS reads on the model's mmap'd weights add latency at startup only, but locks are a real failure mode.
3. **GB10 memory is unified** — `nvidia-smi` shows N/A. Give vLLM an explicit budget: `--gpu-memory-utilization 0.35` is plenty for a 1.7B scorer and leaves the box usable for data prep.
4. Spark serves **scoring only** (prefill-bound). Never schedule rollout *generation* here — GB10's memory bandwidth makes long decode painfully slow; that work belongs to the 5090.

## Part 4 — Verification gauntlet (both machines; all must pass)

**turing:**

```bash
cd ~/workspace/whetstone
.venv/bin/python scripts/smoke_verify.py                    # verifier logic, no GPU
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

Then a real generation with thinking enabled (write as `scripts/smoke_qwen3_thinking.py`, commit it):

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 * 23?"}],
    tokenize=False, add_generation_prompt=True,
    enable_thinking=True,          # REQUIRED on every hybrid-Qwen3 call (design §11)
)
llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=8192, gpu_memory_utilization=0.85)
out = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=2048))[0].outputs[0].text
assert "</think>" in out, "no think block — enable_thinking not honored"
print(out[-500:])
```

Pass = output contains a `<think>…</think>` block and the answer 391 after it.

**spark** — verify the exact scoring mechanism the pipeline depends on (design §12.2: one teacher-forced prefill returning per-position actual-token logprob AND top-1 logprob, i.e. `prompt_logprobs ≥ 2`):

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=8192, gpu_memory_utilization=0.35)
sp = SamplingParams(max_tokens=1, prompt_logprobs=2)
out = llm.generate(["The capital of France is Paris."], sp)[0]
pl = out.prompt_logprobs
assert pl is not None and any(p is not None and len(p) >= 1 for p in pl)
# For at least one position, confirm you can read BOTH the actual token's logprob
# and the rank-1 token's logprob out of the returned dict. Print one example.
```

**Gotcha:** `prompt_logprobs` entries are dicts keyed by token id; the actual token is always included even when outside top-k, with a `rank` attribute. d_t = (rank-1 logprob) − (actual-token logprob). Confirm you can compute this for a made-up sentence and that d_t = 0 wherever the actual token IS rank 1. This one-liner check is the foundation of G_spike, Round-0 metrics, and ZPD gates — get it airtight now.

Also verify cross-machine reachability (the trainer will call the scorer over HTTP later):

```bash
# on spark: .venv/bin/vllm serve Qwen/Qwen3-1.7B --port 8000 --gpu-memory-utilization 0.35
# on turing:
curl -s http://192.168.1.253:8000/v1/models | head -c 300
```

If turing can also reach spark via the direct-link subnet (`ip route | grep 198.18`), note both addresses; the LAN address is fine for scoring payloads.

## Definition of done

- [ ] All Part-4 checks pass; outputs pasted into the activity file.
- [ ] `pyproject.toml` updated to the verified stack and committed.
- [ ] `scripts/smoke_qwen3_thinking.py` and the spark scoring check committed.
- [ ] Activity file `NNN-environment-rebuild.md` written: exact versions (`uv pip list | grep -E 'vllm|torch|transformers|trl'` on both machines), which install path spark needed (wheel vs container), reward-server launch command, and any deviation from this packet.
- [ ] Packet status flipped to `done (activity NNN)`.
