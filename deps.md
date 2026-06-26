---
name: cavethought-deps
description: >
  Full dependency installation for the CaveThought project on an Element GPU machine via SSH.
  Installs the proven stack: PyTorch 2.10.0+cu128, Flash Attention 2.8.3 (prebuilt wheel),
  Triton 3.6.0, vLLM 0.19.1, transformers, accelerate, peft, datasets, trl, tensorboard,
  liger-kernel, faiss-cpu, sentence-transformers, arize-phoenix, openinference,
  flash-linear-attention 0.5.0 + fla-core 0.5.0, and causal-conv1d 1.6.2.post1.
  Also applies the Triton jit.py inspect.getsourcelines fallback patch required for FLA
  (and Qwen3.5/Qwen3-Next via transformers) to import without the
  `^def\s+\w+\s*\(` regex returning None on multi-line `@triton.heuristics({lambda...})`
  decorator stacks. Default host is element-thought. Skips unsloth/bitsandbytes/xformers/torchao
  by design — CaveThought uses full-parameter FSDP2 SFT, not adapter bolt-ons.
  Use when the user asks to restore the CaveThought environment after a container reset,
  set up a fresh Element machine for CaveThought, or install CaveThought training/inference deps.
user-invocable: true
allowed-tools:
  - Bash(ssh *)
  - Bash(sshpass *)
  - Bash(cat *)
  - Bash(base64 *)
  - Bash(sleep *)
  - Read
  - Write
---

# /cavethought-deps — CaveThought dependency installation on Element GPU

Install the proven CaveThought stack on an Element GPU machine, from baseline container state to fully verified end-to-end (torch + flash-attn + vllm + FLA + causal-conv1d + Triton-patched).

Arguments passed: `$ARGUMENTS`

---

## Step 0 — Parse arguments

| Form | Meaning |
|---|---|
| empty | Default to `element-thought` SSH host |
| `<ssh-host>` | Use the given SSH host alias |
| `<ssh-host> --skip-fla` | Skip flash-linear-attention + causal-conv1d + Triton patch (faster; skip when you don't need FLA-based models like Qwen3.5/Qwen3-Next) |
| `<ssh-host> --skip-verify` | Skip the final deep-import verification |

Defaults:
- **ssh_host**: `element-thought`
- **password for SSH**: `element` (use `sshpass -p 'element' ssh ...`)

Always use `-o ControlMaster=no -o ControlPath=none` on the SSH invocations — element jump hosts get flaky with ControlMaster reuse across long sessions, and the container occasionally reprovisions mid-flight, leaving stale control sockets.

---

## Step 1 — Verify connectivity and capture container state

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "echo connected && /libraries/t9c8fa/bin/python3 -c \"
import importlib
for pkg in ['torch', 'flash_attn', 'vllm', 'transformers', 'trl', 'datasets', 'fla', 'causal_conv1d']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'installed')
        print(f'{pkg}: {v}')
    except ImportError:
        print(f'{pkg}: NOT INSTALLED')
\" && nvidia-smi 2>&1 | grep 'Driver Version' | head -1"
```

Expected baseline state on a fresh container: `torch 2.9.1+cu128`, `flash_attn 2.7.4`, everything else NOT INSTALLED. Anything else means the container has partial state — proceed anyway; pip will reconcile.

**Driver note:** element-thought containers oscillate between two driver versions:
- `535.183.06 / CUDA 12.2` — the common case
- `580.126.09 / CUDA 13.0` — rarer

Both work with `+cu128` wheels. **Do not** try torch+cu130 or vllm 0.20.2 — vllm 0.20.2 ships only a CUDA-13 wheel and fails on the 535 container with `libcudart.so.13: cannot open shared object file`. This has been tested. Stay on `+cu128`.

---

## Step 2 — Install PyTorch 2.10.0+cu128

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
   --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2"
```

Pulls in `triton 3.6.0` as a dep. Check `torch.__version__` shows `2.10.0+cu128` (not `+xpu` — if `+xpu`, the Intel package set leaked in; clean up via the recipe in `mininfo-dependency-install` Step 3 error handling).

---

## Step 3 — Install Flash Attention 2.8.3 (prebuilt wheel, with `--no-deps`)

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install --no-deps 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl' 2>&1 | tail -2"
```

The `--no-deps` flag is **required**: without it pip's resolver fails (the wheel pins torch 2.10 strictly but pip considers other transitive constraints already installed and aborts with `ResolutionImpossible`).

Auto-mode may flag the mjun0812 URL as untrusted third-party. It is the documented MinInfo / CaveThought wheel. If asked, approve it — it's the same wheel used across all Element machines in this project.

---

## Step 4 — Install vLLM 0.19.1 (with explicit constraints)

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install 'vllm==0.19.1' --constraint <(echo 'torch==2.10.0+cu128
torchaudio==2.10.0+cu128
torchvision==0.25.0+cu128') 2>&1 | tail -2"
```

This also pulls in `transformers 5.8.x` as a transitive. The explicit constraint protects the torch stack — vLLM is the most likely package to silently replace torch.

---

## Step 5 — Install training packages + Phoenix + TRL + datasets

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install accelerate peft tensorboard liger-kernel faiss-cpu sentence-transformers \
   'arize-phoenix>=8.0.0' 'openinference-semantic-conventions>=0.1.5' trl datasets 2>&1 | tail -2"
```

Versions land at approximately: accelerate 1.13.0, peft 0.19.1, liger-kernel 0.8.0, faiss-cpu 1.13.2, sentence-transformers 5.4.x–5.5.x, arize-phoenix 15.x, trl 1.4.0, datasets 4.8.5.

---

## Step 6 — Install flash-linear-attention + causal-conv1d *(skip if `--skip-fla`)*

### 6a — flash-linear-attention 0.5.0 + fla-core 0.5.0

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install flash-linear-attention 2>&1 | tail -3"
```

This installs both `flash-linear-attention 0.5.0` and its kernel package `fla-core 0.5.0`.

### 6b — causal-conv1d 1.6.2.post1 (CUDA source build, `--no-build-isolation` required)

```bash
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "pip3 install causal-conv1d --no-build-isolation 2>&1 | tail -5"
```

`--no-build-isolation` is **mandatory** — without it, pip creates a fresh build env that doesn't see the installed torch and fails dep resolution. With it, the package builds against the system torch and produces a `cp311` wheel in ~30 seconds.

---

## Step 7 — Patch Triton's `jit.py` for FLA compatibility *(skip if `--skip-fla`)*

**Why this patch is required:** FLA kernels use a decorator stack of
`@triton.heuristics({lambda args: ...})` → `@triton.autotune(...)` → `@triton.jit` → `def kernel(...)`.
Python's `inspect.getsourcelines(fn)` truncates the returned source at the closing `})` of
the multi-line heuristics dict (a Python `inspect` bug that has been present since at least
Python 3.10). Triton's `JITFunction.__init__` then runs:

```python
src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
```

The regex matches `None` (no `def name(` in the truncated string) and Triton crashes with
`AttributeError: 'NoneType' object has no attribute 'start'` at FLA import time. This
breaks the transformers loader for **Qwen3.5, Qwen3.5-MoE, and Qwen3-Next** (their
modeling files all `import fla`).

**The bug is present in every Triton version from 3.3 through 3.7** — downgrading does not
help. The fix is a 5-line fallback in `JITFunction.__init__`: when the regex returns
`None`, read the source file directly via `inspect.getsourcefile(fn)` and grep for the
function definition there.

Write the patch script locally, base64-encode it (avoids quoting hell over nested SSH
heredocs), and execute on the remote:

```bash
cat > /tmp/cavethought_patch_triton.py <<'PY'
import re
path = '/libraries/t9c8fa/lib/python3.11/site-packages/triton/runtime/jit.py'
with open(path) as f:
    src = f.read()
old = '        src = src[re.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE).start():]'
new = '''        _m = re.search(r"^def\\s+\\w+\\s*\\(", src, re.MULTILINE)
        if _m is None:
            try:
                import inspect as _ins
                _file = _ins.getsourcefile(fn)
                if _file:
                    with open(_file) as _f:
                        _full = _f.read()
                    _m = re.search(r"^def\\s+" + re.escape(fn.__name__) + r"\\s*\\(", _full, re.MULTILINE)
                    if _m is not None:
                        src = _full[_m.start():]
                    else:
                        raise RuntimeError("triton jit fallback: def not found for " + fn.__name__)
                else:
                    raise RuntimeError("triton jit fallback: no source file for " + fn.__name__)
            except Exception as _e:
                raise
        else:
            src = src[_m.start():]'''
if new.splitlines()[0] in src:
    print('ALREADY PATCHED')
elif old in src:
    src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('PATCHED', path)
else:
    print('PATTERN NOT FOUND - jit.py may have changed; inspect manually')
PY
b64=$(base64 -w0 /tmp/cavethought_patch_triton.py)
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "echo '$b64' | base64 -d > /tmp/patch_triton.py && /libraries/t9c8fa/bin/python3 /tmp/patch_triton.py"
```

Expected output: `PATCHED /libraries/t9c8fa/lib/python3.11/site-packages/triton/runtime/jit.py`.

If you see `PATTERN NOT FOUND`, the Triton version has changed its source layout — open `jit.py`, find the `re.search(r"^def\s+\w+\s*\(", ...)` line, and apply the same fallback pattern by hand.

---

## Step 8 — Verification *(skip if `--skip-verify`)*

Use a single in-process verify script that imports every key module (avoid per-module subprocess walks — they take ~20 minutes against 369 FLA submodules).

```bash
cat > /tmp/cavethought_verify.py <<'PY'
import torch, triton, flash_attn, transformers, accelerate, peft, liger_kernel, faiss, sentence_transformers, vllm, phoenix, trl, datasets
from vllm import LLM
liger_v = getattr(liger_kernel, '__version__', 'installed')
print(f'STACK | torch {torch.__version__} | cuda_ok {torch.cuda.is_available()} | triton {triton.__version__} | flash_attn {flash_attn.__version__} | trans {transformers.__version__} | accelerate {accelerate.__version__} | peft {peft.__version__} | liger {liger_v} | faiss {faiss.__version__} | st {sentence_transformers.__version__} | vllm {vllm.__version__} | phoenix {phoenix.__version__} | trl {trl.__version__} | datasets {datasets.__version__} | GPUs {torch.cuda.device_count()}')

# FLA section (only if Step 6 ran)
try:
    import fla, causal_conv1d
    import fla.ops.simple_gla.parallel  # exercises the Triton patch
    from causal_conv1d import causal_conv1d_fn
    fla_v = getattr(fla, '__version__', 'installed')
    print(f'FLA   | fla {fla_v} | causal_conv1d {causal_conv1d.__version__} | triton-patch effective')
except Exception as e:
    print(f'FLA   | NOT VERIFIED ({type(e).__name__}: {e})')

# Optional: confirm Qwen3.5 transformers loader path
try:
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_next import modeling_qwen3_next
    print('QWEN  | qwen3_5 + qwen3_next modeling imports OK')
except Exception as e:
    print(f'QWEN  | LOADER BROKEN ({type(e).__name__}: {e})')
PY
b64=$(base64 -w0 /tmp/cavethought_verify.py)
sshpass -p 'element' ssh -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none <ssh_host> \
  "echo '$b64' | base64 -d > /tmp/verify.py && LD_LIBRARY_PATH=/libraries/t9c8fa/lib:\$LD_LIBRARY_PATH /libraries/t9c8fa/bin/python3 /tmp/verify.py 2>&1 | grep -E '^(STACK|FLA|QWEN|Error|Traceback|.*Error)' | tail -10"
```

Expected:
```
STACK | torch 2.10.0+cu128 | cuda_ok True | triton 3.6.0 | flash_attn 2.8.3 | trans 5.8.x | accelerate 1.13.0 | peft 0.19.1 | liger 0.8.0 | faiss 1.13.2 | st 5.5.0 | vllm 0.19.1 | phoenix 15.x | trl 1.4.0 | datasets 4.8.5 | GPUs 8
FLA   | fla 0.5.0 | causal_conv1d 1.6.2.post1 | triton-patch effective
QWEN  | qwen3_5 + qwen3_next modeling imports OK
```

All three lines must appear without `Error` / `Traceback` for the install to be considered done.

---

## Step 9 — Report

Print a single summary table:

| Package | Expected version |
|---|---|
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 (patched jit.py) |
| Flash Attention | 2.8.3+cu128torch2.10 |
| vLLM | 0.19.1 |
| Transformers | 5.8.x |
| Accelerate | 1.13.0 |
| PEFT | 0.19.1 |
| Liger Kernel | 0.8.0 |
| FAISS | 1.13.2 |
| Sentence Transformers | 5.4–5.5 |
| Arize Phoenix | 15.x |
| TRL | 1.4.0 |
| Datasets | 4.8.5 |
| flash-linear-attention / fla-core | 0.5.0 |
| causal-conv1d | 1.6.2.post1 |

Remind the user:
- The Triton patch lives in `/libraries/t9c8fa/lib/python3.11/site-packages/triton/runtime/jit.py` — **site-packages is wiped on every container reset**, so the patch must be re-applied (re-run `/cavethought-deps`) whenever the container reprovisions.
- `~/.bashrc`, `~/pip-constraints.txt`, and `~/.config/pip/pip.conf` (if previously created by `/mininfo-dependency-install`) **do** persist across container resets. The CaveThought stack does not strictly require those, but they don't hurt.
- Unsloth, bitsandbytes, xformers, torchao are **intentionally skipped** — CaveThought uses full-parameter FSDP2 SFT (per plan v2 §6.7), not adapter bolt-on.
- 9 of FLA's 369 submodules fail to import (4 optional `tilelang` backend, 1 internal `delta_rule.parallel` symbol bug, 4 `deltaformer` circular imports). None affect the Qwen3.5 / Qwen3-Next loader path. Don't try to fix them; they are pre-existing upstream issues.

---

## Error handling

| Symptom | Action |
|---|---|
| `libcudart.so.13: cannot open shared object file` | Container has driver 535 / CUDA 12.2. Stay on torch+cu128 + vllm 0.19.1. Do **not** upgrade to vllm 0.20.2 (CUDA 13 only). |
| flash-attn install: `ResolutionImpossible` | Forgot `--no-deps` on the wheel install. Add it. |
| causal-conv1d build fails with "torch not found" | Forgot `--no-build-isolation`. Add it. |
| `AttributeError: 'NoneType' object has no attribute 'start'` at `import fla` | Triton patch missing or was wiped by container reset. Re-run Step 7. |
| `ModuleNotFoundError: No module named 'tilelang'` from `fla.ops.common.backends.tilelang.*` | Optional FLA backend, safe to ignore. |
| `cannot import name 'fwd_prepare_T' from 'fla.ops.delta_rule.wy_fast'` | Pre-existing FLA 0.5.0 upstream bug. Doesn't affect Qwen3.5 path. |
| SSH "Session open refused by peer" / "Connection closed by UNKNOWN port 65535" | Container is mid-reset. Wait 3–5 minutes and retry; check `/dev/tcp/<inner-host>/22` via the jump host to confirm sshd is back. |
| Container fully reprovisioned mid-install (site-packages wiped) | Re-run the entire skill from Step 2. `~/.bashrc`, constraints, and known-host entries persist; only site-packages doesn't. |
