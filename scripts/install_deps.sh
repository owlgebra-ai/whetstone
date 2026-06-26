#!/usr/bin/env bash
# Install whetstone's full dependency stack into a uv venv on /workspace.
#
# Designed to run inside the nvcr.io/nvidia/pytorch:26.05-py3 container (cu130
# torch, Python 3.12) but works on any host with a system Python ≥3.11 and
# network access. Idempotent — re-running is cheap.
#
# What it does (in order):
#   1. Install uv if missing (lives under /root or $HOME — wiped each
#      `docker run --rm`, so reinstall is the expected path inside the container).
#   2. Create .venv on the bind-mounted workspace, based on the container's
#      system python so the symlinks survive `docker run --rm`.
#      --system-site-packages lets the venv reuse the heavy NGC libs
#      (torch, vllm, transformers, etc.) — uv then layers whetstone-only
#      pins on top.
#   3. uv pip install -e . (editable whetstone + deps from pyproject.toml).
#   4. Patch peft's is_te_available() to short-circuit. NGC ships a
#      transformer_engine native lib that dlopens libcublasLt.so.13 looking
#      for a symbol our cu130 torch's cuBLAS doesn't expose
#      (undefined cublasLtGroupedMatrixLayoutInit_internal). peft >=0.16's
#      is_te_pytorch_available imports transformer_engine without
#      try/except OSError, so the dlopen failure leaks all the way to user
#      code. Patching is_te_available to return False side-steps that path.
#      The .venv lives on the bind mount so the patch persists.
#   5. Print torch/transformers/vllm versions for sanity.
#
# Usage:
#   bash scripts/install_deps.sh                 # default workspace /workspace
#   WHETSTONE_WORKSPACE=/path bash scripts/install_deps.sh
#
# Env:
#   WHETSTONE_WORKSPACE  directory holding pyproject.toml (default /workspace).
#   WHETSTONE_PYTHON     interpreter to base the venv on
#                        (default /usr/bin/python3.12 if present, else python3).
#   WHETSTONE_FORCE      if set, force-rebuild the venv and re-run uv pip install.
set -euo pipefail

WORKSPACE="${WHETSTONE_WORKSPACE:-/workspace}"
if [[ -n "${WHETSTONE_PYTHON:-}" ]]; then
  BASE_PY="$WHETSTONE_PYTHON"
elif [[ -x /usr/bin/python3.12 ]]; then
  BASE_PY=/usr/bin/python3.12
else
  BASE_PY="$(command -v python3)"
fi

cd "$WORKSPACE"

# --- 1. uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo "[deps] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="${HOME:-/root}/.local/bin:$PATH"

# --- 2. venv ---
if [[ -n "${WHETSTONE_FORCE:-}" ]] || [[ ! -x .venv/bin/python ]]; then
  echo "[deps] creating venv from $BASE_PY..."
  rm -rf .venv
  uv venv --python "$BASE_PY" --system-site-packages .venv
fi

# --- 3. whetstone deps ---
MARKER="$WORKSPACE/.deps_installed.marker"
if [[ -n "${WHETSTONE_FORCE:-}" ]] || [[ ! -f "$MARKER" ]] || [[ pyproject.toml -nt "$MARKER" ]]; then
  echo "[deps] layering whetstone deps onto .venv..."
  uv pip install --python .venv/bin/python -e .
  touch "$MARKER"
fi

# --- 4. peft transformer_engine workaround ---
PEFT_IU=".venv/lib/python3.12/site-packages/peft/import_utils.py"
if [[ -f "$PEFT_IU" ]] && ! grep -q 'WHETSTONE_TE_PATCH' "$PEFT_IU"; then
  echo "[deps] patching peft is_te_available (NGC cuBLAS-Lt ABI mismatch)..."
  python - <<PY
import re, pathlib
p = pathlib.Path("$PEFT_IU")
src = p.read_text()
src = re.sub(
    r"def is_te_available\(\):\s*\n(\s*)return .+\n",
    r"def is_te_available():  # WHETSTONE_TE_PATCH\n\1return False\n",
    src, count=1,
)
p.write_text(src)
PY
fi

# --- 5. report ---
echo "[deps] ready. versions:"
.venv/bin/python - <<'PY'
import torch, transformers
try:
    import vllm
    vllm_v = vllm.__version__
except Exception as e:
    vllm_v = f"IMPORT_FAIL: {e}"
print(f"  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}")
print(f"  vllm={vllm_v}")
print(f"  gpus={torch.cuda.device_count()}")
PY
