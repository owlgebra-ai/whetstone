#!/usr/bin/env bash
# Launch the WHETSTONE dev container (nvcr.io/nvidia/pytorch:26.05-py3) on the
# dev box. Mounts the workspace and HF cache, sets up uv on first run, and
# drops into a shell by default. Pass `--smoke` to run a one-shot smoke test
# instead of an interactive shell.
#
# Usage:
#   scripts/dev_container.sh                 # interactive bash
#   scripts/dev_container.sh --smoke         # run smoke tests, then exit
#   scripts/dev_container.sh --cmd "..."     # run a custom command
#
# Container layout (inside):
#   /workspace           -> host /home/bajajra/workspace/whetstone (rw)
#   /root/.cache/huggingface -> host ~/.cache/huggingface (rw, for downloads)
#   /workspace/.venv     -> uv-managed venv shared across container runs
set -euo pipefail

IMAGE="${WHETSTONE_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
DEV_HOST="${WHETSTONE_DEV_HOST:-bajajra@192.168.1.220}"
WORKSPACE_HOST="${WHETSTONE_WORKSPACE_HOST:-/home/bajajra/workspace/whetstone}"
HF_CACHE_HOST="${WHETSTONE_HF_CACHE_HOST:-/home/bajajra/.cache/huggingface}"
DATA_ROOT_HOST="${WHETSTONE_DATA_ROOT:-/home/bajajra/workspace/whetstone/data}"

MODE="interactive"
CMD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift;;
    --cmd) MODE="cmd"; CMD="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Container-side init: install uv (if missing), restore the venv, run smoke tests
# when requested. Heredoc is base64-escaped below to dodge nested-SSH quoting.
read -r -d '' DEV_INIT <<'EOF' || true
set -euo pipefail
cd /workspace
# uv lives at /root/.local/bin which gets wiped on each --rm; reinstall is ~5s.
if ! command -v uv >/dev/null 2>&1; then
  echo "[init] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="/root/.local/bin:$PATH"
# Build .venv on the bind-mounted workspace, but base it on the NGC container's
# system python (/usr/bin/python3.12) which lives in the image and survives --rm.
# --system-site-packages lets the venv reuse the heavy NGC libs (torch, vllm,
# transformers, etc.) — uv then layers in whetstone-only pins on top.
if [[ ! -x .venv/bin/python ]]; then
  echo "[init] creating venv from NGC system python..."
  rm -rf .venv
  uv venv --python /usr/bin/python3.12 --system-site-packages .venv
fi
MARKER=/workspace/.deps_installed.marker
if [[ ! -f "$MARKER" ]] || [[ pyproject.toml -nt "$MARKER" ]]; then
  echo "[init] layering whetstone deps onto .venv..."
  uv pip install --python .venv/bin/python -e .
  touch "$MARKER"
fi
# NGC's transformer_engine native lib was linked against a cuBLAS-Lt ABI that
# our cu130 torch's cuBLAS doesn't expose (undefined cublasLtGroupedMatrixLayoutInit_internal).
# peft >=0.16's is_te_pytorch_available imports transformer_engine without a
# try/except — the OSError from dlopen leaks all the way to user code. Patch
# peft to short-circuit TE detection to False. The .venv lives on the bind
# mount so the patch persists across container restarts.
PEFT_IU=/workspace/.venv/lib/python3.12/site-packages/peft/import_utils.py
if [[ -f "$PEFT_IU" ]] && ! grep -q 'WHETSTONE_TE_PATCH' "$PEFT_IU"; then
  echo "[init] patching peft is_te_available to short-circuit (NGC ABI mismatch)..."
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
echo "[init] ready. versions:"
.venv/bin/python -c "
import torch, transformers
try: import vllm; vllm_v = vllm.__version__
except Exception as e: vllm_v = f'IMPORT_FAIL: {e}'
print(f'  torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__}')
print(f'  vllm={vllm_v}')
print(f'  gpus={torch.cuda.device_count()}')
"
EOF

DEV_INIT_B64="$(echo "$DEV_INIT" | base64 -w0)"

DOCKER_TTY="-it"
SSH_FLAGS="-t"
case "$MODE" in
  smoke)
    SSH_CMD="echo '$DEV_INIT_B64' | base64 -d > /tmp/dev_init.sh && bash /tmp/dev_init.sh && cd /workspace && .venv/bin/python scripts/smoke_verify.py"
    DOCKER_TTY="-i"; SSH_FLAGS=""
    ;;
  cmd)
    SSH_CMD="echo '$DEV_INIT_B64' | base64 -d > /tmp/dev_init.sh && bash /tmp/dev_init.sh && cd /workspace && $CMD"
    DOCKER_TTY="-i"; SSH_FLAGS=""
    ;;
  *)
    SSH_CMD="echo '$DEV_INIT_B64' | base64 -d > /tmp/dev_init.sh && bash /tmp/dev_init.sh && cd /workspace && exec bash"
    ;;
esac

# We invoke docker over SSH so the container can be launched from the user's
# laptop. The container itself runs with --network host so the dev box's
# network is its own (matters for HF downloads and distributed training).
DOCKER_RUN="docker run --rm $DOCKER_TTY --gpus all --network host --ipc=host --shm-size=32g \
  -v $WORKSPACE_HOST:/workspace \
  -v $HF_CACHE_HOST:/root/.cache/huggingface \
  -v $DATA_ROOT_HOST:/workspace/data \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  $IMAGE bash -lc '$SSH_CMD'"

exec ssh $SSH_FLAGS "$DEV_HOST" "$DOCKER_RUN"
