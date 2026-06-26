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

# Container-side init runs scripts/install_deps.sh (idempotent), then dispatches
# on MODE. install_deps.sh lives on the bind-mounted workspace, so the container
# always picks up the latest version.
INIT="bash /workspace/scripts/install_deps.sh && cd /workspace"

DOCKER_TTY="-it"
SSH_FLAGS="-t"
case "$MODE" in
  smoke)
    SSH_CMD="$INIT && .venv/bin/python scripts/smoke_verify.py"
    DOCKER_TTY="-i"; SSH_FLAGS=""
    ;;
  cmd)
    SSH_CMD="$INIT && $CMD"
    DOCKER_TTY="-i"; SSH_FLAGS=""
    ;;
  *)
    SSH_CMD="$INIT && exec bash"
    ;;
esac

# We invoke docker over SSH so the container can be launched from the user's
# laptop. The container itself runs with --network host so the dev box's
# network is its own (matters for HF downloads and distributed training).
SSH_CMD_B64="$(printf '%s' "$SSH_CMD" | base64 | tr -d '\n')"
DOCKER_RUN="docker run --rm $DOCKER_TTY --gpus all --network host --ipc=host --shm-size=32g \
  -v $WORKSPACE_HOST:/workspace \
  -v $HF_CACHE_HOST:/root/.cache/huggingface \
  -v $DATA_ROOT_HOST:/workspace/data \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  $IMAGE bash -lc 'eval \"\$(echo $SSH_CMD_B64 | base64 -d)\"'"

exec ssh $SSH_FLAGS "$DEV_HOST" "$DOCKER_RUN"
