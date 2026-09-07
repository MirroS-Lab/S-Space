#!/usr/bin/env bash
set -euo pipefail

MODEL_ENV=$1
MASTER_PORT=$2
CONFIG_PATH=$3
DRY_RUN=${4:-0}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi
MODEL_PYTHON="$PROJECT_ROOT/$MODEL_ENV/bin/python"
CONFIG_PYTHON=${PYTHON:-python3}
WORLD_SIZE=$(
  "$CONFIG_PYTHON" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1]))["world_size"]))' \
    "$CONFIG_PATH"
)
if [[ $WORLD_SIZE -le 0 ]]; then
  printf 'Config world_size must be positive: %s\n' "$WORLD_SIZE" >&2
  exit 2
fi

if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then
  CUDA_VISIBLE_DEVICES=0
  for ((DEVICE = 1; DEVICE < WORLD_SIZE; DEVICE++)); do
    CUDA_VISIBLE_DEVICES+=",$DEVICE"
  done
  export CUDA_VISIBLE_DEVICES
fi
IFS=',' read -r -a VISIBLE_DEVICES <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#VISIBLE_DEVICES[@]} -ne $WORLD_SIZE ]]; then
  printf 'Expected %s visible GPUs, found %s in CUDA_VISIBLE_DEVICES=%s\n' \
    "$WORLD_SIZE" "${#VISIBLE_DEVICES[@]}" "$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export TMPDIR="$PROJECT_ROOT/.tmp"
export PYTHONPATH="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -n ${CUBLAS_WORKSPACE_CONFIG+x} && $CUBLAS_WORKSPACE_CONFIG != :4096:8 ]]; then
  printf 'CUBLAS_WORKSPACE_CONFIG must be unset or equal to :4096:8\n' >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$TMPDIR"
cd "$PROJECT_ROOT"
COMMAND=(
  "$PROJECT_ROOT/$MODEL_ENV/bin/torchrun"
  --standalone \
  --master-port="$MASTER_PORT" \
  --nproc-per-node="$WORLD_SIZE" \
  -m sspace.core.cli \
  --config "$CONFIG_PATH"
)
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf 'CUBLAS_WORKSPACE_CONFIG=%q ' "$CUBLAS_WORKSPACE_CONFIG"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -x "$MODEL_PYTHON" ]]; then
  printf 'Missing model environment Python: %s\n' "$MODEL_PYTHON" >&2
  exit 2
fi
exec "${COMMAND[@]}"
