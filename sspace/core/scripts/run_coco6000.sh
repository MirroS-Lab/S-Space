#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
MODEL=${1:?"usage: run_coco6000.sh [--dry-run] MODEL [CONFIG]"}
CONFIG_PATH=${2:-configs/sspace/train/train_${MODEL}_coco6000.json}

case "$MODEL" in
  molmo2_er) MODEL_ENV=.envs/spatialmqa; MASTER_PORT=29500 ;;
  molmoact2|molmoact2_pretrain) MODEL_ENV=.envs/molmoact; MASTER_PORT=29501 ;;
  qwen35_4b|qwen36_27b) MODEL_ENV=.envs/qwen36; MASTER_PORT=29502 ;;
  *) printf 'Unknown model: %s\n' "$MODEL" >&2; exit 2 ;;
esac

exec "$SCRIPT_DIR/_run_distributed.sh" \
  "$MODEL_ENV" "$MASTER_PORT" "$CONFIG_PATH" "$DRY_RUN"
