#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DRY=()
if [[ ${1:-} == --dry-run ]]; then DRY=(--dry-run); shift; fi
MODEL=${1:-all}
MODELS=(molmo2_er molmoact2 molmoact2_pretrain qwen35_4b qwen36_27b)
run_one() { "$ROOT/sspace/core/scripts/run_coco6000.sh" "${DRY[@]}" "$1"; }
if [[ $MODEL == all ]]; then
  for name in "${MODELS[@]}"; do run_one "$name"; done
  exit 0
fi
for name in "${MODELS[@]}"; do
  if [[ $MODEL == "$name" ]]; then run_one "$name"; exit 0; fi
done
printf 'Unknown model: %s\n' "$MODEL" >&2
exit 2

