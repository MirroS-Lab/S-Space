#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/configs/experiments/action_supervision/instructpart_prompt_ensemble.json"
CONTROLLER_PYTHON=${CONTROLLER_PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
MOLMO2_PYTHON=${MOLMO2_PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
MOLMOACT_PYTHON=${MOLMOACT_PYTHON:-$ROOT/.envs/molmoact/bin/python}
DRY_RUN=0
LIMIT=
DEVICE=
OUTPUT_ROOT=
WORKERS=16
while (($#)); do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
for executable in "$CONTROLLER_PYTHON" "$MOLMO2_PYTHON" "$MOLMOACT_PYTHON"; do
  if [[ ! -x $executable ]]; then
    printf 'Required experiment Python is missing: %s\n' "$executable" >&2
    exit 2
  fi
done
if [[ -n $LIMIT && -z $OUTPUT_ROOT ]]; then
  OUTPUT_ROOT="$ROOT/outputs/small/action_supervision/instructpart_prompt_ensemble/limit_$LIMIT"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
cd "$ROOT"
run_command() {
  if ((DRY_RUN)); then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
prepare=(
  "$CONTROLLER_PYTHON" -m
  sspace.experiments.action_supervision.instructpart.prepare
  --config "$CONFIG" --workers "$WORKERS"
)
if [[ -n $DEVICE ]]; then prepare+=(--device "$DEVICE"); fi
run_command "${prepare[@]}"
for model in molmo2_er molmoact2_pretrain molmoact2; do
  if [[ $model == molmo2_er ]]; then PYTHON=$MOLMO2_PYTHON; else PYTHON=$MOLMOACT_PYTHON; fi
  command=(
    "$PYTHON" -m sspace.experiments.action_supervision.instructpart.projection
    --config "$CONFIG" --adapter "$model"
  )
  if [[ -n $LIMIT ]]; then command+=(--limit "$LIMIT"); fi
  if [[ -n $DEVICE ]]; then command+=(--device "$DEVICE"); fi
  if [[ -n $OUTPUT_ROOT ]]; then command+=(--output-root "$OUTPUT_ROOT"); fi
  run_command "${command[@]}"
done
score=(
  "$CONTROLLER_PYTHON" -m sspace.experiments.action_supervision.instructpart.scoring
  --config "$CONFIG"
)
if [[ -n $LIMIT ]]; then score+=(--limit "$LIMIT"); fi
if [[ -n $OUTPUT_ROOT ]]; then
  score+=(--projection-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/results.json")
fi
run_command "${score[@]}"
