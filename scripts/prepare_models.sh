#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
ASSET_ROOT=${ASSET_ROOT:-$ROOT/.cache/assets}
MODEL=${1:?"usage: prepare_models.sh MODEL|object_lens|action_supervision|all [--dry-run]"}
shift
if [[ ! -x $PYTHON ]]; then
  printf 'Model-preparation Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Create .envs/spatialmqa as described in README.md.\n' >&2
  exit 2
fi
case "$MODEL" in
  all)
    selection=(--group models)
    ;;
  object_lens)
    selection=(--group object_lens)
    ;;
  action_supervision)
    selection=(--group action_supervision)
    ;;
  molmo2_er|molmoact2|molmoact2_pretrain|qwen35_4b|qwen36_27b|grounding_dino|sam_vit_base|slimsam_50_uniform|depth_anything_v2_small_hf)
    selection=(--model "$MODEL")
    ;;
  *)
    printf 'Unknown model: %s\n' "$MODEL" >&2
    exit 2
    ;;
esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec "$PYTHON" -m sspace.data.assets_cli \
  --asset-root "$ASSET_ROOT" "${selection[@]}" "$@"
