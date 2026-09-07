#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NAME=${1:-}
if [[ $NAME == --dry-run ]]; then NAME=${2:-}; fi
case "$NAME" in
  evolving-cot-*|mmsi-extract)
    DEFAULT_PYTHON="$ROOT/.envs/qwen36/bin/python"
    ;;
  *)
    DEFAULT_PYTHON="$ROOT/.envs/spatialmqa/bin/python"
    ;;
esac
PYTHON=${PYTHON:-$DEFAULT_PYTHON}
if [[ ! -x $PYTHON ]]; then
  printf 'Experiment stage Python is missing or not executable: %s\n' "$PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec "$PYTHON" -m sspace.experiment_steps "$@"
