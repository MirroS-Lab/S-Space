#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
CONFIG=${1:?"usage: reproduce_small.sh CONFIG [LIMIT] [launcher arguments]"}
LIMIT=${2:-6}
shift $(( $# >= 2 ? 2 : 1 ))
if [[ ! $LIMIT =~ ^[1-9][0-9]*$ ]]; then
  printf 'LIMIT must be a positive integer: %s\n' "$LIMIT" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --config|--config=*|--limit|--limit=*|--output-root|--output-root=*)
      printf 'reproduce_small.sh controls %s; remove that launcher argument\n' \
        "${argument%%=*}" >&2
      exit 2
      ;;
  esac
done
if [[ $CONFIG != /* ]]; then CONFIG="$ROOT/$CONFIG"; fi
if [[ ! -x $PYTHON ]]; then
  printf 'Controller Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Create .envs/spatialmqa as described in README.md.\n' >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec "$PYTHON" -m sspace.experiments.scripts.run_suite \
  --config "$CONFIG" \
  --output-root "$ROOT/outputs/small/limit_$LIMIT" \
  --limit "$LIMIT" \
  "$@"
