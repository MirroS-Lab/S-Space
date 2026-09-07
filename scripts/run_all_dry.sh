#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
if [[ ! -x $PYTHON ]]; then
  printf 'Controller Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Create .envs/spatialmqa as described in README.md.\n' >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
"$PYTHON" -m sspace.registry
exec "$PYTHON" -m sspace.experiments.workflows --all --dry-run "$@"
