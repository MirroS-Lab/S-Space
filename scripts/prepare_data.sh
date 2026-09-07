#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
if [[ ! -x $PYTHON ]]; then
  printf 'Data-preparation Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Create .envs/spatialmqa as described in README.md.\n' >&2
  exit 2
fi
export PYTHON
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
for dataset in coco spatialtunnel embspatial cvbench spinbench hstar; do
  echo "[$dataset]"
  "$ROOT/sspace/data/$dataset/prepare.sh" "$@"
done
