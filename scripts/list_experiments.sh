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
cd "$ROOT"
"$PYTHON" -m sspace.registry
"$PYTHON" -m sspace.experiments.scripts.run_suite --list
"$PYTHON" -m sspace.experiment_steps list
printf '\nLaunch with scripts/run_experiment.sh: CONFIG, --suite NAME, --experiment NAME, --all, or --step NAME.\n'
