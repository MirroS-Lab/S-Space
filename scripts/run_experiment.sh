#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
case "${1:-}" in
  -h|--help|"")
    cat <<'HELP'
Usage:
  run_experiment.sh CONFIG [launcher arguments]
  run_experiment.sh --suite NAME [suite arguments]
  run_experiment.sh --experiment NAME [--dry-run] [--case-config PATH]
  run_experiment.sh --all [--dry-run] [--case-config PATH]
  run_experiment.sh --step NAME [stage arguments]
  run_experiment.sh --workflow action_supervision [workflow arguments]
  run_experiment.sh --list

Choose one route:
  --experiment NAME  Run all configured steps of an experiment in order.
  --suite NAME       Run every configuration in an evaluation group in order.
  CONFIG             Run one configuration with its model and settings.
  --step NAME        Run one step, such as plotting saved results.

Use --list to find names and --dry-run to preview commands without running them.
Run the README setup first; launchers select the model environment automatically.
HELP
    exit 0
    ;;
  --list)
    exec "$ROOT/scripts/list_experiments.sh"
    ;;
  --step|--extra)
    shift
    DRY_RUN=0
    EXTRA_ARGS=()
    for argument in "$@"; do
      if [[ $argument == --dry-run ]]; then
        DRY_RUN=1
      else
        EXTRA_ARGS+=("$argument")
      fi
    done
    if ((DRY_RUN)); then
      exec "$ROOT/scripts/run_step.sh" --dry-run "${EXTRA_ARGS[@]}"
    fi
    exec "$ROOT/scripts/run_step.sh" "${EXTRA_ARGS[@]}"
    ;;
  --workflow)
    shift
    if [[ ${1:-} != action_supervision ]]; then
      printf 'Supported workflow: action_supervision\n' >&2
      exit 2
    fi
    shift
    exec "$ROOT/scripts/run_action_supervision.sh" "$@"
    ;;
esac
export TMPDIR="$ROOT/.tmp"
mkdir -p "$TMPDIR"
if [[ ${1:-} == --all || ${1:-} == --experiment ]]; then
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT"
  exec "$PYTHON" -m sspace.experiments.workflows "$@"
fi
if [[ ${1:-} == --suite ]]; then
  shift
  if [[ $# == 0 ]]; then
    printf 'Usage: run_experiment.sh --suite NAME [suite arguments]\n' >&2
    exit 2
  fi
  if [[ ! -x $PYTHON ]]; then
    printf 'Controller Python is missing or not executable: %s\n' "$PYTHON" >&2
    exit 2
  fi
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT"
  exec "$PYTHON" -m sspace.experiments.scripts.run_suite "$@"
fi
CONFIG=${1:?"usage: run_experiment.sh CONFIG [launcher arguments]"}
shift
if [[ $CONFIG != /* ]]; then CONFIG="$ROOT/$CONFIG"; fi
if [[ ! -x $PYTHON ]]; then
  printf 'Controller Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Create .envs/spatialmqa as described in README.md.\n' >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" -m sspace.experiments.launch --config "$CONFIG" "$@"
