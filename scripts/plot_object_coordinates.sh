#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT/.envs/spatialmqa/bin/python}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  printf 'Usage: plot_object_coordinates.sh [OUTPUT_DIR]\n'
  printf 'Plot object coordinates from the bundled Molmo2-ER L21 COCO projections.\n'
  printf 'Set PYTHON to an environment with NumPy, pandas, and Matplotlib.\n'
  exit 0
fi
if (($# > 1)); then
  printf 'Usage: plot_object_coordinates.sh [OUTPUT_DIR]\n' >&2
  exit 2
fi
if [[ ! -x $PYTHON ]]; then
  printf 'Python is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'See README.md for the CPU-only environment.\n' >&2
  exit 2
fi
OUTPUT_DIR=${1:-$ROOT/outputs/reproduction/object_coordinates}
REFERENCE="$ROOT/sspace/data/object_coordinates/reference/coco_molmo2_er_l21"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" -m sspace.experiments.analysis.object_coordinates.plot \
  --input "$REFERENCE/single_object_points.csv" \
  --input-format standard_v1 \
  --selection 'Molmo2-ER=21' \
  --reference-manifest "$REFERENCE/manifest.json" \
  --output-dir "$OUTPUT_DIR"
