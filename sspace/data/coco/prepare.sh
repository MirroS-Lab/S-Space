#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$PROJECT_ROOT/.envs/spatialmqa/bin/python}
ASSET_ROOT=${ASSET_ROOT:-$PROJECT_ROOT/.cache/assets}

cd "$PROJECT_ROOT"
DRY_RUN=0
for argument in "$@"; do
  if [[ $argument == --dry-run ]]; then DRY_RUN=1; fi
done
if ((DRY_RUN)); then
  "$PYTHON" -m sspace.data.assets_cli --asset-root "$ASSET_ROOT" --dataset coco "$@"
  echo "$PYTHON -m sspace.data.coco.frozen_cli --asset-root $ASSET_ROOT"
  exit 0
fi

"$PYTHON" -m sspace.data.assets_cli --asset-root "$ASSET_ROOT" --dataset coco "$@"
"$PYTHON" -m sspace.data.coco.frozen_cli --asset-root "$ASSET_ROOT"
