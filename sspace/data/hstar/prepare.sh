#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$PROJECT_ROOT/.envs/spatialmqa/bin/python}
ASSET_ROOT=${ASSET_ROOT:-$PROJECT_ROOT/.cache/assets}
SOURCE_DIR=$ASSET_ROOT/datasets/hstar/hos_train

cd "$PROJECT_ROOT"
DRY_RUN=0
for argument in "$@"; do
  if [[ $argument == --dry-run ]]; then DRY_RUN=1; fi
done
if ((DRY_RUN)); then
  "$PYTHON" -m sspace.data.assets_cli --asset-root "$ASSET_ROOT" --dataset hstar "$@"
  echo "$PYTHON -m sspace.data.hstar.cli --source-dir $SOURCE_DIR --output-dir data/processed/hstar_hos600"
else
  "$PYTHON" -m sspace.data.assets_cli --asset-root "$ASSET_ROOT" --dataset hstar "$@"
  "$PYTHON" -m sspace.data.hstar.cli \
    --source-dir "$SOURCE_DIR" --output-dir data/processed/hstar_hos600
fi
echo "H* reuses the published visual audit when rebuilt rows and image hashes match the released fingerprint. Changed datasets require a new visual review."
