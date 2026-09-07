#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$PROJECT_ROOT/.envs/spatialmqa/bin/python}
ASSET_ROOT=${ASSET_ROOT:-$PROJECT_ROOT/.cache/assets}
exec "$PYTHON" -m sspace.data.assets_cli \
  --asset-root "$ASSET_ROOT" --dataset spinbench "$@"
