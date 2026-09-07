#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export TMPDIR="$ROOT/.tmp"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv/python"
export HF_HOME="$ROOT/.cache/huggingface"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME"

install_environment() {
  local name=$1 version=$2 requirements=$3
  local python="$ROOT/.envs/$name/bin/python"
  if [[ ! -x $python ]]; then
    uv venv --python "$version" "$ROOT/.envs/$name"
  fi
  uv pip install --python "$python" -r "$requirements"
}

install_environment spatialmqa 3.10 requirements-molmo2.txt
install_environment molmoact 3.11 requirements-molmoact2.txt
install_environment qwen36 3.11 requirements-qwen.txt

export PYTHON="$ROOT/.envs/spatialmqa/bin/python"
export ASSET_ROOT="$ROOT/.cache/assets"
"$ROOT/scripts/prepare_models.sh" all
"$ROOT/scripts/prepare_models.sh" action_supervision
