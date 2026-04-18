#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR/architecture:${PYTHONPATH:-}"
export ENCODER_DINOV2_REPO="${ENCODER_DINOV2_REPO:-$ROOT_DIR/architecture/third_party/facebookresearch_dinov2_main}"

python "$ROOT_DIR/infer.py" "$@"
