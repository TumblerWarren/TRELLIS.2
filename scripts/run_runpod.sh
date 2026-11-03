#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi
export OPENCV_IO_ENABLE_OPENEXR=1
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
python scripts/check_runpod.py
exec python app.py
