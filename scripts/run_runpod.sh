#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi
# Always remove deprecated legacy transfer setting from the shell environment.
# Older huggingface_hub versions can fail if it is set but hf_transfer is absent;
# newer versions ignore it. hf-xet is the supported accelerated downloader.
if [[ -n "${HF_HUB_ENABLE_HF_TRANSFER+x}" ]]; then
  echo "[TRELLIS.2] Removing deprecated HF_HUB_ENABLE_HF_TRANSFER from RunPod environment." >&2
  unset HF_HUB_ENABLE_HF_TRANSFER
fi
export OPENCV_IO_ENABLE_OPENEXR=1
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
python scripts/check_runpod.py
exec python app.py
