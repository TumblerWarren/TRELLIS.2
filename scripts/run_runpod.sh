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
# Request a temporary gradio.live link for first-time self-hosted RunPod use.
# The Python app protects it with randomly generated auth unless credentials
# were supplied or authentication was explicitly disabled.
export GRADIO_SHARE="${GRADIO_SHARE:-true}"
echo "[TRELLIS.2] GRADIO_SHARE=${GRADIO_SHARE}; temporary public URLs require working outbound connectivity." >&2
echo "[TRELLIS.2] RunPod alternative: expose HTTP port ${GRADIO_SERVER_PORT} in your Pod template/Connect settings." >&2
python scripts/check_runpod.py
exec python app.py
