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
# OAuth browser login lives in an isolated modern CLI; refresh the credential
# before the older TRELLIS.2 runtime Hub library reads it.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
auth_cli_venv="${HF_BROWSER_CLI_VENV:-$HOME/.local/share/trellis2/hf-auth-cli}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[TRELLIS.2] Using HF_TOKEN secret (it overrides saved browser credentials)." >&2
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "[TRELLIS.2] Offline mode: skipping browser credential refresh." >&2
elif [[ -x "$auth_cli_venv/bin/hf" ]]; then
  bash scripts/login_huggingface.sh --check
else
  echo "[TRELLIS.2] Optional browser sign-in: bash scripts/login_huggingface.sh" >&2
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
