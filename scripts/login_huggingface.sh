#!/usr/bin/env bash
# Modern browser/device-code login, isolated from TRELLIS.2's pinned Python packages.
set -euo pipefail

mode="${1:-login}"
case "$mode" in
  login|--check) ;;
  -h|--help)
    echo "Usage: bash scripts/login_huggingface.sh [login|--check]"
    exit 0
    ;;
  *) echo "[TRELLIS.2] Invalid mode: $mode" >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
cli_venv="${HF_BROWSER_CLI_VENV:-$HOME/.local/share/trellis2/hf-auth-cli}"
cli="$cli_venv/bin/hf"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[TRELLIS.2] HF_TOKEN overrides browser login. Run 'unset HF_TOKEN' in your terminal first." >&2
  echo "[TRELLIS.2] For unattended Pods, keep the HF_TOKEN secret instead of browser login." >&2
  exit 2
fi

if [[ "$mode" == "--check" ]]; then
  if [[ ! -x "$cli" ]]; then
    echo "[TRELLIS.2] Browser CLI not installed. Run 'bash scripts/login_huggingface.sh' first." >&2
    exit 3
  fi
  if ! "$cli" auth whoami; then
    echo "[TRELLIS.2] Browser session unavailable. Run 'bash scripts/login_huggingface.sh' again." >&2
    exit 1
  fi
  exit 0
fi

if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "[TRELLIS.2] Browser authentication requires internet; disable HF_HUB_OFFLINE first." >&2
  exit 2
fi

umask 077
mkdir -p "$HF_HOME"
if [[ ! -x "$cli" ]]; then
  python_bin="${HF_BROWSER_CLI_PYTHON:-python3.12}"
  command -v "$python_bin" >/dev/null 2>&1 || {
    echo "[TRELLIS.2] Python 3.12 not found; set HF_BROWSER_CLI_PYTHON." >&2
    exit 2
  }
  "$python_bin" -m venv "$cli_venv"
fi

echo "[TRELLIS.2] Updating a standalone Hugging Face CLI without altering TRELLIS.2's inference environment."
"$cli_venv/bin/python" -m pip install --disable-pip-version-check -U 'huggingface_hub>=1.26,<2'
echo "[TRELLIS.2] Choose 'Log in with your browser'; open the provided URL and approve its verification code."
"$cli" auth login --force
"$cli" auth whoami
echo "[TRELLIS.2] Browser credentials are stored in HF_HOME=$HF_HOME"
echo "[TRELLIS.2] Next: python scripts/check_runpod.py"
