#!/usr/bin/env bash
# Repair incompatible FlexGEMM wheel by compiling native CUDA kernels for the
# GPU actually assigned to this RunPod instance. Run explicitly after installing
# requirements-runpod.txt; never compile automatically during app startup.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[FLEXGEMM] Missing nvcc. Use a RunPod CUDA *devel* image/toolkit matching PyTorch's CUDA, not runtime-only." >&2
  exit 2
fi
for binary in git g++ ; do
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "[FLEXGEMM] Missing build tool: $binary. Install build essentials first." >&2
    exit 2
  fi
done

# Detect the actual physical or MIG GPU compute capability. A memory quota
# (24GB/36GB) does NOT establish which architecture kernels must target.
gpu_arch="$(python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
export TORCH_CUDA_ARCH_LIST="$gpu_arch"
export MAX_JOBS="${MAX_JOBS:-4}"
echo "[FLEXGEMM] Building specifically for SM $TORCH_CUDA_ARCH_LIST with MAX_JOBS=$MAX_JOBS."
nvcc --version
python - <<'PY'
import re
import subprocess
import torch
import sys
major, minor = torch.cuda.get_device_capability(0)
toolkit = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", toolkit)
if match is None:
    sys.exit("Unable to parse nvcc version.")
toolkit_ver = (int(match.group(1)), int(match.group(2)))
torch_ver = tuple(map(int, torch.version.cuda.split(".")[:2])) if torch.version.cuda else None
print(f"[FLEXGEMM] NVCC toolkit: {toolkit_ver}; PyTorch CUDA: {torch_ver}")
if major >= 12 and toolkit_ver < (12, 8):
    sys.exit("Blackwell sm_120 requires CUDA Toolkit 12.8 or newer. Select a compatible devel image.")
if torch_ver and toolkit_ver[0] != torch_ver[0]:
    sys.exit("NVCC toolkit major version differs from PyTorch CUDA; use a matching devel image.")
PY

python -m pip install -U ninja
source_dir="${FLEXGEMM_SOURCE_DIR:-$HOME/.cache/trellis2-build/FlexGEMM-v1.1.0}"
if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "[FLEXGEMM] The build source path exists but is not a Git checkout: $source_dir" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$source_dir")"
  git clone --recursive --depth 1 --branch v1.1.0 https://github.com/JeffreyXiang/FlexGEMM.git "$source_dir"
fi
git -C "$source_dir" submodule update --init --recursive

echo "[FLEXGEMM] Installing from upstream v1.1.0 against the EXISTING PyTorch."
echo "[FLEXGEMM] This does not reinstall Torch, Gradio or model weights."
python -m pip install --force-reinstall --no-deps --no-build-isolation "$source_dir"

echo "[FLEXGEMM] Validating the exact sparse-convolution CUDA operation..."
python scripts/check_cuda_kernels.py
echo "[FLEXGEMM] Repair complete if the smoke test above passed."
echo "[FLEXGEMM] IMPORTANT: Re-running requirements-runpod.txt may restore its prebuilt wheel; rebuild afterward if needed."
