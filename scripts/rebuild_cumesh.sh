#!/usr/bin/env bash
# Rebuild CuMesh and its CUDA extensions for the *actual* GPU compute capability.
# Run manually after requirements install; no automatic compilation during start.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi

for tool in git g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[CUMESH] Missing build tool: $tool (install git and build-essential)." >&2
    exit 2
  fi
done

torch_cuda="$(python -c 'import torch; assert torch.version.cuda, "CUDA-enabled PyTorch required"; print(torch.version.cuda)')"
arch="$(python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU"; major,minor=torch.cuda.get_device_capability(0); print(f"{major}.{minor}")')"
candidate="/usr/local/cuda-$torch_cuda"

# Prefer a matching installed toolkit when CUDA_HOME is not supplied, rather
# than silently selecting a different nvcc through /usr/local/cuda symlink.
if [[ -z "${CUDA_HOME:-}" && -x "$candidate/bin/nvcc" ]]; then
  export CUDA_HOME="$candidate"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "[CUMESH] CUDA_HOME=$CUDA_HOME does not contain nvcc." >&2
    exit 2
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[CUMESH] nvcc is missing. Install a CUDA development toolkit matching PyTorch CUDA $torch_cuda." >&2
  exit 2
fi
hash -r
export TORCH_CUDA_ARCH_LIST="$arch"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[CUMESH] Target architecture SM $arch; torch CUDA $torch_cuda."
echo "[CUMESH] Using nvcc: $(command -v nvcc)"
nvcc --version

python - <<'PY'
import re
import subprocess
import sys
import torch

nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", nvcc)
if match is None:
    sys.exit("[CUMESH] Could not parse nvcc version.")
toolkit = tuple(map(int, match.groups()))
torch_version = tuple(map(int, torch.version.cuda.split(".")[:2]))
print(f"[CUMESH] CUDA toolkit: {toolkit}; PyTorch CUDA: {torch_version}")
if toolkit[0] != torch_version[0]:
    sys.exit("[CUMESH] CUDA major mismatch. Select/install a toolkit matching torch CUDA.")
if torch.cuda.get_device_capability(0)[0] >= 12 and toolkit < (12, 8):
    sys.exit("[CUMESH] Blackwell SM 12.0 requires CUDA toolkit 12.8 or newer.")
PY

# Pin the source revision for repeatable builds, including its cubvh submodule.
source_dir="${CUMESH_SOURCE_DIR:-$HOME/.cache/trellis2-build/CuMesh}"
source_commit="12289e1062f0603f2f0d0771b02e1395d247f26f"
if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "[CUMESH] Refusing to overwrite non-Git source at $source_dir" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$source_dir")"
  git clone --recursive --no-checkout https://github.com/JeffreyXiang/CuMesh.git "$source_dir"
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" fetch --depth 1 origin "$source_commit"
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" -c advice.detachedHead=false checkout --detach --force "$source_commit"
git -C "$source_dir" submodule sync --recursive
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" submodule update --init --recursive
if [[ ! -f "$source_dir/setup.py" || ! -f "$source_dir/third_party/cubvh/src/bvh.cu" ]]; then
  echo "[CUMESH] CuMesh source or cubvh submodule is incomplete." >&2
  exit 2
fi

echo "[CUMESH] Compiling CuMesh from pinned source for SM $arch."
echo "[CUMESH] PyTorch, FlexGEMM, O-Voxel and model weights will NOT be reinstalled."
python -m pip install --disable-pip-version-check ninja
python -m pip install --disable-pip-version-check --force-reinstall \
  --no-deps --no-build-isolation --no-cache-dir "$source_dir"

echo "[CUMESH] Checking the exact mesh connectivity CUDA operation..."
python scripts/check_cumesh_kernels.py
echo "[CUMESH] Rebuild complete if the CUDA test above passed."
echo "[CUMESH] Do not reinstall requirements-runpod.txt; it can restore the incompatible CuMesh wheel."
