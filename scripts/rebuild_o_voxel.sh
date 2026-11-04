#!/usr/bin/env bash
# Compile O-Voxel's CUDA extension for the actual GPU, without reinstalling torch,
# FlexGEMM, or other TRELLIS.2 dependencies. Intended for RunPod *devel* images.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi
for binary in git g++; do
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "[O-VOXEL] Missing build tool '$binary'. Install Git and C++ build tools." >&2
    exit 2
  fi
done

# Use the same CUDA toolkit major version as the already installed PyTorch.
torch_cuda="$(python -c 'import torch; assert torch.version.cuda, "PyTorch CUDA is missing"; print(torch.version.cuda)')"
gpu_arch="$(python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
preferred="/usr/local/cuda-$torch_cuda"
if [[ -z "${CUDA_HOME:-}" && -x "$preferred/bin/nvcc" ]]; then
  export CUDA_HOME="$preferred"
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[O-VOXEL] nvcc not found. Install the CUDA development toolkit that matches PyTorch CUDA $torch_cuda." >&2
  exit 2
fi
hash -r
export TORCH_CUDA_ARCH_LIST="$gpu_arch"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[O-VOXEL] GPU SM $gpu_arch; PyTorch CUDA $torch_cuda; MAX_JOBS=$MAX_JOBS"
echo "[O-VOXEL] Using nvcc: $(command -v nvcc)"
nvcc --version
python - <<'PY'
import re, subprocess, sys, torch
out = subprocess.check_output(["nvcc", "--version"], text=True)
m = re.search(r"release\s+(\d+)\.(\d+)", out)
if not m:
    sys.exit("[O-VOXEL] Could not read CUDA compiler version.")
toolkit = (int(m.group(1)), int(m.group(2)))
runtime = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
print(f"[O-VOXEL] Compiler CUDA: {toolkit}; PyTorch CUDA: {runtime}")
if toolkit[0] != runtime[0]:
    sys.exit("[O-VOXEL] CUDA major versions mismatch. Select/install matching CUDA devel toolkit.")
if torch.cuda.get_device_capability(0)[0] >= 12 and toolkit < (12, 8):
    sys.exit("[O-VOXEL] Blackwell requires CUDA Toolkit 12.8 or newer.")
PY

# Pin Microsoft's source to a known revision, rather than rely on a
# branch tip that could move while a user is troubleshooting a running Pod.
upstream_commit="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
source_repo="${OVOXEL_SOURCE_REPO:-$HOME/.cache/trellis2-build/TRELLIS2-ovoxel-source}"
if [[ ! -d "$source_repo/.git" ]]; then
  if [[ -e "$source_repo" ]]; then
    echo "[O-VOXEL] Source directory exists but isn't Git: $source_repo" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$source_repo")"
  git init --quiet "$source_repo"
  git -C "$source_repo" remote add origin https://github.com/microsoft/TRELLIS.2.git
fi
echo "[O-VOXEL] Fetching upstream TRELLIS.2 o-voxel source (pinned $upstream_commit)."
git -C "$source_repo" sparse-checkout init --cone
git -C "$source_repo" sparse-checkout set o-voxel
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_repo" fetch --depth 1 origin "$upstream_commit"
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_repo" -c advice.detachedHead=false checkout --detach --force FETCH_HEAD
# O-Voxel uses Eigen as a Git submodule.
GIT_LFS_SKIP_SMUDGE=1 git -C "$source_repo" submodule update --init --recursive -- o-voxel/third_party/eigen
source_dir="$source_repo/o-voxel"
if [[ ! -f "$source_dir/setup.py" || ! -d "$source_dir/third_party/eigen/Eigen" ]]; then
  echo "[O-VOXEL] Incomplete source checkout or missing Eigen submodule: $source_dir" >&2
  exit 2
fi

echo "[O-VOXEL] Compiling against existing PyTorch (no dependencies reinstalled)."
python -m pip install --disable-pip-version-check ninja
python -m pip install --disable-pip-version-check --force-reinstall \
  --no-deps --no-build-isolation --no-cache-dir "$source_dir"

echo "[O-VOXEL] Running real CUDA extension smoke test..."
python scripts/check_o_voxel_kernels.py
echo "[O-VOXEL] Repair completed if test above passed."
echo "[O-VOXEL] WARNING: Reinstalling requirements-runpod.txt can overwrite this build with an incompatible prebuilt wheel."
