#!/usr/bin/env bash
# Build native CUDA extensions without needing a GPU during docker build.
# A real GPU is needed to validate the installed kernels at runtime.
set -euo pipefail
: "${TORCH_CUDA_ARCH_LIST:?Set TORCH_CUDA_ARCH_LIST (e.g. 8.9 for RTX 4090)}"
export MAX_JOBS="${MAX_JOBS:-4}"
export GIT_LFS_SKIP_SMUDGE=1
build_root="$(mktemp -d /tmp/trellis-ext-XXXXXX)"
trap 'rm -rf "$build_root"' EXIT

echo "[TRELLIS-DOCKER] Building native extensions for $TORCH_CUDA_ARCH_LIST"
python - <<'PY'
import re
import subprocess
import sys
import torch
if not torch.version.cuda:
    sys.exit("[TRELLIS-DOCKER] CUDA-enabled PyTorch required.")
result = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", result)
if not match:
    sys.exit("[TRELLIS-DOCKER] nvcc version not detected.")
toolkit = (int(match.group(1)), int(match.group(2)))
pytorch = tuple(map(int, torch.version.cuda.split(".")[:2]))
print(f"[TRELLIS-DOCKER] nvcc CUDA {toolkit}; PyTorch CUDA {pytorch}")
if toolkit[0] != pytorch[0]:
    sys.exit("[TRELLIS-DOCKER] CUDA toolkit major version must match PyTorch.")
PY

# For nvdiffrec's -lcuda link dependency. Do not add stubs to runtime LD_LIBRARY_PATH.
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"

echo "[TRELLIS-DOCKER] Building FlexGEMM v1.1.0 from source."
git clone --recursive --depth 1 --branch v1.1.0 \
  https://github.com/JeffreyXiang/FlexGEMM.git "$build_root/FlexGEMM"
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  "$build_root/FlexGEMM"

echo "[TRELLIS-DOCKER] Building CuMesh for selected GPU architecture."
git clone --recursive --no-checkout https://github.com/JeffreyXiang/CuMesh.git "$build_root/CuMesh"
git -C "$build_root/CuMesh" fetch --depth 1 origin 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C "$build_root/CuMesh" checkout --detach 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C "$build_root/CuMesh" submodule update --init --recursive
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  "$build_root/CuMesh"

echo "[TRELLIS-DOCKER] Building Microsoft's O-Voxel from official source."
git init -q "$build_root/trellis-upstream"
git -C "$build_root/trellis-upstream" remote add origin https://github.com/microsoft/TRELLIS.2.git
git -C "$build_root/trellis-upstream" sparse-checkout init --cone
git -C "$build_root/trellis-upstream" sparse-checkout set o-voxel
git -C "$build_root/trellis-upstream" fetch --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$build_root/trellis-upstream" checkout --detach FETCH_HEAD
git -C "$build_root/trellis-upstream" submodule update --init --recursive -- o-voxel/third_party/eigen
test -d "$build_root/trellis-upstream/o-voxel/third_party/eigen/Eigen"
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  "$build_root/trellis-upstream/o-voxel"

echo "[TRELLIS-DOCKER] Building native rendering extensions."
git clone --depth 1 --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$build_root/nvdiffrast"
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  "$build_root/nvdiffrast"

git clone --branch renderutils --no-checkout https://github.com/JeffreyXiang/nvdiffrec.git \
  "$build_root/nvdiffrec"
git -C "$build_root/nvdiffrec" fetch --depth 1 origin b296927cc7fd01c2ac1087c8065c4d7248f72da4
git -C "$build_root/nvdiffrec" checkout --detach b296927cc7fd01c2ac1087c8065c4d7248f72da4
python -m pip install --no-deps --no-build-isolation --force-reinstall \
  "$build_root/nvdiffrec"

# Imports only. GPU kernels are tested on RunPod, not during Docker build.
python - <<'PY'
import importlib
for name in ("flex_gemm", "cumesh", "o_voxel", "nvdiffrast", "nvdiffrec_render"):
    importlib.import_module(name)
    print(f"[TRELLIS-DOCKER] Import OK: {name}")
print("[TRELLIS-DOCKER] No model weights were downloaded.")
PY
