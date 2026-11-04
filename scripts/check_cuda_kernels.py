#!/usr/bin/env python3
"""Execute the same FlexGEMM sparse-convolution path that TRELLIS.2 requires.

This checks *runtime kernel execution*, not just torch.cuda.is_available() or
package imports. It uses only tiny synthetic tensors (no model downloads).
Run this before starting Gradio or a costly 3D generation.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Test TRELLIS.2 FlexGEMM CUDA kernels on the assigned GPU.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal (default: 0)")
    args = parser.parse_args()

    try:
        import torch
        import flex_gemm
        from flex_gemm.ops.spconv import sparse_submanifold_conv3d
    except (ImportError, OSError) as exc:
        print(f"[FLEXGEMM] Cannot load dependencies: {exc}", file=sys.stderr)
        print("[FLEXGEMM] Install requirements-runpod.txt first.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("[FLEXGEMM] No CUDA GPU available.", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{args.device}")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"[FLEXGEMM] GPU: {torch.cuda.get_device_name(device)}; SM {major}.{minor}", flush=True)
    print(f"[FLEXGEMM] PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}", flush=True)
    print(f"[FLEXGEMM] PyTorch built-in architectures: {torch.cuda.get_arch_list()}", flush=True)
    print("[FLEXGEMM] Running small sparse_submanifold_conv3d forward pass...", flush=True)

    # Small, unique (batch, x, y, z) coordinates, to exercise the actual
    # CUDA hashmap/neighbor map path that fails with cudaErrorNoKernelImageForDevice.
    coords = torch.tensor(
        [[0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 1], [0, 2, 1, 1],
         [0, 2, 2, 2], [0, 3, 3, 3], [0, 4, 4, 4], [0, 5, 5, 5]],
        dtype=torch.int32, device=device,
    )
    ci, co = 16, 16
    feats = torch.randn(len(coords), ci, device=device, dtype=torch.float16)
    weights = torch.randn(co, 3, 3, 3, ci, device=device, dtype=torch.float16) * 0.01
    bias = torch.zeros(co, device=device, dtype=torch.float16)
    shape = torch.Size([1, ci, 8, 8, 8])

    try:
        flex_gemm.ops.spconv.set_algorithm("masked_implicit_gemm_splitk")
        output, _ = sparse_submanifold_conv3d(feats, coords, shape, weights, bias)
        torch.cuda.synchronize(device)
        if output.shape != (len(coords), co) or not torch.isfinite(output).all().item():
            raise RuntimeError(f"Unexpected FlexGEMM output shape or nonfinite values: {output.shape}")
    except Exception as exc:
        message = str(exc)
        print(f"[FLEXGEMM] CUDA kernel smoke test FAILED: {type(exc).__name__}: {message}", file=sys.stderr)
        if any(x in message.lower() for x in ("no kernel image", "invalid device function", "unsupported gpu architecture")):
            print(
                "[FLEXGEMM] The installed CUDA extension is likely not compiled for this GPU. "
                "Run: bash scripts/rebuild_flexgemm.sh",
                file=sys.stderr,
            )
        else:
            print(
                "[FLEXGEMM] Check the FlexGEMM version, CUDA toolkit, torch/CUDA ABI, "
                "and available GPU architecture. See README_RUNPOD.md.",
                file=sys.stderr,
            )
        return 1

    print("[FLEXGEMM] PASS: sparse convolution CUDA kernel executed successfully.", flush=True)
    print("[FLEXGEMM] This is not a full TRELLIS.2 generation/GLB test.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
