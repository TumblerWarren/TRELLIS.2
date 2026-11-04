#!/usr/bin/env python3
"""Run native O-Voxel CUDA hashmap kernels needed for TRELLIS.2 mesh decoding.

Tests tiny synthetic data, not a full 3D generation. Run in a fresh process:
a CUDA 'no kernel image' error poisons the CUDA context in that process.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify O-Voxel mesh CUDA kernels on the assigned GPU.")
    parser.add_argument("--device", type=int, default=0, help="CUDA GPU index")
    args = parser.parse_args()

    try:
        import torch
        from o_voxel import _C
    except (ImportError, OSError) as exc:
        print(f"[O-VOXEL] Cannot load O-Voxel CUDA extension: {exc}", file=sys.stderr)
        print("[O-VOXEL] Ensure requirements are installed; see scripts/rebuild_o_voxel.sh", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[O-VOXEL] CUDA GPU unavailable.", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{args.device}")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"[O-VOXEL] GPU: {torch.cuda.get_device_name(device)} / SM {major}.{minor}", flush=True)
    print(f"[O-VOXEL] PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)

    # O-Voxel flexible_dual_grid_to_mesh uses precisely these native hashmap
    # insert and lookup entrypoints. Use a power-of-two capacity and unique
    # in-range (batch, x, y, z) coordinates, exactly as in upstream.
    points = [
        [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3],
        [0, 4, 4, 4], [0, 5, 5, 5], [0, 6, 6, 6], [0, 7, 7, 7],
    ]
    try:
        coords = torch.tensor(points, dtype=torch.int32, device=device)
        capacity = 64
        keys = torch.full((capacity,), torch.iinfo(torch.uint32).max,
                          dtype=torch.uint32, device=device)
        values = torch.empty((capacity,), dtype=torch.uint32, device=device)
        print("[O-VOXEL] Testing native hashmap_insert_3d_idx_as_val_cuda ...", flush=True)
        _C.hashmap_insert_3d_idx_as_val_cuda(keys, values, coords, 8, 8, 8)
        torch.cuda.synchronize(device)
        print("[O-VOXEL] Testing native hashmap_lookup_3d_cuda ...", flush=True)
        found = _C.hashmap_lookup_3d_cuda(keys, values, coords, 8, 8, 8)
        torch.cuda.synchronize(device)
        if found.numel() != len(points) or not torch.equal(
            found.long(), torch.arange(len(points), device=device, dtype=torch.long)
        ):
            raise RuntimeError(f"O-Voxel hashmap lookup returned unexpected indices: {found.tolist()}")
    except Exception as exc:
        print(f"[O-VOXEL] CUDA kernel smoke test FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "[O-VOXEL] For missing GPU kernels, rebuild O-Voxel from source: "
            "bash scripts/rebuild_o_voxel.sh",
            file=sys.stderr,
        )
        print(
            "[O-VOXEL] The full-generation error may be asynchronous. "
            "Re-run the app with CUDA_LAUNCH_BLOCKING=1 if errors continue.",
            file=sys.stderr,
        )
        return 1

    print("[O-VOXEL] PASS: native CUDA hashmap insert and lookup executed correctly.", flush=True)
    print("[O-VOXEL] This is not a full mesh decode, CUDA-extension suite or GLB export test.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
