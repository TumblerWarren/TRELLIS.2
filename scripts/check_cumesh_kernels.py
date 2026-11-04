#!/usr/bin/env python3
"""Execute CuMesh's native CUDA connectivity operation on a tiny synthetic mesh.

The exact observed RunPod failure was get_edges() in src/connectivity.cu.
Use a fresh Python process: failed CUDA launches can poison the current context.
This does not download models or modify installed packages.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CuMesh native CUDA kernels on the assigned GPU.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    try:
        import torch
        from cumesh import CuMesh
    except (ImportError, OSError) as exc:
        print(f"[CUMESH] Import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[CUMESH] Install RunPod requirements or run bash scripts/rebuild_cumesh.sh.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[CUMESH] No CUDA device is available.", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{args.device}")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"[CUMESH] GPU: {torch.cuda.get_device_name(device)}; SM {major}.{minor}", flush=True)
    print(f"[CUMESH] PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)

    # Upstream CuMesh::init copies float32 vertices and int32 faces using
    # data_ptr<float>() and data_ptr<int>() into its CUDA buffers.
    # Two triangles in a square: four vertices, five unique undirected edges.
    vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=torch.float32, device=device,
    ).contiguous()
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device,
    ).contiguous()

    try:
        mesh = CuMesh()
        mesh.init(vertices, faces)
        print("[CUMESH] Executing the same native get_edges() path as TRELLIS.2 fill_holes() ...", flush=True)
        mesh.get_edges()
        torch.cuda.synchronize(device)
        edges = mesh.read_edges()
        torch.cuda.synchronize(device)
        if mesh.num_edges != 5 or edges.ndim != 2 or edges.shape[0] != 5:
            raise RuntimeError(
                f"Unexpected edges: count={mesh.num_edges}, tensor shape={tuple(edges.shape)}."
            )
    except Exception as exc:
        print(f"[CUMESH] Native CUDA test FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "[CUMESH] The installed CuMesh binary may lack kernels for this GPU. "
            "Try: bash scripts/rebuild_cumesh.sh",
            file=sys.stderr,
        )
        return 1

    print("[CUMESH] PASS: native CUDA mesh connectivity/get_edges kernel ran successfully.", flush=True)
    print("[CUMESH] This does not validate all CuMesh operations, rendering, or GLB export.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
