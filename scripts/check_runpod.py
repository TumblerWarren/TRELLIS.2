#!/usr/bin/env python3
"""Preflight Hugging Face model access and local GPU before running TRELLIS.2."""
import os
import sys
from hf_download_compat import configure_hf_download_environment, explain_hub_error

# Must run before importing Hugging Face Hub, Transformers or PyTorch.
configure_hf_download_environment()

def main():
    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install requirements-runpod.txt first.") from exc

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: use an NVIDIA GPU pod and compatible PyTorch.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} / {vram:.1f} GiB / compute {capability[0]}.{capability[1]}")
    if vram < 23:
        print("WARNING: The official TRELLIS.2 model expects a 24 GB class GPU.")

    rembg = os.getenv("TRELLIS_REMBG_MODEL", "ZhengPeng7/BiRefNet")
    checks = [
        ("microsoft/TRELLIS.2-4B", "pipeline.json",
         "Main model is public; check network access."),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "config.json",
         "Accept Meta's DINOv3 terms, then run bash scripts/login_huggingface.sh (browser sign-in) or use an approved HF_TOKEN secret."),
        (rembg, "config.json",
         "Check background-model access and license; the default ZhengPeng7/BiRefNet is public."),
    ]
    failed = False
    for repo, filename, hint in checks:
        try:
            hf_hub_download(repo_id=repo, filename=filename)
            print(f"OK: {repo}/{filename}")
        except Exception as exc:
            print(f"FAILED: {repo}/{filename}: {type(exc).__name__}: {exc}")
            print(f"  Troubleshooting: {explain_hub_error(exc)}")
            print(f"  Model-specific hint: {hint}")
            failed = True
    if failed:
        raise SystemExit("Model preflight failed. Correct access before launching the app.")
    # A successful torch CUDA check does not prove third-party CUDA extensions
    # include kernels for the assigned GPU (e.g. Blackwell SM 12.0).
    # Test the exact FlexGEMM operation before expensive model initialization.
    if os.getenv("SPARSE_CONV_BACKEND", "flex_gemm") == "flex_gemm":
        if os.getenv("TRELLIS_SKIP_KERNEL_PREFLIGHT", "").lower() in {"1", "true", "yes"}:
            print("WARNING: Skipping FlexGEMM runtime kernel test by explicit override.")
        else:
            import subprocess
            from pathlib import Path
            kernel_check = Path(__file__).with_name("check_cuda_kernels.py")
            print("Checking FlexGEMM CUDA kernel compatibility...", flush=True)
            try:
                subprocess.run([sys.executable, "-u", str(kernel_check)], check=True, timeout=240)
            except subprocess.TimeoutExpired as exc:
                raise SystemExit(
                    "FlexGEMM CUDA smoke test timed out. Inspect the CUDA toolchain or run "
                    "python scripts/check_cuda_kernels.py manually."
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    "FlexGEMM failed on this GPU. Run bash scripts/rebuild_flexgemm.sh "
                    "using a compatible CUDA developer toolkit, then retry. "
                    "See README_RUNPOD.md."
                ) from exc
    else:
        print("WARNING: Nondefault SPARSE_CONV_BACKEND selected; FlexGEMM smoke test not run.")
    # O-Voxel contains separate CUDA extension kernels needed for mesh decoding.
    # A working FlexGEMM test does NOT imply O-Voxel kernels support this GPU.
    if os.getenv("TRELLIS_SKIP_KERNEL_PREFLIGHT", "").lower() not in {"1", "true", "yes"}:
        import subprocess
        from pathlib import Path
        o_voxel_check = Path(__file__).with_name("check_o_voxel_kernels.py")
        print("Checking O-Voxel mesh-decoding CUDA kernels...", flush=True)
        try:
            subprocess.run([sys.executable, "-u", str(o_voxel_check)], check=True, timeout=240)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                "O-Voxel kernel check timed out; run python scripts/check_o_voxel_kernels.py."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "O-Voxel CUDA kernels failed on the assigned GPU. "
                "Run bash scripts/rebuild_o_voxel.sh with a matching CUDA developer toolkit, "
                "then retry. See README_RUNPOD.md."
            ) from exc
    else:
        print("WARNING: Skipping O-Voxel kernel test by explicit TRELLIS_SKIP_KERNEL_PREFLIGHT override.")
    print("Preflight complete. This does not verify full TRELLIS.2 generation or GLB export.")

if __name__ == "__main__":
    main()
