#!/usr/bin/env python3
"""Preflight Hugging Face model access and local GPU before running TRELLIS.2."""
import os
import sys

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
         "Accept Meta's gated model terms and authenticate with hf auth login or HF_TOKEN."),
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
            print(f"  Hint: {hint}")
            failed = True
    if failed:
        raise SystemExit("Model preflight failed. Correct access before launching the app.")
    print("Preflight complete. This does not test GPU generation or custom CUDA kernels.")

if __name__ == "__main__":
    main()
