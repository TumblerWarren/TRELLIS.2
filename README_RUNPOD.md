# RunPod RTX 4090 deployment

This fork is intended for Linux-based RunPod pods with **RTX 4090 24 GB**, **Python 3.12**, an internet connection, and a CUDA/PyTorch environment matching the pinned custom wheels.

## 1. Hugging Face permissions (do this before starting the app)

| Model | Access | Used for |
| --- | --- | --- |
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | Public / MIT | Main 3D model (~16 GB files on Hub) |
| [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | **Gated**; log in and accept Meta's terms, wait for approval if required | Image conditioning |
| [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | Public (MIT metadata) | Default local background remover |
| [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | **Gated** and noncommercial by default; separate commercial licensing | Optional, only when `TRELLIS_REMBG_MODEL=briaai/RMBG-2.0` |

The default fork **does not use** remote ZeroGPU inference. Hugging Face PRO is not required for approved model downloads. An HF read token must belong to the account granted DINOv3 access.

From your RunPod terminal:

```bash
cd /workspace/TRELLIS.2   # use the actual clone path
source env/bin/activate
hf auth login
hf auth whoami
hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json
```

If `hf auth login` is unavailable: `python -m pip install -U huggingface_hub`.

For unattended RunPod deployments use a **RunPod secret** named `HF_TOKEN` with read access to the gated model. Do not put the token in Git, Gradio, screenshots, or notebook logs. The token permits downloads; it does **not** replace acceptance of Meta's model terms.

## 2. Installation

```bash
git clone https://github.com/TumblerWarren/TRELLIS.2.git
cd TRELLIS.2
python3.12 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-runpod.txt
python scripts/check_runpod.py
```

**CUDA warning:** `requirements.txt` is inherited from the official Hugging Face Space and pins CUDA 13, PyTorch 2.11, and custom prebuilt CUDA extension wheels. Some wheels were built for newer GPU architectures and may not function on a 4090 (sm_89). A successful `pip install` is not proof of runtime compatibility. If imports or generation fail with "no kernel image" or unsupported architecture, follow the [official TRELLIS.2 source-build guide](https://github.com/microsoft/TRELLIS.2#installation-steps) using a compatible CUDA toolkit; do not indiscriminately upgrade torch while retaining custom wheels.

Optional persistent model cache on RunPod network volume:

```bash
export HF_HOME=/workspace/hf-cache
```

The first launch downloads weights (including the DINOv3 gated weights) and three small HDRI assets. The repository intentionally does **not** commit model weights, Hugging Face credentials, or Git LFS pointer files.

## 3. Start the service

```bash
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export GRADIO_SHARE=false

# Recommended for internet-exposed endpoints:
export GRADIO_AUTH_USER=admin
read -rsp 'Choose a Gradio password: ' GRADIO_AUTH_PASSWORD; echo
export GRADIO_AUTH_PASSWORD

bash scripts/run_runpod.sh
```

RunPod built-in public proxy requires HTTP port 7860 in **Expose HTTP Ports**:

`https://<POD-ID>-7860.proxy.runpod.net`

If you do not want a public endpoint, leave Gradio behind SSH and forward the port from your computer:

```bash
ssh -N -L 7860:127.0.0.1:7860 -p <EXPOSED-SSH-PORT> root@<POD-IP>
```

Then open `http://localhost:7860` on the forwarding computer.

Optional temporary Gradio share link: `export GRADIO_SHARE=true`. Anyone who can access an unprotected link can consume GPU time. For long-running/public use, prefer RunPod proxy or an authenticated **named** Cloudflare Tunnel instead of an unauthenticated share link.

## 4. Troubleshooting

- **401 / GatedRepoError for DINOv3:** accept model terms in your Hugging Face account and login with a read token from that same account.
- **401 / GatedRepoError for BRIA:** use default public `ZhengPeng7/BiRefNet`; opting into BRIA requires separately accepting BRIA terms. BRIA noncommercial licensing is not automatically suitable for commercial services.
- **ZeroGPU quota error:** should not occur in this fork; search for leftover remote `rmbg_client` if it does.
- **DINOv3 `layer` error:** fixed in `trellis2/modules/image_feature_extractor.py`.
- **`forest.exr` invalid / 131-byte text pointer:** restart; the app fetches and validates required HDRI images from Microsoft Space. Confirm `HF_TOKEN` if download fails.
- **`FileNotFoundError` from end_session:** missing temporary folders are now tolerated.
- **Out of memory on 24 GB:** start at resolution `512`; avoid multiple concurrent generations and other GPU workloads.
- **Custom CUDA wheel error:** check `nvidia-smi`, `torch.cuda.get_device_capability()`, and installed wheel architecture compatibility. GPU-specific source builds may be necessary.

## Verification scope

CI checks Python syntax and source regressions only. It does not have a CUDA GPU or permission to download gated model weights; full generate + export GLB on RunPod must be tested separately.
