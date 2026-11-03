# TRELLIS.2 — RunPod-ready Hugging Face Space fork

RunPod deployment of Microsoft's [TRELLIS.2 Hugging Face Space](https://huggingface.co/spaces/microsoft/TRELLIS.2), adapted for a **single NVIDIA RTX 4090 (24 GB)**. Upstream project: [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Changes from the Space

- Local image background removal via `ZhengPeng7/BiRefNet` on **CPU**; no Hugging Face ZeroGPU API, quota, or BRIA model gate by default.
- Configurable background-removal model (`TRELLIS_REMBG_MODEL`), with optional BRIA selection subject to its license and access conditions.
- Compatibility with alternate DINOv3 Transformers layer layouts.
- Automatically download and validate forest/sunset/courtyard HDRI files through the Hugging Face Hub, instead of relying on Git LFS pointers.
- Handle missing optional example images and missing per-session temporary directories.
- Provide Gradio 6.1.0 dependency, RunPod launcher, Hugging Face access preflight, and optional Gradio basic authentication.
- Default to 512 resolution for a conservative first run.

## Model access checklist

Complete these steps **before starting TRELLIS.2 on RunPod**:

- [ ] **Approve Meta DINOv3 access:** Sign in to Hugging Face and accept the model's access/license conditions at [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Wait for approval if required.
- [ ] **Create a read token:** Create a [Hugging Face access token](https://huggingface.co/settings/tokens) from the **same account** approved for DINOv3. If using a fine-grained token, grant it read access to the gated DINOv3 repository.
- [ ] **Authenticate RunPod:** Run `hf auth login` and `hf auth whoami`, **or** supply your token via a protected RunPod secret/environment variable named `HF_TOKEN`. Never commit or paste your token into the repository.
- [ ] **Verify model access:** Run `hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json`, followed by `python scripts/check_runpod.py` once dependencies are installed.

The main [Microsoft TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) weights and default [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) background remover are public; **no separate approval is normally required** for them. [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) is **optional** and has separate access and licensing conditions. A token cannot bypass model-owner approval; Hugging Face PRO is not required merely to download models your account is allowed to access.

For full instructions, see [RunPod model permissions](README_RUNPOD.md#1-hugging-face-permissions-do-this-before-starting-the-app).

## Quick start on RunPod

Python **3.12**, Linux, a working NVIDIA CUDA environment, and an internet-enabled 24 GB GPU are needed.

```bash
git clone https://github.com/TumblerWarren/TRELLIS.2.git
cd TRELLIS.2
python3.12 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-runpod.txt

# Accept the DINOv3 model terms on Hugging Face first.
hf auth login
python scripts/check_runpod.py
bash scripts/run_runpod.sh
```

Open `https://<POD-ID>-7860.proxy.runpod.net` **only if port 7860 is exposed in RunPod**. For a private alternative, use SSH forwarding. Secure any publicly accessible interface.

For detailed model access, license notes, troubleshooting, and CUDA wheel caveats see [README_RUNPOD.md](README_RUNPOD.md).

**Status:** source compatibility changes + static syntax tests; end-to-end inference must be verified on the actual RTX 4090 pod. The Hugging Face Space's pinned CUDA 13 / PyTorch 2.11 third-party wheel set may not support every CUDA architecture. If an extension fails to execute on your GPU, use the upstream source-build instructions instead of assuming that Python package installation alone proves compatibility.

## Credits and licensing

The application derives from Microsoft's TRELLIS.2 Space and [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), released under MIT. Model weights and third-party components have their **own terms**. In particular, Meta DINOv3 requires accepting its model license; BRIA RMBG-2.0 has separate noncommercial/commercial licensing conditions. No model weights or private Hugging Face credentials belong in this repository.
