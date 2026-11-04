# TRELLIS.2 — RunPod-ready Hugging Face Space fork

RunPod deployment of Microsoft's [TRELLIS.2 Hugging Face Space](https://huggingface.co/spaces/microsoft/TRELLIS.2), adapted for a **single NVIDIA RTX 4090 (24 GB)**. Upstream project: [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Changes from the Space

- Local image background removal via `ZhengPeng7/BiRefNet` on **CPU**; no Hugging Face ZeroGPU API, quota, or BRIA model gate by default.
- Configurable background-removal model (`TRELLIS_REMBG_MODEL`), with optional BRIA selection subject to its license and access conditions.
- Compatibility with alternate DINOv3 Transformers layer layouts.
- Automatically download and validate forest/sunset/courtyard HDRI files through the Hugging Face Hub, instead of relying on Git LFS pointers.
- Handle missing optional example images and missing per-session temporary directories.
- Provide Gradio 6.1.0 dependency, RunPod launcher, Hugging Face access preflight, and automatically generated Gradio login credentials when needed.
- Use a separate modern Hugging Face CLI for browser/device-code sign-in without upgrading the inference environment; verify/refresh OAuth credentials at launch.
- Ignore the deprecated RunPod `HF_HUB_ENABLE_HF_TRANSFER` setting before Hub imports; use `hf-xet` instead of requiring legacy `hf_transfer`.
- Preflight-test FlexGEMM sparse convolution, O-Voxel mesh hashmap and CuMesh connectivity CUDA kernels before loading the 4B model; provide architecture-specific rebuild scripts for incompatible wheels.
- Default to 512 resolution for a conservative first run.

## Model access checklist

Complete these steps **before starting TRELLIS.2 on RunPod**:

- [ ] **Approve Meta DINOv3 access:** Sign in to Hugging Face and accept the model's access/license conditions at [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Wait for approval if required.
- [ ] **Sign in using your browser (recommended):** Run `bash scripts/login_huggingface.sh`. The script installs a modern CLI in an isolated environment, shows a browser URL and verification code, and saves an OAuth credential in `HF_HOME`. No manually created access token is required.
- [ ] **Use your approved account:** Make sure the Hugging Face account used for browser sign-in is the same one approved for DINOv3. The login script verifies your identity.
- [ ] **Verify gated-model access:** Run `python scripts/check_runpod.py` and then `bash scripts/run_runpod.sh`. The launcher checks and refreshes an existing browser session before model loading.
- [ ] **Unattended Pod alternative:** Store a read `HF_TOKEN` as a RunPod secret if interactive browser login is inappropriate. `HF_TOKEN` overrides browser credentials. Run `unset HF_TOKEN` in your current terminal before switching methods.

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
bash scripts/login_huggingface.sh
python scripts/check_runpod.py
bash scripts/run_runpod.sh
```

After model loading, the app **attempts to print a temporary `https://*.gradio.live` link by default**, protected by a randomly generated username/password displayed in the RunPod terminal unless you supply your own. The public URL is only created if Gradio's outbound tunnel succeeds. Set `GRADIO_SHARE=false` to disable this link. RunPod also offers `https://<POD-ID>-7860.proxy.runpod.net` **when port 7860 is exposed**; it does not depend on Gradio sharing. For private access, use SSH forwarding.

If you provide no credentials, both public sharing and non-loopback binding use a generated password. See [README_RUNPOD.md](README_RUNPOD.md#3-start-the-service-and-get-a-public-url).

For detailed model access, license notes, troubleshooting, and CUDA wheel caveats see [README_RUNPOD.md](README_RUNPOD.md).

**Status:** source compatibility changes + static syntax tests; GPU-specific FlexGEMM kernels are checked at runtime and full TRELLIS.2 inference/export still requires verification on your pod. The Hugging Face Space's pinned CUDA 13 / PyTorch 2.11 third-party wheel set may not support every CUDA architecture. If an extension fails to execute on your GPU, use the upstream source-build instructions instead of assuming that Python package installation alone proves compatibility.

## Credits and licensing

The application derives from Microsoft's TRELLIS.2 Space and [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), released under MIT. Model weights and third-party components have their **own terms**. In particular, Meta DINOv3 requires accepting its model license; BRIA RMBG-2.0 has separate noncommercial/commercial licensing conditions. No model weights or private Hugging Face credentials belong in this repository.
