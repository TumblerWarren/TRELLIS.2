# RunPod RTX 4090 deployment

This fork is intended for Linux-based RunPod pods with **RTX 4090 24 GB**, **Python 3.12**, an internet connection, and a CUDA/PyTorch environment matching the pinned custom wheels.

## 1. Hugging Face permissions (do this before starting the app)

| Model | Access | Used for |
| --- | --- | --- |
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | Public / MIT | Main 3D model (~16 GB files on Hub) |
| [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | **Gated**; log in and accept Meta's terms, wait for approval if required | Image conditioning |
| [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | Public (MIT metadata) | Default local background remover |
| [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | **Gated** and noncommercial by default; separate commercial licensing | Optional, only when `TRELLIS_REMBG_MODEL=briaai/RMBG-2.0` |

The default fork **does not use** remote ZeroGPU inference. Hugging Face PRO is not required for approved model downloads.

### Recommended: browser/device-code sign-in (no manual token creation)

Before starting, accept Meta's DINOv3 model conditions at [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) using the Hugging Face account that will run the application.

After installing the RunPod dependencies, run the following inside your Pod:

```bash
cd /TRELLIS.2  # or your own cloned project path
source env/bin/activate
unset HF_TOKEN  # only needed if a previous HF_TOKEN was set

bash scripts/login_huggingface.sh
python scripts/check_runpod.py
bash scripts/run_runpod.sh
```

The login command creates a **separate Python virtual environment** (by default `$HOME/.local/share/trellis2/hf-auth-cli`) for a modern Hugging Face CLI. It does **not** upgrade the older `huggingface_hub` dependency inside TRELLIS.2's pinned Transformers environment.

Choose **Log in with your browser**. Open the URL printed in your terminal on your own computer, enter the short code, and approve. This is Hugging Face's device-code OAuth flow. You no longer need to create and paste a token manually, although an OAuth access/refresh credential is saved locally under `HF_HOME` (default `$HOME/.cache/huggingface`).

On every RunPod launch, `scripts/run_runpod.sh` detects the separate CLI and invokes `hf auth whoami` to verify and, where needed, refresh the browser-login credential *before* the older Python runtime accesses gated models. If the session expires or is revoked, run `bash scripts/login_huggingface.sh` again.

Both CLI and runtime must use **the same Linux user and `HF_HOME` path**. For a persistent model/credential cache, configure `HF_HOME` *before* logging in, and use the same setting on every subsequent launch:

```bash
export HF_HOME=/workspace/hf-cache
bash scripts/login_huggingface.sh
```

You can check the login again without interactive prompts or package installation:

```bash
bash scripts/login_huggingface.sh --check
```

Protect `HF_HOME`: it contains credentials. Browser login does not bypass DINOv3 approval. Device-code login also requires outbound network access; an offline Pod cannot perform it.

### Alternative: read token for unattended automation

Unattended pods may instead provide a protected RunPod secret called `HF_TOKEN`, belonging to the **same account approved for DINOv3**. `HF_TOKEN` takes precedence over saved browser credentials; the browser script warns and refuses to run while it is set, rather than silently using the wrong identity. Do not commit either kind of credential to GitHub or expose it in Gradio.

## 2. Installation

```bash
git clone https://github.com/TumblerWarren/TRELLIS.2.git
cd TRELLIS.2
python3.12 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-runpod.txt
# Browser sign-in (requires accepting Meta's DINOv3 conditions first):
bash scripts/login_huggingface.sh
python scripts/check_runpod.py
```

**CUDA warning:** `requirements.txt` is inherited from the official Hugging Face Space and pins CUDA 13, PyTorch 2.11, and custom prebuilt CUDA extension wheels. Some wheels were built for newer GPU architectures and may not function on a 4090 (sm_89). A successful `pip install` is not proof of runtime compatibility. If imports or generation fail with "no kernel image" or unsupported architecture, follow the [official TRELLIS.2 source-build guide](https://github.com/microsoft/TRELLIS.2#installation-steps) using a compatible CUDA toolkit; do not indiscriminately upgrade torch while retaining custom wheels.

Optional persistent model cache on RunPod network volume:

```bash
export HF_HOME=/workspace/hf-cache
```

The first launch downloads weights (including the DINOv3 gated weights) and three small HDRI assets. The repository intentionally does **not** commit model weights, Hugging Face credentials, or Git LFS pointer files.

## 3. Start the service and get a public URL

By default the RunPod launcher requests a **temporary Gradio `gradio.live` public link**. This setting is now applied both through `bash scripts/run_runpod.sh` and direct `python app.py`.

The app protects a publicly bound service by default. If `GRADIO_AUTH_USER` and `GRADIO_AUTH_PASSWORD` are not set, it generates a strong, random username/password combination for each app launch and prints them in the RunPod console. Keep the console output private; credentials change after a restart.

```bash
cd /TRELLIS.2  # or your cloned project path
source env/bin/activate

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
# Optional: omit this, the launch default is true.
export GRADIO_SHARE=true

bash scripts/run_runpod.sh
```

After model initialization, you should see Gradio's local address and—**only if the share tunnel connects**—a public URL:

```text
* Running on local URL:  http://0.0.0.0:7860
* Running on public URL: https://<random>.gradio.live
```

Use the credentials printed in the RunPod console to log in. The URL is **temporary**, not a production domain, and depends on the Gradio share service plus working outbound internet connectivity. We cannot guarantee that an external URL is generated.

### Set persistent credentials explicitly (recommended for repeated launches)

```bash
export GRADIO_AUTH_USER=admin
read -rsp 'Choose a Gradio password: ' GRADIO_AUTH_PASSWORD; echo
export GRADIO_AUTH_PASSWORD
bash scripts/run_runpod.sh
```

Keep passwords/tokens in protected RunPod secrets for automated deployments, not in Git.

### Alternative: RunPod's own HTTPS proxy (no Gradio share tunnel)

If a Gradio public URL does not appear, the service may still be running correctly. RunPod's built-in HTTP proxy is **independent of** `GRADIO_SHARE`; your Pod must have HTTP port **7860** exposed in its template/Pod configuration.

Open the HTTP service link in **RunPod → Pods → Connect**. Its format is:

`https://<POD-ID>-7860.proxy.runpod.net`

Use the same Gradio credentials. Changing exposed ports may require restarting/recreating the Pod; preserve files on persistent storage.

To run without the temporary Gradio share URL (e.g. proxy or SSH only):

```bash
export GRADIO_SHARE=false
bash scripts/run_runpod.sh
```

### Private SSH forwarding

If you do not expose a public HTTP port, forward the remote port locally:

```bash
ssh -N -L 7860:127.0.0.1:7860 -p <SSH-TCP-PORT> root@<POD-IP>
```

Open `http://localhost:7860` on your own computer. The same login requirements apply.

### Troubleshooting if the link doesn't appear

1. Look at the **full startup output**, not just the last line. Model downloads, GPU initialization or errors can prevent Gradio from starting.
2. If `Running on local URL` is displayed but no `Running on public URL`, confirm `GRADIO_SHARE=true` and that outbound access to the Gradio share service isn't blocked. See [Gradio sharing documentation](https://www.gradio.app/guides/sharing-your-app) and [Gradio status](https://status.gradio.app/).
3. Confirm the server is reachable inside RunPod: `curl -I http://127.0.0.1:7860`. An HTTP status such as 200, 302 or 401 shows a response; connection refused means the server isn't listening.
4. Use the RunPod HTTP proxy if port 7860 is exposed, or SSH forwarding. **Do not assume** that `http://0.0.0.0:7860` is a browser URL on your local computer; it is a listening interface inside the Pod.
5. If another process occupies port 7860, stop that process or change `GRADIO_SERVER_PORT` and your exposed port together.
6. If you see a Gradio authentication form, use the username/password printed when this session started.

**Security:** Anyone with a public URL can attempt to access the application. Gradio's basic authentication is useful for testing, but for shared/production environments prefer a private pod connection or an authenticated reverse proxy (e.g. Cloudflare Access). To explicitly opt out of password protection, set `TRELLIS_ALLOW_UNAUTHENTICATED=true`; this is **not recommended** on public interfaces.

## 4. Download backend compatibility (automatic)

Some RunPod images export `HF_HUB_ENABLE_HF_TRANSFER=1` even though the optional `hf_transfer` package is not installed. With older `huggingface_hub` releases this can cause **every** model preflight to fail before authentication is even checked:

```text
ValueError: Fast download using 'hf_transfer' is enabled
but 'hf_transfer' package is not available
```

This fork **automatically removes the deprecated variable before importing Hugging Face libraries**, in all supported launch paths:

- `scripts/check_runpod.py` clears it before checking Hugging Face access.
- `scripts/run_runpod.sh` clears it in the shell for both the preflight and the app.
- Direct `python app.py` clears it before importing Gradio and the model pipeline.

This is deliberate **even if `hf_transfer` happens to be installed**, because newer Hub versions no longer support the legacy backend. The fix works whether the setting is absent, `0`, `1`, `true`, or another value. It does **not** change `HF_TOKEN`, `HF_HOME`, `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_XET`, or proxy settings.

The supported `hf-xet` downloader is installed through `requirements-runpod.txt` and selected automatically by compatible Hub versions. To optionally utilize more bandwidth/CPU during model downloads, set:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

Only enable high-performance mode if your RunPod Pod has sufficient bandwidth, CPU, and disk throughput. If Xet is disabled or unavailable, compatible Hub versions may use standard downloads. Authentication failures (401/403) and gated-model approval remain separate issues and **are never bypassed**.

## 5. Troubleshooting

- **401 / GatedRepoError for DINOv3:** accept Meta's terms on the model page using your chosen Hugging Face account; run `bash scripts/login_huggingface.sh` and `bash scripts/login_huggingface.sh --check`. If using `HF_TOKEN` for automation, ensure it belongs to the approved account.
- **401 / GatedRepoError for BRIA:** use default public `ZhengPeng7/BiRefNet`; opting into BRIA requires separately accepting BRIA terms. BRIA noncommercial licensing is not automatically suitable for commercial services.
- **Browser login isn't offered by `hf auth login`:** your original Python environment has an older CLI. Use `bash scripts/login_huggingface.sh`, which installs a modern CLI separately. Do not upgrade TRELLIS.2's pinned inference dependencies just to get browser login.
- **Browser login succeeds, but preflight fails with 401:** confirm your approved account, `HF_HOME`, and whether `HF_TOKEN` is overriding the saved browser session. For an expired OAuth session, rerun the browser login script.
- **ZeroGPU quota error:** should not occur in this fork; search for leftover remote `rmbg_client` if it does.
- **DINOv3 `layer` error:** fixed in `trellis2/modules/image_feature_extractor.py`.
- **`forest.exr` invalid / 131-byte text pointer:** restart; the app fetches and validates required HDRI images from Microsoft Space. Confirm `HF_TOKEN` if download fails.
- **`FileNotFoundError` from end_session:** missing temporary folders are now tolerated.
- **Out of memory on 24 GB:** start at resolution `512`; avoid multiple concurrent generations and other GPU workloads.
- **Custom CUDA wheel error:** run `python scripts/check_cuda_kernels.py`; if FlexGEMM fails with no kernel image, use `bash scripts/rebuild_flexgemm.sh` with a compatible CUDA developer toolkit. Additional extensions may require separate rebuilds.

## 6. CUDA sparse-convolution crash: no kernel image on device

If Gradio opens but **Generate** crashes with:

```text
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
  flex_gemm/ops/spconv/submanifold_conv3d.py
  kernels.cuda.hashmap_build_submanifold_conv_neighbour_map
```

the failing component is a **compiled FlexGEMM CUDA extension**, not Gradio, Hugging Face access, CPU RAM or a lack of GPU memory. Our Space-derived `requirements.txt` installs a third-party prebuilt `flex_gemm` wheel, which may not include kernels for the assigned GPU. Blackwell compute capability 12.0 (including RTX PRO 6000 MIG partitions) needs an `sm_120`-capable extension.

### Detect this before loading model weights

After installing requirements:

```bash
cd /TRELLIS.2
source env/bin/activate
python scripts/check_cuda_kernels.py
```

This now also runs as part of `python scripts/check_runpod.py` and `bash scripts/run_runpod.sh`, **after the model-access checks**. It executes a small real `flex_gemm.ops.spconv.sparse_submanifold_conv3d` on the GPU and synchronizes it. Unlike `torch.cuda.is_available()`, this can identify missing GPU-specific kernels. Preflight cannot guarantee every other CUDA extension will work.

### Repair: rebuild FlexGEMM for the GPU you actually have

The explicit repair script (not automatically executed, because compiling costs compute and requires a CUDA development toolkit):

```bash
cd /TRELLIS.2
source env/bin/activate

# Check that CUDA nvcc is installed and visible:
nvcc --version

bash scripts/rebuild_flexgemm.sh

# Check the extension without downloading models:
python scripts/check_cuda_kernels.py

# Then resume:
bash scripts/run_runpod.sh
```

The script detects your GPU's compute capability, selects `TORCH_CUDA_ARCH_LIST` accordingly, verifies the installed CUDA compiler, clones upstream FlexGEMM **v1.1.0** (which includes a CUDA 13 build fix), and compiles it against the **existing PyTorch** using `--no-build-isolation --no-deps`. It does not upgrade PyTorch or alter model weights. This source version may still require adjustments for some GPU/driver combinations; a successful build is not a guarantee until the smoke test passes.

**Use a CUDA *devel* image**, not a runtime-only image, to get `nvcc` and C++ build tools. For Blackwell SM 12.0, CUDA toolkit 12.8 or newer is required. A CUDA compiler major version matching PyTorch is strongly recommended. If `nvcc` is missing, change to a compatible RunPod development image rather than installing random CUDA packages into the current environment.

**Install order matters:** `requirements-runpod.txt` contains a URL to a prebuilt FlexGEMM wheel. Reinstalling those requirements **after** rebuilding may overwrite your source-built extension. Install requirements first, then repair.

If rebuilding fails, inspect the compiler output. If the smoke test succeeds but full inference later crashes in another CUDA extension (CuMesh, o_voxel, FlashAttention), that is a separate compatibility problem and must be diagnosed individually.

Do **not** simply set `SPARSE_CONV_BACKEND=spconv` as an automatic workaround: alternative backends can change sparse tensor representations and may not produce correct results with the current pipeline.

To bypass the runtime test **only if you already verified the binary elsewhere**:

```bash
export TRELLIS_SKIP_KERNEL_PREFLIGHT=1
```

This does not fix an incompatible extension.

## Verification scope

CI checks Python syntax and source regressions only. It does not have a CUDA GPU or permission to download gated model weights; full generate + export GLB on RunPod must be tested separately.
