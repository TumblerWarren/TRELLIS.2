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
