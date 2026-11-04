# Docker deployment — TRELLIS.2 without bundled models

This Docker image includes CUDA 13.0, Python 3.12, PyTorch 2.11, TRELLIS.2,
and compiled GPU extensions. **It does not contain pretrained model weights,
Hugging Face credentials or a downloaded model cache.**

By default ``TRELLIS_LAZY_LOAD=true``: Gradio opens first, and model weights
download **on the first generation request**, using a persistent cache
at ``/workspace/hf-cache``. The initial request can take significantly longer
than normal generation. Future requests reuse the loaded model, and container
restarts reuse the persisted downloaded files.

**Validation status:** Dockerfile and source-level tests are present.
A full Docker image build, first-request download, successful image-to-3D
generation and GLB export have **not yet been tested on a real RunPod RTX 4090**.

## 1. Build for an RTX 4090

Use Docker with Linux AMD64 support and plenty of free disk space.
The image compiles native CUDA libraries from source for **SM 8.9**.
The build uses an NVIDIA CUDA 13.0.2 development base; this matches the
CUDA major version of PyTorch 2.11.0+cu130.

```bash
git clone https://github.com/TumblerWarren/TRELLIS.2.git
cd TRELLIS.2

docker build --platform linux/amd64 \
  --build-arg CUDA_ARCHS=8.9 \
  --build-arg MAX_JOBS=4 \
  -t trellis2-runpod:sm89 .
```

No NVIDIA GPU is required to build the Docker image; the source compilation
is explicitly configured for a target GPU architecture at build time.
The image will be large even without model weights because CUDA, PyTorch,
compiler tools and dependencies remain included.

At build time, Docker replaces the Space's Blackwell-specific binary wheels
for FlexGEMM, CuMesh, O-Voxel, nvdiffrast and nvdiffrec_render with source
builds. FlashAttention is still the pinned prebuilt wheel and may need
additional testing on your target GPU.

For other GPUs, change ``CUDA_ARCHS`` to the GPU's compute capability;
for example ``12.0`` for the RTX PRO 6000 Blackwell. Native extensions built
for only one architecture are not automatically portable to another.

## 2. Start it with a persistent cache

Your Docker host needs NVIDIA Container Toolkit and a supported GPU driver.

```bash
mkdir -p "$PWD/trellis-workspace"

docker run --gpus all --rm --name trellis2 \
  --shm-size=8g \
  -p 7860:7860 \
  -v "$PWD/trellis-workspace:/workspace" \
  -e GRADIO_SHARE=false \
  trellis2-runpod:sm89
```

Open ``http://localhost:7860`` and use the Gradio username/password printed
in container logs. If you don't provide ``GRADIO_AUTH_USER`` and
``GRADIO_AUTH_PASSWORD``, the application creates temporary credentials for
each launch. For a stable password, use a protected secret or environment
variable; do not bake it into a Dockerfile.

A bind mount or RunPod persistent/network volume is essential. When
``/workspace`` is persistent, model files stay under ``hf-cache/`` and
generated work lives under ``trellis-tmp/``. Deleting the volume removes
cached weights and browser OAuth credentials. **No model files are part
of the image itself.**

## 3. Hugging Face login without pasting a token

The DINOv3 model remains gated. Accept its access conditions at
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
using the account you will use inside the container.

While the container is running, open another terminal:

```bash
docker exec -it trellis2 bash scripts/login_huggingface.sh
docker restart trellis2
```

The login script uses the browser/device-code flow. The OAuth credential
is saved in ``/workspace/hf-cache``, while the separate newer CLI environment
lives at ``/workspace/hf-auth-cli``. Neither is in the Docker image.

In a RunPod console running the image, use:

```bash
cd /opt/trellis
bash scripts/login_huggingface.sh
```

Restart the app process after login if an inference request had previously
failed authorization. For unattended Pods, a protected ``HF_TOKEN`` secret
can still be used, but must never be passed as a build argument.

## 4. RunPod deployment

Build and push the image to a registry you control (such as GHCR or Docker Hub).
Use its published image name in a custom RunPod Pod template, expose **HTTP
port 7860**, and attach persistent storage mounted at ``/workspace``.

The default is ``GRADIO_SHARE=false`` to avoid relying on the temporary
``gradio.live`` tunnel. Open the HTTP service URL under
**RunPod → Pods → Connect**; the interface remains protected by
Gradio authentication.

When the container starts, ``scripts/run_runpod.sh`` runs the GPU smoke
checks for FlexGEMM, O-Voxel and CuMesh but skips network-based model
preflight. It starts Gradio without model downloads, then downloads/loads
models lazily on first use. To restore eager startup loading (still no
weights inside the image), set ``TRELLIS_LAZY_LOAD=false``.

## 5. Warnings and troubleshooting

- **Native CUDA compile failure:** verify CUDA compiler version and
  PyTorch CUDA major version, decrease ``MAX_JOBS`` for low build RAM.
  On a real GPU, review the container logs for architecture errors.
- **No GPU visible:** ensure the Pod has a GPU assigned, or use Docker's
  ``--gpus all`` with NVIDIA Container Toolkit.
- **401/403 during first generation:** sign in and accept DINOv3 terms.
  A browser login does not bypass gating.
- **First generation takes a long time:** the 4B model and associated
  DINOv3/background models are downloading and initializing. On subsequent
  container runs they are served from the persistent cache.
- **The Gradio URL works but generation fails:** the CUDA extension build
  has not been fully validated. Use ``CUDA_LAUNCH_BLOCKING=1`` temporarily
  to locate the failing operation; test the result, not just imports.
- **Images are not small:** excluding model weights reduces the image
  size but the CUDA SDK, PyTorch and native dependencies remain substantial.
- **Do not mount your home directory as a Docker build context:** it may
  contain secrets or cached weights. The ``.dockerignore`` supplied here
  excludes common model, token and development artifacts.

For the non-Docker/manual installation path see ``README_RUNPOD.md``.
