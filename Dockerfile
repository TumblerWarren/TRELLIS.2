# GPU-ready TRELLIS.2 application image; no Hugging Face model weights included.
ARG CUDA_BASE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_BASE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS=8.9
ARG MAX_JOBS=4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCHS} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/workspace/hf-cache \
    HF_BROWSER_CLI_VENV=/workspace/hf-auth-cli \
    TRELLIS_TMP_DIR=/workspace/trellis-tmp \
    TRELLIS_LAZY_LOAD=true \
    TRELLIS_SKIP_MODEL_PREFLIGHT=true \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    git git-lfs ca-certificates curl build-essential cmake ninja-build pkg-config \
    libgl1 libegl1 libglib2.0-0 libjpeg-dev libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/trellis-venv
ENV PATH=/opt/trellis-venv/bin:${PATH}
WORKDIR /opt/trellis

COPY requirements.txt requirements-runpod.txt ./
RUN python -m pip install --upgrade pip setuptools wheel ninja && \
    python -m pip install -r requirements-runpod.txt

# Override the Space's Blackwell-specific extension wheels using source builds
# targeted at the intended GPU. No GPU or model downloads needed at build time.
COPY scripts/docker_build_extensions.sh /usr/local/bin/trellis-build-extensions
RUN bash /usr/local/bin/trellis-build-extensions

COPY . /opt/trellis
RUN python -m compileall -q app.py scripts trellis2 && \
    mkdir -p /workspace/hf-cache /workspace/trellis-tmp /workspace/outputs

EXPOSE 7860
# Kernel smoke checks run before the UI. Model/network preflight is skipped
# because model downloads and gated authorization happen on first generation.
CMD ["bash", "scripts/run_runpod.sh"]
