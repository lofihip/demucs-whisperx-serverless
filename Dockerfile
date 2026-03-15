FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    WORK_ROOT=/workspace/runtime \
    MODEL_CACHE_DIR=/workspace/models \
    TEMP_ROOT=/workspace/tmp \
    HF_HOME=/workspace/models/huggingface \
    TORCH_HOME=/workspace/models/torch \
    XDG_CACHE_HOME=/workspace/models/.cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install \
    dora-search \
    einops \
    julius>=0.2.3 \
    lameenc>=1.2 \
    openunmix \
    pyyaml \
    tqdm

RUN python3 -m pip install "whisperx @ git+https://github.com/m-bain/whisperX.git"

RUN python3 -m pip install --no-deps "demucs @ git+https://github.com/facebookresearch/demucs.git"

COPY . /app

RUN mkdir -p /workspace/runtime /workspace/models /workspace/tmp && chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/start.sh"]
