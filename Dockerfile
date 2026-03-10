FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    OUTPUT_DIR=/workspace/outputs \
    TMP_ROOT=/workspace/workspace_tmp \
    MODEL_VENV=/opt/model-venv \
    WORKER_VENV=/opt/worker-venv \
    MODEL_SERVER_URL=http://127.0.0.1 \
    MODEL_SERVER_PORT=8000

WORKDIR /workspace/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip \
    ffmpeg git git-lfs curl ca-certificates build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ===== model venv =====
RUN python -m venv /opt/model-venv
RUN /opt/model-venv/bin/python -m pip install --upgrade pip setuptools wheel
RUN /opt/model-venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchaudio==2.3.1+cu121

COPY requirements-model.txt ./
RUN sed -i 's/\r$//' /workspace/app/requirements-model.txt
RUN /opt/model-venv/bin/pip install -r requirements-model.txt

# ===== worker venv =====
RUN python -m venv /opt/worker-venv
RUN /opt/worker-venv/bin/python -m pip install --upgrade pip setuptools wheel

COPY requirements-worker.txt ./
RUN sed -i 's/\r$//' /workspace/app/requirements-worker.txt
RUN /opt/worker-venv/bin/pip install -r requirements-worker.txt

# sanity checks
RUN /opt/model-venv/bin/python -c "import torch, fastapi, whisperx, demucs, soundfile; print('model imports ok')"
RUN /opt/worker-venv/bin/python -c "from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig; print('worker imports ok')"

COPY server.py worker.py start.sh test.py .env.example README.md .gitignore ./
RUN sed -i 's/\r$//' /workspace/app/start.sh /workspace/app/server.py /workspace/app/worker.py /workspace/app/test.py /workspace/app/README.md /workspace/app/.env.example /workspace/app/.gitignore
RUN chmod +x /workspace/app/start.sh

EXPOSE 8000
CMD ["/workspace/app/start.sh"]
