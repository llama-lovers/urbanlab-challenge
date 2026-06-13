FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: Python 3.10 + audio deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    ffmpeg libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make `python` and `pip` available (optional but convenient)
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Install Python deps (cached layer)
COPY req.txt /app/req.txt

ARG USE_CUDA

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/req.txt && \
    if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing PyTorch with CUDA..." && \
        python -m pip install --force-reinstall --no-cache-dir \
          --index-url https://download.pytorch.org/whl/cu128 \
          torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0; \
    fi

# Copy only app code (keeps cache efficient)
COPY app /app/app

EXPOSE 8000

# Default command (prod). In dev override via docker-compose command with --reload
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]