FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch (CUDA 11.3 wheels) + deps + FrEIA
RUN python -m pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1+cu113 \
    && pip install \
        numpy \
        scipy \
        scikit-learn \
        scikit-image \
        pillow \
        tqdm \
        matplotlib \
        opencv-python-headless \
        wandb \
        git+https://github.com/VLL-HD/FrEIA.git

# Copy code into image if building from repo root
# COPY . /workspace

# Default command: show help
CMD ["python", "main.py", "--help"]
