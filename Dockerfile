# Minimal CUDA-enabled training image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

# Example usage:
# docker build -t rl-trainer .
# docker run --gpus all -it --rm -v %cd%/models:/app/models -v %cd%/runs:/app/runs rl-trainer \
#   python3 train.py --config presets/chet_sim_rainbow.json
