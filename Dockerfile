# ─────────────────────────────────────────────
# Stage 1: Build the React / Vite frontend
# ─────────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /app/web_interface

# Install dependencies first (better layer caching)
COPY web_interface/package*.json ./
RUN npm install

# Copy source and build
COPY web_interface/ ./
RUN npm run build

# ─────────────────────────────────────────────
# Stage 2: Python + PyTorch runtime
# ─────────────────────────────────────────────
# Use NVIDIA CUDA base image so PyTorch can access the GPU.
# If you don't have a GPU, swap this for: python:3.11-slim
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source code
COPY core_packages/ ./core_packages/
COPY config.example.yaml ./config.example.yaml

# Copy the built React frontend from stage 1
COPY --from=frontend-builder /app/web_interface/dist ./web_interface/dist

# Create a directory to persist HuggingFace model cache
# Mount this as a Docker volume so models aren't re-downloaded on restart
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Expose the FastAPI server port
EXPOSE 8000

# Set PYTHONPATH so core_packages module is found
ENV PYTHONPATH=/app

# Run the server
CMD ["python3", "core_packages/server.py"]
