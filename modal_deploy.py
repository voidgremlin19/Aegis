"""
modal_deploy.py

Modal deployment for Project Aegis backend.

Deploys the FastAPI server (core_packages/server.py) as a serverless
GPU-backed ASGI app on Modal. Supports:
  - GPT-2          (T4 GPU, ~15s cold start)
  - Llama 3.2 3B   (A10G GPU, ~40s cold start)
  - Qwen 2.5 3B    (A10G GPU, ~40s cold start)

Deploy:
    modal deploy modal_deploy.py

Run locally (for testing):
    modal serve modal_deploy.py

Set secrets in Modal dashboard (or via CLI):
    modal secret create aegis-secrets \\
        AEGIS_API_KEY=your-secret-key \\
        AEGIS_ALLOWED_ORIGINS=https://your-vercel-app.vercel.app \\
        HF_TOKEN=your-huggingface-token
"""

import modal

# ---------------------------------------------------------------------------
# Image: all Python dependencies + model cache pre-warmed
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "websockets>=12.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "python-json-logger>=2.0.7",
        "slowapi>=0.1.9",
    )
    # Pre-download GPT-2 into the image so it loads instantly (no cold-start download)
    .run_commands(
        "python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "AutoModelForCausalLM.from_pretrained('gpt2'); "
        "AutoTokenizer.from_pretrained('gpt2')\""
    )
)

# ---------------------------------------------------------------------------
# App and Volume (model cache for larger models downloaded at runtime)
# ---------------------------------------------------------------------------

app = modal.App("aegis-backend", image=image)

# Persistent volume to cache HuggingFace models across cold starts
# Llama/Qwen are too large to bake into the image, so we cache them here.
model_cache = modal.Volume.from_name("aegis-model-cache", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Secrets (set these in the Modal dashboard or via `modal secret create`)
# ---------------------------------------------------------------------------

secrets = [modal.Secret.from_name("aegis-secrets", required=False)]

# ---------------------------------------------------------------------------
# ASGI app — wraps the existing FastAPI `app` from core_packages/server.py
# ---------------------------------------------------------------------------

@app.function(
    # A10G handles GPT-2, Llama-3.2-3B, and Qwen-2.5-3B comfortably.
    # Swap to gpu="A100" if you add Mistral-7B or Gemma-2-9B.
    gpu="A10G",
    memory=20480,  # 20 GB RAM — comfortable for 3B models in float32
    timeout=600,   # 10 min max per request (generous for long generations)
    container_idle_timeout=300,  # Keep warm for 5 min after last request
    volumes={HF_CACHE_DIR: model_cache},
    secrets=secrets,
    # Mount the local core_packages directory into the container
    mounts=[
        modal.Mount.from_local_dir(
            "./core_packages",
            remote_path="/root/core_packages",
        ),
        modal.Mount.from_local_file(
            "./pyproject.toml",
            remote_path="/root/pyproject.toml",
        ),
    ],
)
@modal.asgi_app()
def fastapi_app():
    import sys
    import os

    # Add project root to path so `from core_packages.X import Y` works
    sys.path.insert(0, "/root")

    # Ensure HuggingFace uses the volume-backed cache
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

    # Default to GPT-2 on startup (user can switch via the model selector UI)
    os.environ.setdefault("AEGIS_MODEL_NAME", "gpt2")

    # Import and return the FastAPI app
    from core_packages.server import app as aegis_app
    return aegis_app
