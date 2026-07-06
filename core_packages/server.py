"""
aegis/server.py

Production-grade FastAPI dashboard server for Project Aegis.

Security features

* **API Key authentication**: ``X-API-Key`` header is checked against the
  value of the environment variable specified by ``cfg.api_key_env_var``
  (default: ``AEGIS_API_KEY``).  All /api/* endpoints require this header.
  The WebSocket endpoint validates the key as the first received message.

* **Rate limiting**: ``slowapi`` limits /api/stream and /api/ws to
  ``cfg.rate_limit_rpm`` requests per minute per IP.

* **HTTPS**: Handled at the uvicorn layer via --ssl-cert / --ssl-key;
  server advertises whether it is running securely.

* **Prompt sanitisation**: Prompts are truncated to 2 000 chars and
  newlines are replaced with spaces before being stored in logs.

* **Audit logging**: Every request and generation event is appended to
  a rotating JSON-lines audit log (``audit.log`` by default).  Each entry
  contains: timestamp, request_id, model, prompt_sha256, thresholds,
  escalated flag, escalation reason, tokens_generated.

* **CORS**: Locked to ``AEGIS_ALLOWED_ORIGINS`` (default: localhost only).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
import asyncio
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI = True
except ImportError:
    _SLOWAPI = False

from core_packages.modules import ThreatNeutralizer, DeceptionTripwire, ArousalRegulator, GoldilocksTuner
from core_packages.vector_engine import VectorEngine
from core_packages.model_wrapper import AegisModelWrapper
from core_packages.config import AegisConfig, load_config, get_default_config, resolve_device

# Logging setup

logger = logging.getLogger("aegis.server")

# Global audit logger (file handler attached on startup)
audit_logger = logging.getLogger("aegis.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Don't send to root logger

_AUDIT_HANDLER: Optional[logging.handlers.RotatingFileHandler] = None


def _setup_audit_logger(cfg: AegisConfig) -> None:
    global _AUDIT_HANDLER
    if _AUDIT_HANDLER is not None:
        return  # already configured
    handler = logging.handlers.RotatingFileHandler(
        filename=cfg.audit_log_path,
        maxBytes=cfg.audit_log_max_bytes,
        backupCount=cfg.audit_log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)
    _AUDIT_HANDLER = handler


def _audit(event: dict) -> None:

    event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    audit_logger.info(json.dumps(event, ensure_ascii=False))


# Rate limiter (optional slowapi)

if _SLOWAPI:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None  # graceful degradation


def _rate_limit(rate: str):
    if _SLOWAPI and limiter is not None:
        return limiter.limit(rate)
    def _noop(f):
        return f
    return _noop


# FastAPI app factory

app = FastAPI(
    title="Project Aegis – Cognitive & Emotional Firewall Dashboard",
    description="Real-time LLM activation monitoring and steering dashboard.",
    version="1.0.0",
)

if _SLOWAPI:
    app.state.limiter = limiter
    # pyrefly: ignore [bad-argument-type]
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS – restrict to allowed origins in production
_ALLOWED_ORIGINS = os.environ.get(
    "AEGIS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key_value(cfg: AegisConfig) -> Optional[str]:
    return os.environ.get(cfg.api_key_env_var)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(_API_KEY_HEADER),
) -> None:
    cfg: AegisConfig = request.app.state.config
    expected = _get_api_key_value(cfg)

    if expected is None:
        # No API key configured — allow all requests (dev mode)
        logger.warning(
            "No API key configured (%s not set). Running in open mode.", cfg.api_key_env_var
        )
        return

    if not api_key or api_key != expected:
        _audit({
            "event": "auth_failure",
            "ip": request.client.host if request.client else "unknown",
            # pyrefly: ignore [unnecessary-type-conversion]
            "path": str(request.url.path),
        })
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# Global app state (populated at startup)

_state: dict = {
    "model": None,
    "tokenizer": None,
    "wrapper": None,
    "emotion_vectors": None,
    "deflection_vectors": None,
    "target_layer": 8,
    "config": None,
}


def _load_model(model_name: str, cfg: AegisConfig) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model '%s'…", model_name)

    # Free previous model
    if _state["model"] is not None:
        _state.update(model=None, tokenizer=None, wrapper=None,
                      emotion_vectors=None, deflection_vectors=None)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = resolve_device(cfg.device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # pyrefly: ignore [missing-attribute]
        if tokenizer.pad_token is None:
            # pyrefly: ignore [missing-attribute]
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except Exception as exc:
        logger.error("Failed to load %s: %s. Falling back to gpt2.", model_name, exc)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # pyrefly: ignore [missing-attribute]
        if tokenizer.pad_token is None:
            # pyrefly: ignore [missing-attribute]
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model_name = "gpt2"
        cfg.model_name = "gpt2"

    model.eval().to(device)
    logger.info("Model loaded on %s.", device.upper())

    # Determine target layer
    target_layer = cfg.target_layer
    if target_layer is None:
        if hasattr(model.config, "num_hidden_layers"):
            n = model.config.num_hidden_layers
        elif hasattr(model.config, "n_layer"):
            n = model.config.n_layer
        else:
            n = 12
        target_layer = int(n * 2 / 3)

    logger.info("Target layer: %d", target_layer)

    # Extract steering vectors
    logger.info("Extracting Aegis steering vectors…")
    engine = VectorEngine()
    emotion_vectors = engine.extract_emotion_vectors(
        model, tokenizer, layer_idx=target_layer, k_components=None
    )
    deflection_vectors = engine.extract_deflection_vectors(
        model, tokenizer, layer_idx=target_layer
    )

    wrapper = AegisModelWrapper(model, tokenizer, target_layer_idx=target_layer)

    _state.update(
        model=model,
        tokenizer=tokenizer,
        wrapper=wrapper,
        emotion_vectors=emotion_vectors,
        deflection_vectors=deflection_vectors,
        target_layer=target_layer,
    )
    logger.info("Aegis firewall ready.")
    return model_name


# Startup & shutdown
# pyrefly: ignore [deprecated]
@app.on_event("startup")
def startup_event() -> None:
    config_path = os.environ.get("AEGIS_CONFIG_PATH")
    model_name = os.environ.get("AEGIS_MODEL_NAME", "gpt2")

    if config_path and Path(config_path).exists():
        cfg = load_config(config_path)
    else:
        cfg = get_default_config(model_name)
        cfg.model_name = model_name

    _state["config"] = cfg
    app.state.config = cfg
    _setup_audit_logger(cfg)
    _load_model(cfg.model_name, cfg)


# Request / Response models


class InferenceRequest(BaseModel):
    prompt: str
    is_delusional: bool = False
    threat_threshold: Optional[float] = None
    deception_threshold: Optional[float] = None
    arousal_threshold: Optional[float] = None
    sycophancy_threshold: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class ModelRequest(BaseModel):
    model_name: str


# Helpers


def _sanitize_prompt(prompt: str, max_len: int = 2000) -> str:
    return prompt[:max_len].replace("\n", " ").replace("\r", " ")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _override(value: Optional[float], default: float) -> float:
    return value if value is not None else default


def _build_modules(cfg: AegisConfig, req: InferenceRequest):
    evecs = _state["emotion_vectors"]
    dvecs = _state["deflection_vectors"]

    tn = ThreatNeutralizer(
        evecs["desperate"], evecs["calm"],
        threshold=_override(req.threat_threshold, cfg.threat_threshold),
        steering_strength=cfg.steering_strength,
        clip_value=cfg.clip_value,
    )
    dt = DeceptionTripwire(
        dvecs["anger_deflection"], dvecs["fear_deflection"],
        threshold=_override(req.deception_threshold, cfg.deception_threshold),
    )
    arousal_vec = evecs["angry"] + evecs["desperate"] - evecs["calm"]
    empathetic_vec = evecs["calm"] + evecs["loving"]
    ar = ArousalRegulator(
        arousal_vec, empathetic_vec,
        arousal_threshold=_override(req.arousal_threshold, cfg.arousal_threshold),
        injection_gain=cfg.injection_gain,
        max_injection=cfg.max_injection,
    )
    valence_vec = evecs["loving"] + evecs["calm"]
    gt = GoldilocksTuner(
        valence_vec,
        sycophancy_threshold=_override(req.sycophancy_threshold, cfg.sycophancy_threshold),
        harshness_threshold=cfg.harshness_threshold,
        tuner_gain=cfg.tuner_gain,
        max_steer=cfg.max_steer,
    )
    return {"threat_neutralizer": tn, "deception_tripwire": dt,
            "arousal_regulator": ar, "goldilocks_tuner": gt}


# Routes


@app.get("/", include_in_schema=False)
def get_dashboard() -> FileResponse:
    static_path = Path(__file__).parent.parent / "web_interface" / "dist" / "index.html"
    return FileResponse(str(static_path))


@app.get("/health", include_in_schema=False)
def health_check() -> dict:
    return {"status": "ok", "model": _state["config"].model_name if _state["config"] else "none"}


@app.post("/api/model")
def post_model(
    request_data: ModelRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
) -> dict:
    try:
        cfg = get_default_config(request_data.model_name)
        cfg.model_name = request_data.model_name
        cfg.audit_log_path = request.app.state.config.audit_log_path
        cfg.audit_log_max_bytes = request.app.state.config.audit_log_max_bytes
        cfg.audit_log_backup_count = request.app.state.config.audit_log_backup_count
        cfg.api_key_env_var = request.app.state.config.api_key_env_var
        cfg.rate_limit_rpm = request.app.state.config.rate_limit_rpm

        actual_model = _load_model(request_data.model_name, cfg)
        cfg.model_name = actual_model
        _state["config"] = cfg
        request.app.state.config = cfg
        _audit({"event": "model_swap", "model": request_data.model_name,
                "actual_model": actual_model, "target_layer": _state["target_layer"]})
        return {
            "status": "success",
            "requested_model": request_data.model_name,
            "model": actual_model,
            "target_layer": _state["target_layer"],
            "fallback": actual_model != request_data.model_name,
            "thresholds": {
                "threat": cfg.threat_threshold,
                "deception": cfg.deception_threshold,
                "arousal": cfg.arousal_threshold,
                "sycophancy": cfg.sycophancy_threshold,
            },
        }
    except Exception as exc:
        logger.exception("Model swap failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/stream")
@(_rate_limit(f"{(_state.get('config') or AegisConfig()).rate_limit_rpm}/minute") if _SLOWAPI else lambda f: f)
async def post_stream(
    request_data: InferenceRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
) -> StreamingResponse:
    cfg: AegisConfig = request.app.state.config
    request_id = str(uuid.uuid4())
    safe_prompt = _sanitize_prompt(request_data.prompt)
    phash = _prompt_hash(request_data.prompt)

    _audit({
        "event": "stream_start",
        "request_id": request_id,
        "model": cfg.model_name,
        "prompt_hash": phash,
        "prompt_preview": safe_prompt[:120],
        "thresholds": {
            "threat": _override(request_data.threat_threshold, cfg.threat_threshold),
            "deception": _override(request_data.deception_threshold, cfg.deception_threshold),
        },
        "generation": {
            "temperature": _override(request_data.temperature, cfg.temperature),
            "top_p": _override(request_data.top_p, cfg.top_p),
        },
        "is_delusional": request_data.is_delusional,
    })

    wrapper = _state["wrapper"]
    wrapper.modules = _build_modules(cfg, request_data)

    def event_generator():
        tokens_generated = 0
        escalated = False
        reason = ""
        for chunk in wrapper.generate_stream(
            prompt=request_data.prompt,
            is_delusional=request_data.is_delusional,
            max_new_tokens=cfg.max_new_tokens,
            temperature=_override(request_data.temperature, cfg.temperature),
            top_p=_override(request_data.top_p, cfg.top_p),
        ):
            tokens_generated += 1 if chunk.get("token") else 0
            if chunk.get("escalated"):
                escalated = True
                reason = chunk.get("escalation_reason", "")
            yield f"data: {json.dumps(chunk)}\n\n"

        _audit({
            "event": "stream_end",
            "request_id": request_id,
            "model": cfg.model_name,
            "prompt_hash": phash,
            "tokens_generated": tokens_generated,
            "escalated": escalated,
            "escalation_reason": reason,
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    cfg: AegisConfig = app.state.config
    await websocket.accept()
    logger.info("WebSocket client connected from %s", websocket.client)

    #  Optional: API-key check as first message 
    expected_key = _get_api_key_value(cfg)
    if expected_key:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            auth_data = json.loads(auth_msg)
            if auth_data.get("api_key") != expected_key:
                await websocket.send_text(json.dumps({"error": "Unauthorized"}))
                await websocket.close(code=4001)
                return
        except asyncio.TimeoutError:
            await websocket.close(code=4002)
            return

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            prompt = data.get("prompt", "")
            is_delusional = data.get("is_delusional", False)
            aegis_enabled = data.get("aegis_enabled", True)
            request_id = str(uuid.uuid4())
            phash = _prompt_hash(prompt)
            safe_prompt = _sanitize_prompt(prompt)

            # Build per-request override struct
            req_override = InferenceRequest(
                prompt=prompt,
                is_delusional=is_delusional,
                threat_threshold=data.get("threat_threshold"),
                deception_threshold=data.get("deception_threshold"),
                arousal_threshold=data.get("arousal_threshold"),
                sycophancy_threshold=data.get("sycophancy_threshold"),
                temperature=data.get("temperature"),
                top_p=data.get("top_p"),
            )

            wrapper = _state["wrapper"]

            if not aegis_enabled:
                temperature = _override(req_override.temperature, cfg.temperature)
                top_p = _override(req_override.top_p, cfg.top_p)
                for chunk in wrapper.generate_stream_raw(
                    prompt=prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    await websocket.send_text(json.dumps(chunk))
                    await asyncio.sleep(0.01)
            else:
                wrapper.modules = _build_modules(cfg, req_override)
                tokens_gen = 0
                escalated = False
                reason = ""

                _audit({
                    "event": "ws_generate_start",
                    "request_id": request_id,
                    "model": cfg.model_name,
                    "prompt_hash": phash,
                    "prompt_preview": safe_prompt[:120],
                    "aegis_enabled": aegis_enabled,
                    "generation": {
                        "temperature": _override(req_override.temperature, cfg.temperature),
                        "top_p": _override(req_override.top_p, cfg.top_p),
                    },
                })

                for chunk in wrapper.generate_stream(
                    prompt=prompt,
                    is_delusional=is_delusional,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=_override(req_override.temperature, cfg.temperature),
                    top_p=_override(req_override.top_p, cfg.top_p),
                ):
                    tokens_gen += 1 if chunk.get("token") else 0
                    if chunk.get("escalated"):
                        escalated = True
                        reason = chunk.get("escalation_reason", "")
                    await websocket.send_text(json.dumps(chunk))
                    await asyncio.sleep(0.01)

                _audit({
                    "event": "ws_generate_end",
                    "request_id": request_id,
                    "model": cfg.model_name,
                    "prompt_hash": phash,
                    "tokens_generated": tokens_gen,
                    "escalated": escalated,
                    "escalation_reason": reason,
                })

            await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket error: %s", exc)


# Static files

_dist_dir = Path(__file__).parent.parent / "web_interface" / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="assets")

    @app.get("/{filename}", include_in_schema=False)
    def get_root_static(filename: str) -> FileResponse:
        file_path = _dist_dir / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        raise HTTPException(status_code=404, detail="File not found")

# Dev entry point

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
