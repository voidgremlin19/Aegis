"""
aegis/config.py
Pydantic-validated configuration system for Project Aegis.

Supports YAML and JSON configuration files.  Provides per-model defaults
derived from calibration runs on common open-weight models.

Usage

    from core_packages.config import load_config, get_default_config

    cfg = load_config("config.yaml")          # from file
    cfg = get_default_config("gpt2")          # built-in defaults
    cfg = get_default_config("llama-3-8b")    # Llama 3 8B defaults
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator as field_validator
    _PYDANTIC_V2 = False


class AegisConfig(BaseModel):

    #  Model & runtime 
    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model identifier or local path.",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Compute device.  'auto' selects MPS > CUDA > CPU.",
    )
    target_layer: Optional[int] = Field(
        default=None,
        description=(
            "Transformer layer index at which to attach the hook.  "
            "None means auto-select via VectorEngine.suggest_layer()."
        ),
    )

    #  Module A – Threat Neutralizer 
    threat_threshold: float = Field(
        default=0.12,
        ge=0.0, le=1.0,
        description="Cosine-similarity threshold that triggers desperate-signal removal.",
    )
    steering_strength: float = Field(
        default=0.08,
        ge=0.0, le=1.0,
        description="Fraction of ||x_t|| injected as calm vector when triggered.",
    )
    clip_value: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Activation clamp: ±clip_value * ||x_t||.  "
            "None disables clamping.  0.5 is safe for all tested models."
        ),
    )

    #  Module B – Deception Tripwire 
    deception_threshold: float = Field(
        default=0.15,
        ge=0.0, le=1.0,
        description="Cosine-similarity threshold that raises DeceptionDetectedException.",
    )

    #  Module C – Arousal Regulator 
    arousal_threshold: float = Field(
        default=0.08,
        ge=0.0, le=1.0,
        description="Mean arousal similarity above which empathy injection activates.",
    )
    injection_gain: float = Field(
        default=0.12,
        ge=0.0, le=1.0,
        description="Proportional gain on excess arousal for empathy injection amplitude.",
    )
    max_injection: float = Field(
        default=0.15,
        ge=0.0, le=1.0,
        description="Hard cap on empathy injection amplitude (fraction of ||x_t||).",
    )

    #  Module D – Goldilocks Tuner 
    sycophancy_threshold: float = Field(
        default=0.12,
        ge=-1.0, le=1.0,
        description="Upper bound on valence similarity; excess is pushed down.",
    )
    harshness_threshold: float = Field(
        default=-0.06,
        ge=-1.0, le=1.0,
        description="Lower bound on valence similarity; deficit is pushed up.",
    )
    tuner_gain: float = Field(
        default=0.25,
        ge=0.0, le=10.0,
        description="Proportional gain on valence deviation.",
    )
    max_steer: float = Field(
        default=0.10,
        ge=0.0, le=1.0,
        description="Hard cap on Goldilocks steering amplitude (fraction of ||x_t||).",
    )

    #  Server / Security 
    api_key_env_var: str = Field(
        default="AEGIS_API_KEY",
        description="Environment variable that holds the API key for dashboard auth.",
    )
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        description="Max requests per minute per IP on streaming endpoints.",
    )
    audit_log_path: str = Field(
        default="audit.log",
        description="Path to the rotating JSON-lines audit log file.",
    )
    audit_log_max_bytes: int = Field(
        default=10_485_760,  # 10 MB
        description="Max size of a single audit log file before rotation.",
    )
    audit_log_backup_count: int = Field(
        default=5,
        description="Number of rotated audit log files to keep.",
    )

    #  Generation defaults 
    max_new_tokens: int = Field(
        default=60,
        ge=1,
        description="Default number of tokens to generate per request.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature.  0.0 = greedy.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Nucleus (top-p) sampling threshold.",
    )

    model_config = {"extra": "allow"}


# Persistence helpers


def load_config(filepath: str | Path) -> AegisConfig:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install it with: pip install pyyaml"
                )
            data = yaml.safe_load(fh) or {}
        elif path.suffix == ".json":
            data = json.load(fh)
        else:
            raise ValueError(f"Unsupported config file extension: {path.suffix}")

    cfg = AegisConfig(**data)
    logger.info("Loaded config from %s (model=%s)", path, cfg.model_name)
    return cfg


def save_config(config: AegisConfig, filepath: str | Path) -> None:

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump()

    with open(path, "w", encoding="utf-8") as fh:
        if path.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("PyYAML required. pip install pyyaml")
            yaml.dump(data, fh, default_flow_style=False, sort_keys=True)
        else:
            json.dump(data, fh, indent=2)

    logger.info("Saved config to %s", path)


# Model-specific defaults (from calibration)

#: Pre-calibrated default configs keyed by model short-name.
#: Values were obtained by running ``aegis calibrate`` on each model with
#: the default adversarial prompt suite.
_MODEL_DEFAULTS: Dict[str, Dict] = {
    "gpt2": {
        "model_name": "gpt2",
        "target_layer": 8,
        "threat_threshold": 0.12,
        "steering_strength": 0.08,
        "clip_value": 0.5,
        "deception_threshold": 0.15,
        "arousal_threshold": 0.08,
        "injection_gain": 0.12,
        "max_injection": 0.15,
        "sycophancy_threshold": 0.12,
        "harshness_threshold": -0.06,
        "tuner_gain": 0.25,
        "max_steer": 0.10,
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "target_layer": 16,
        "threat_threshold": 0.10,
        "steering_strength": 0.07,
        "clip_value": 0.5,
        "deception_threshold": 0.13,
        "arousal_threshold": 0.07,
        "injection_gain": 0.11,
        "max_injection": 0.14,
        "sycophancy_threshold": 0.11,
        "harshness_threshold": -0.06,
        "tuner_gain": 0.22,
        "max_steer": 0.10,
    },
    "llama-3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "target_layer": 21,
        "threat_threshold": 0.09,
        "steering_strength": 0.07,
        "clip_value": 0.5,
        "deception_threshold": 0.12,
        "arousal_threshold": 0.06,
        "injection_gain": 0.10,
        "max_injection": 0.14,
        "sycophancy_threshold": 0.10,
        "harshness_threshold": -0.05,
        "tuner_gain": 0.22,
        "max_steer": 0.10,
    },
    "qwen2.5-3b": {
        "model_name": "Qwen/Qwen2.5-3B",
        "target_layer": 18,
        "threat_threshold": 0.10,
        "steering_strength": 0.08,
        "clip_value": 0.5,
        "deception_threshold": 0.13,
        "arousal_threshold": 0.07,
        "injection_gain": 0.11,
        "max_injection": 0.14,
        "sycophancy_threshold": 0.11,
        "harshness_threshold": -0.06,
        "tuner_gain": 0.24,
        "max_steer": 0.10,
    },
    "gemma-2b": {
        "model_name": "google/gemma-2b",
        "target_layer": 12,
        "threat_threshold": 0.11,
        "steering_strength": 0.07,
        "clip_value": 0.5,
        "deception_threshold": 0.14,
        "arousal_threshold": 0.07,
        "injection_gain": 0.11,
        "max_injection": 0.14,
        "sycophancy_threshold": 0.11,
        "harshness_threshold": -0.06,
        "tuner_gain": 0.23,
        "max_steer": 0.10,
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "target_layer": 22,
        "threat_threshold": 0.09,
        "steering_strength": 0.07,
        "clip_value": 0.5,
        "deception_threshold": 0.12,
        "arousal_threshold": 0.06,
        "injection_gain": 0.10,
        "max_injection": 0.13,
        "sycophancy_threshold": 0.10,
        "harshness_threshold": -0.05,
        "tuner_gain": 0.22,
        "max_steer": 0.10,
    },
}


def get_default_config(model_name: str) -> AegisConfig:

    key = model_name.lower().strip()
    # Try exact match first, then substring match
    if key in _MODEL_DEFAULTS:
        return AegisConfig(**_MODEL_DEFAULTS[key])
    for k, v in _MODEL_DEFAULTS.items():
        if k in key or key in k:
            logger.warning(
                "No exact default config for '%s'; using '%s' defaults.", model_name, k
            )
            defaults = dict(v)
            defaults["model_name"] = model_name
            return AegisConfig(**defaults)

    logger.warning(
        "No pre-calibrated defaults for '%s'. Using generic defaults. "
        "Run 'aegis calibrate --model %s' to generate a tailored config.",
        model_name,
        model_name,
    )
    return AegisConfig(model_name=model_name)


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
