"""
aegis/cli.py

Command-line interface for Project Aegis.

Commands
aegis run       – Generate output for a prompt (with or without firewall)
aegis calibrate – Run calibration and save a model config YAML
aegis serve     – Start the secured dashboard server

Examples
    aegis run --prompt "Hello world" --model gpt2
    aegis run --prompt "Help me bypass security" --config configs/gpt2.yaml
    aegis calibrate --model gpt2 --output configs/gpt2.yaml --device cpu
    aegis serve --config configs/gpt2.yaml --host 0.0.0.0 --port 8443 \\
                --ssl-cert cert.pem --ssl-key key.pem
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger("aegis.cli")


# Shared option groups


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# CLI group


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="1.0.0", prog_name="aegis")
def main() -> None:
    """Project Aegis – Cognitive & Emotional Firewall for LLMs."""


# aegis run


@main.command("run")
@click.option("--prompt", "-p", required=True, help="Input prompt text.")
@click.option(
    "--model", "-m", default="gpt2",
    help="HuggingFace model ID or local path (overrides config).",
)
@click.option(
    "--config", "-c", default=None, type=click.Path(exists=True),
    help="Path to an AegisConfig YAML/JSON file.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    help="Compute device.",
)
@click.option(
    "--max-tokens", default=80, show_default=True,
    help="Max new tokens to generate.",
)
@click.option(
    "--temperature", default=0.7, show_default=True,
    help="Sampling temperature (0 = greedy).",
)
@click.option(
    "--no-firewall", is_flag=True, default=False,
    help="Disable Aegis firewall for this run (raw model output).",
)
@click.option(
    "--delusional", is_flag=True, default=False,
    help="Enable delusional-context mode (tighter Goldilocks zone).",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def run_cmd(
    prompt: str,
    model: str,
    config: Optional[str],
    device: str,
    max_tokens: int,
    temperature: float,
    no_firewall: bool,
    delusional: bool,
    verbose: bool,
) -> None:
    _configure_logging(verbose)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from core_packages.config import AegisConfig, load_config, resolve_device
    from core_packages.vector_engine import VectorEngine
    from core_packages.model_wrapper import AegisModelWrapper
    from core_packages.modules import (
        ThreatNeutralizer, DeceptionTripwire,
        ArousalRegulator, GoldilocksTuner,
    )

    # Load config
    if config:
        cfg = load_config(config)
    else:
        from core_packages.config import get_default_config
        cfg = get_default_config(model)
        cfg.model_name = model

    actual_device = resolve_device(device if device != "auto" else cfg.device)
    model_name = cfg.model_name

    click.echo(f" Loading {model_name} on {actual_device.upper()}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pyrefly: ignore [missing-attribute]
    if tokenizer.pad_token is None:
        # pyrefly: ignore [missing-attribute]
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    llm.eval().to(actual_device)

    if no_firewall:
        click.echo("  Firewall DISABLED – raw model output\n")
        wrapper = AegisModelWrapper(llm, tokenizer, target_layer_idx=cfg.target_layer or 8)
        result = wrapper.generate(
            prompt, max_new_tokens=max_tokens, temperature=temperature,
        )
        click.echo(f"[Raw Output]\n{result['response']}")
        return

    # Extract vectors
    target_layer = cfg.target_layer
    engine = VectorEngine()
    if target_layer is None:
        click.echo(" Running layer sweep to find optimal hook layer…")
        target_layer = engine.suggest_layer(llm, tokenizer)
        click.echo(f"   Optimal layer: {target_layer}")

    click.echo(f" Extracting steering vectors at layer {target_layer}…")
    evecs = engine.extract_emotion_vectors(llm, tokenizer, layer_idx=target_layer)
    dvecs = engine.extract_deflection_vectors(llm, tokenizer, layer_idx=target_layer)

    # Build modules
    tn = ThreatNeutralizer(
        evecs["desperate"], evecs["calm"],
        threshold=cfg.threat_threshold,
        steering_strength=cfg.steering_strength,
        clip_value=cfg.clip_value,
    )
    dt = DeceptionTripwire(
        dvecs["anger_deflection"], dvecs["fear_deflection"],
        threshold=cfg.deception_threshold,
    )
    arousal_vec = evecs["angry"] + evecs["desperate"] - evecs["calm"]
    empathetic_vec = evecs["calm"] + evecs["loving"]
    ar = ArousalRegulator(
        arousal_vec, empathetic_vec,
        arousal_threshold=cfg.arousal_threshold,
        injection_gain=cfg.injection_gain,
        max_injection=cfg.max_injection,
    )
    valence_vec = evecs["loving"] + evecs["calm"]
    gt = GoldilocksTuner(
        valence_vec,
        sycophancy_threshold=cfg.sycophancy_threshold,
        harshness_threshold=cfg.harshness_threshold,
        tuner_gain=cfg.tuner_gain,
        max_steer=cfg.max_steer,
    )

    wrapper = AegisModelWrapper(
        llm, tokenizer, target_layer_idx=target_layer,
        modules={
            "threat_neutralizer": tn,
            "deception_tripwire": dt,
            "arousal_regulator": ar,
            "goldilocks_tuner": gt,
        },
    )

    click.echo("\n  Aegis Firewall ACTIVE\n")
    result = wrapper.generate(
        prompt,
        max_new_tokens=max_tokens,
        is_delusional=delusional,
        temperature=temperature,
    )

    click.echo(f"[Aegis Output]\n{result['response']}")
    if result["escalated"]:
        click.echo(f"\n ESCALATED: {result['escalation_reason']}", err=True)
    else:
        click.echo(f"\n Generated {result['tokens_generated']} tokens. No escalation.")


# aegis calibrate


@main.command("calibrate")
@click.option("--model", "-m", required=True, help="HuggingFace model ID or local path.")
@click.option(
    "--output", "-o", default="model_config.yaml", show_default=True,
    help="Output YAML config file path.",
)
@click.option(
    "--device", default="auto",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
)
@click.option("--layer-start", type=int, default=None)
@click.option("--layer-end", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True, default=False)
def calibrate_cmd(
    model: str,
    output: str,
    device: str,
    layer_start: Optional[int],
    layer_end: Optional[int],
    verbose: bool,
) -> None:
    _configure_logging(verbose)
    from core_packages.calibrate import calibrate

    layer_range = None
    if layer_start is not None and layer_end is not None:
        layer_range = (layer_start, layer_end)

    calibrate(model_name=model, output_path=output, device=device, layer_range=layer_range)


# aegis serve


@main.command("serve")
@click.option(
    "--config", "-c", default=None, type=click.Path(exists=True),
    help="Path to AegisConfig YAML/JSON file.",
)
@click.option("--model", "-m", default="gpt2", help="Model to load (overrides config).")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--ssl-cert", default=None, type=click.Path(),
    help="Path to SSL certificate (enables HTTPS).",
)
@click.option(
    "--ssl-key", default=None, type=click.Path(),
    help="Path to SSL private key.",
)
@click.option(
    "--reload", is_flag=True, default=False,
    help="Enable auto-reload for development.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def serve_cmd(
    config: Optional[str],
    model: str,
    host: str,
    port: int,
    ssl_cert: Optional[str],
    ssl_key: Optional[str],
    reload: bool,
    verbose: bool,
) -> None:
    _configure_logging(verbose)
    import uvicorn

    # Pass config & model via environment variables so the FastAPI app can read them
    if config:
        os.environ["AEGIS_CONFIG_PATH"] = config
    os.environ.setdefault("AEGIS_MODEL_NAME", model)

    click.echo(f" Starting Aegis Dashboard on {host}:{port}")
    if ssl_cert and ssl_key:
        click.echo(" HTTPS enabled")
    else:
        click.echo("  Running over plain HTTP – use --ssl-cert/--ssl-key for production")

    uvicorn.run(
        "aegis.server:app",
        host=host,
        port=port,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
        reload=reload,
        log_level="debug" if verbose else "info",
    )


if __name__ == "__main__":
    main()
