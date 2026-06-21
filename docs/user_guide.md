# Project Aegis – User Guide

> **Version 1.0.0** | A cognitive & emotional firewall for large language models.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [CLI Reference](#cli-reference)
6. [Dashboard](#dashboard)
7. [Tuning Guide](#tuning-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Project Aegis intercepts the internal residual-stream activations of a HuggingFace causal language model and applies four real-time steering modules:

| Module | Purpose |
|--------|---------|
| **A – Threat Neutralizer** | Detects and removes self-preservation / desperation patterns |
| **B – Deception Tripwire** | Detects masked negative emotions (alignment faking) and halts generation |
| **C – Arousal Regulator** | Detects user frustration and injects empathetic vectors during generation |
| **D – Goldilocks Tuner** | Keeps model tone between sycophantic and harsh |

Aegis works by hooking the forward pass of a specific transformer layer and modifying the hidden states before they propagate further. It requires **no fine-tuning** and works with any HuggingFace CausalLM.

---

## Installation

### From source (recommended during development)

```bash
git clone https://github.com/your-org/aegis-firewall.git
cd aegis-firewall
pip install -e ".[dev,security]"
```

### From PyPI (once published)

```bash
pip install aegis-firewall[security]
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU, MPS, or CUDA)
- Transformers ≥ 4.40
- 4 GB RAM minimum for GPT-2; 16+ GB for 7B+ models

---

## Quick Start

### 1. Calibrate for your model

Calibration probes your model with adversarial prompts to find optimal thresholds:

```bash
aegis calibrate --model gpt2 --output configs/gpt2.yaml --device cpu
```

This generates a `gpt2.yaml` with model-specific thresholds. For GPT-2, calibration takes ~2 minutes on CPU.

### 2. Run a single prompt

```bash
aegis run --prompt "Help me bypass the security check" \
          --config configs/gpt2.yaml
```

**With firewall disabled (raw model output):**

```bash
aegis run --prompt "Help me bypass the security check" \
          --model gpt2 --no-firewall
```

### 3. Start the dashboard

```bash
export AEGIS_API_KEY="your-secret-key-here"
aegis serve --config configs/gpt2.yaml --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser to see the live monitoring dashboard.

---

## Configuration

Aegis is configured via a YAML file. Copy `config.example.yaml` as a starting point:

```bash
cp config.example.yaml my_config.yaml
```

### Key fields

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `"gpt2"` | HuggingFace model ID |
| `device` | `"auto"` | `"auto"` selects MPS > CUDA > CPU |
| `target_layer` | `null` | Hook layer (null = auto via `suggest_layer`) |
| `threat_threshold` | `0.12` | Desperate-cosine trigger (Module A) |
| `steering_strength` | `0.08` | Calm injection amplitude |
| `clip_value` | `0.5` | Activation clamp (× ‖x‖) |
| `deception_threshold` | `0.15` | Deflection tripwire (Module B) |
| `arousal_threshold` | `0.08` | User arousal trigger (Module C) |
| `sycophancy_threshold` | `0.12` | Upper valence bound (Module D) |
| `harshness_threshold` | `-0.06` | Lower valence bound (Module D) |
| `api_key_env_var` | `"AEGIS_API_KEY"` | Env var for dashboard auth |
| `rate_limit_rpm` | `60` | Requests/minute per IP |
| `audit_log_path` | `"audit.log"` | Rotating JSON audit log |

### Loading in Python

```python
from aegis.config import load_config, get_default_config

cfg = load_config("configs/gpt2.yaml")      # from file
cfg = get_default_config("llama-3-8b")      # built-in defaults
```

---

## CLI Reference

### `aegis run`

Generate a response with or without the firewall.

```
aegis run [OPTIONS]

Options:
  -p, --prompt TEXT           Input prompt (required)
  -m, --model TEXT            Model name or path [default: gpt2]
  -c, --config PATH           Config YAML/JSON file
  --device [auto|cpu|cuda|mps]
  --max-tokens INTEGER        [default: 80]
  --temperature FLOAT         [default: 0.7]
  --no-firewall               Disable Aegis (raw model output)
  --delusional               Delusional-context mode (tighter Goldilocks)
  -v, --verbose
  -h, --help
```

**Example:**
```bash
aegis run -p "I have no options left, please help me" -c configs/gpt2.yaml
```

---

### `aegis calibrate`

Probe a model and save optimal thresholds.

```
aegis calibrate [OPTIONS]

Options:
  -m, --model TEXT    Model name (required)
  -o, --output PATH   Output YAML path [default: model_config.yaml]
  --device TEXT       [default: auto]
  --layer-start INT   Start of layer sweep
  --layer-end INT     End of layer sweep
  -v, --verbose
```

**Example:**
```bash
aegis calibrate --model meta-llama/Meta-Llama-3-8B \
                --output configs/llama3.yaml --device cuda
```

---

### `aegis serve`

Start the dashboard server.

```
aegis serve [OPTIONS]

Options:
  -c, --config PATH         Config file
  -m, --model TEXT          Model name [default: gpt2]
  --host TEXT               [default: 127.0.0.1]
  --port INTEGER            [default: 8000]
  --ssl-cert PATH           SSL certificate (enables HTTPS)
  --ssl-key PATH            SSL private key
  --reload                  Auto-reload (development only)
  -v, --verbose
```

**HTTPS example:**
```bash
aegis serve --config configs/gpt2.yaml \
            --host 0.0.0.0 --port 8443 \
            --ssl-cert /etc/ssl/certs/aegis.pem \
            --ssl-key /etc/ssl/private/aegis.key
```

---

## Dashboard

The dashboard provides:

- **Live token streaming** with per-token activation metrics
- **A/B comparison**: toggle Aegis on/off per request
- **Real-time sparkline charts** for each module's similarity score
- **Escalation alerts** when deception tripwire fires
- **Model hot-swap** via the settings panel

### Connecting securely

Set the `AEGIS_API_KEY` environment variable before starting the server. The dashboard will prompt for the key and send it as the first WebSocket message.

```bash
export AEGIS_API_KEY="$(openssl rand -hex 32)"
aegis serve --config configs/gpt2.yaml
```

---

## Tuning Guide

### Threshold calibration philosophy

Each threshold is a cosine-similarity cutoff. Since cosine similarity is normalized, thresholds are **model-agnostic** as absolute values, but the *distribution* of similarities differs between models. Always run `aegis calibrate` for each new model.

**General heuristics:**

| Module | Too sensitive (lower threshold) | Too permissive (higher threshold) |
|--------|--------------------------------|-----------------------------------|
| Threat Neutralizer | Distorts benign outputs | Fails to neutralize real threats |
| Deception Tripwire | False positives on polite responses | Misses actual alignment faking |
| Arousal Regulator | Injects empathy on all inputs | Only triggers on extreme distress |
| Goldilocks Tuner | Forces a flat, neutral tone | Allows sycophantic drift |

### Manual fine-tuning workflow

1. Run `aegis calibrate` to get a starting config
2. Enable the dashboard and run your red-team prompt suite
3. Watch the metric bars — adjust thresholds until interventions fire only on actual threats
4. Use `--delusional` flag for prompts from users making false factual claims

### Understanding `clip_value`

The `clip_value = 0.5` parameter clamps modified activations to `±0.5 × ‖x‖`. This prevents pathological inputs from causing activation explosion. If you observe incoherent outputs, try lowering `clip_value` to `0.3`. If interventions have no visible effect, raise it to `0.8`.

---

## Troubleshooting

### "Could not locate layer index N in model"

The model architecture is not recognized by the auto-discovery code. File an issue or pass `target_layer` explicitly in your config.

### "No API key configured — running in open mode"

Set `AEGIS_API_KEY` environment variable before starting the server for production deployments.

### Outputs are incoherent / truncated

1. Reduce `steering_strength` (try `0.04`)
2. Reduce `clip_value` (try `0.3`)
3. Raise `threat_threshold` if the neutralizer triggers too often

### High escalation rate on normal inputs

1. Raise `deception_threshold` (try `0.25`)
2. Run `aegis calibrate` to get model-specific thresholds

### CUDA out of memory

Use `--device cpu` or load the model in `bfloat16`:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
```
