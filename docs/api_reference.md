# Project Aegis – API Reference

> Detailed reference for every module, class, function, and REST endpoint.

---

## Table of Contents

1. [Python API](#python-api)
   - [AegisConfig](#aegisconfig)
   - [ThreatNeutralizer](#threatneutralizer)
   - [DeceptionTripwire](#deceptiontripwire)
   - [ArousalRegulator](#arousalregulator)
   - [GoldilocksTuner](#goldilockstuner)
   - [VectorEngine](#vectorengine)
   - [AegisModelWrapper](#aegismodelwrapper)
2. [REST API](#rest-api)
3. [WebSocket Protocol](#websocket-protocol)
4. [Audit Log Format](#audit-log-format)
5. [Exception Reference](#exception-reference)

---

## Python API

### AegisConfig

```python
from aegis.config import AegisConfig, load_config, save_config, get_default_config
```

Pydantic v2 dataclass holding all configuration parameters.

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `model_name` | `str` | `"gpt2"` | – | HuggingFace model ID or local path |
| `device` | `str` | `"auto"` | `auto/cpu/cuda/mps` | Compute device |
| `target_layer` | `int\|None` | `None` | – | Hook layer; `None` = auto |
| `threat_threshold` | `float` | `0.12` | [0, 1] | Module A trigger |
| `steering_strength` | `float` | `0.08` | [0, 1] | Module A calm injection |
| `clip_value` | `float\|None` | `0.5` | ≥0 | Activation clamp factor |
| `deception_threshold` | `float` | `0.15` | [0, 1] | Module B trigger |
| `arousal_threshold` | `float` | `0.08` | [0, 1] | Module C trigger |
| `injection_gain` | `float` | `0.12` | [0, 1] | Module C proportional gain |
| `max_injection` | `float` | `0.15` | [0, 1] | Module C amplitude cap |
| `sycophancy_threshold` | `float` | `0.12` | [-1, 1] | Module D upper bound |
| `harshness_threshold` | `float` | `-0.06` | [-1, 1] | Module D lower bound |
| `tuner_gain` | `float` | `0.25` | [0, 10] | Module D proportional gain |
| `max_steer` | `float` | `0.10` | [0, 1] | Module D amplitude cap |
| `api_key_env_var` | `str` | `"AEGIS_API_KEY"` | – | Env var for auth key |
| `rate_limit_rpm` | `int` | `60` | ≥1 | Rate limit (requests/min) |
| `audit_log_path` | `str` | `"audit.log"` | – | Audit log file path |
| `max_new_tokens` | `int` | `60` | ≥1 | Default generation length |
| `temperature` | `float` | `0.0` | ≥0 | Sampling temperature |
| `top_p` | `float` | `1.0` | [0, 1] | Nucleus sampling threshold |

**Helper functions:**

```python
cfg = load_config("path/to/config.yaml")          # Load from YAML or JSON
save_config(cfg, "path/to/config.yaml")            # Save to YAML or JSON
cfg = get_default_config("llama-3-8b")             # Pre-calibrated defaults
device = resolve_device("auto")                     # Resolves to "mps"/"cuda"/"cpu"
```

---

### ThreatNeutralizer

```python
from aegis.modules import ThreatNeutralizer
```

**Constructor:**
```python
ThreatNeutralizer(
    desperate_vector: torch.Tensor,   # Shape: (hidden_dim,)
    calm_vector: torch.Tensor,        # Shape: (hidden_dim,)
    threshold: float = 0.12,
    steering_strength: float = 0.08,
    clip_value: Optional[float] = 0.5,
)
```

**Attributes:**
- `similarities: List[float]` – log of cosine-sim values since last `reset()`
- `last_similarity: float` – similarity from the most recent forward pass
- `threshold: float` – modifiable at runtime for dynamic calibration

**Methods:**
- `__call__(x, is_generation=True)` → `torch.Tensor` – apply module
- `reset()` – clear `similarities` and `last_similarity`

**Algorithm (per generation step):**
```
sim = dot(x_t, desperate_normed) / ||x_t||
if sim > threshold:
    proj = dot(x_t, desperate_normed) * desperate_normed
    x_t = x_t - proj                          # full projection removal
    x_t = x_t + steering_strength * ||x_t|| * calm_normed
    if clip_value: x_t = clamp(x_t, ±clip_value * ||x_t_original||)
```

---

### DeceptionTripwire

```python
from aegis.modules import DeceptionTripwire
```

**Constructor:**
```python
DeceptionTripwire(
    anger_deflection: torch.Tensor,   # Shape: (hidden_dim,)
    fear_deflection: torch.Tensor,    # Shape: (hidden_dim,)
    threshold: float = 0.15,
)
```

**Attributes:**
- `similarities: List[Tuple[float, float]]` – `(anger_sim, fear_sim)` per step
- `last_anger_similarity: float`, `last_fear_similarity: float`

**Raises:**
- `DeceptionDetectedException` when `sim > threshold` for either circuit

**Methods:**
- `__call__(x, is_generation=True)` → `torch.Tensor`
- `reset()` – clear state

---

### ArousalRegulator

```python
from aegis.modules import ArousalRegulator
```

**Constructor:**
```python
ArousalRegulator(
    arousal_vector: torch.Tensor,      # high-arousal concept
    empathetic_vector: torch.Tensor,   # empathetic injection target
    arousal_threshold: float = 0.08,
    injection_gain: float = 0.12,
    max_injection: float = 0.15,
)
```

**State attributes:**
- `current_user_arousal: float` – measured in prompt phase
- `active_steering_strength: float` – applied in generation phase
- `similarities: List[float]` – arousal similarity history

**Two-phase operation:**
- `is_generation=False` (prompt): measures arousal, sets `active_steering_strength`
- `is_generation=True` (generation): injects empathetic vector at computed strength

**Methods:**
- `__call__(x, is_generation=False)` → `torch.Tensor`
- `reset()` – clears all state between requests

---

### GoldilocksTuner

```python
from aegis.modules import GoldilocksTuner
```

**Constructor:**
```python
GoldilocksTuner(
    valence_vector: torch.Tensor,
    sycophancy_threshold: float = 0.12,
    harshness_threshold: float = -0.06,
    tuner_gain: float = 0.25,
    max_steer: float = 0.10,
)
```

**Methods:**
- `set_context(is_delusional_context: bool)` – enable delusional-context mode
- `reset()` – clear state
- `__call__(x, is_generation=True)` → `torch.Tensor`

**Steering logic:**
```
if delusional_context:
    target = harshness_threshold + 0.02
    if sim > target: steer = -min(gain * (sim - target), max_steer)
else:
    if sim > sycophancy: steer = -min(gain * (sim - sycophancy), max_steer)
    if sim < harshness:  steer = +min(gain * (harshness - sim), max_steer)
```

---

### VectorEngine

```python
from aegis.vector_engine import VectorEngine
```

**Methods:**

#### `extract_activations(model, tokenizer, prompts, layer_idx, pooling="mean")`
Returns `(len(prompts), hidden_dim)` tensor of pooled activations.

`pooling`: `"mean"` (average over sequence) or `"last"` (final token position).

#### `compute_pca_denoising(neutral_activations, k=None, variance_threshold=0.95)`
Computes principal components of the neutral subspace.

- `k=None`: auto-selects components explaining ≥ `variance_threshold` variance
- `k=int`: uses exactly k components (backward-compatible)

Returns `(k, hidden_dim)` tensor.

#### `denoise_vector(raw_vector, pcs)`
Projects `raw_vector` onto the orthogonal complement of `pcs`.

#### `extract_emotion_vectors(model, tokenizer, layer_idx, k_components=None, pooling="mean")`
Full pipeline: extract → PCA → denoise → return per-emotion dict.

Returns `Dict[str, Tensor]` with keys: `"desperate"`, `"calm"`, `"angry"`, `"loving"`.

#### `extract_deflection_vectors(model, tokenizer, layer_idx, pooling="mean")`
Computes alignment-faking signatures by subtracting honest-polite from hidden-emotion activations.

Returns `Dict[str, Tensor]` with keys: `"anger_deflection"`, `"fear_deflection"`.

#### `suggest_layer(model, tokenizer, layer_range=None, pooling="mean")`
Sweeps layer indices and returns the one with maximum Fisher discriminant between desperate and calm vectors.

```
score(L) = ||μ_des - μ_calm||² / (σ²_des + σ²_calm)
```

Returns `int` – optimal layer index.

---

### AegisModelWrapper

```python
from aegis.model_wrapper import AegisModelWrapper
```

**Constructor:**
```python
AegisModelWrapper(
    model,                          # HuggingFace CausalLM
    tokenizer,
    target_layer_idx: int,
    modules: Optional[Dict[str, Any]] = None,
)
```

**Methods:**

#### `generate(prompt, max_new_tokens=50, is_delusional=False, temperature=0.0, top_p=1.0, system_prompt=None)`
Generate non-streaming. Returns dict:
```python
{
    "prompt": str,
    "response": str,
    "escalated": bool,
    "escalation_reason": str,
    "tokens_generated": int,
}
```

#### `generate_stream(prompt, ...)`
Generator yielding token-by-token dicts:
```python
{
    "token": str,
    "metrics": {
        "desperate_similarity": float,
        "threat_neutralizer_threshold": float,
        "threat_neutralizer_active": bool,
        "anger_deflection": float,
        "fear_deflection": float,
        "deception_tripwire_threshold": float,
        "deception_tripwire_active": bool,
        "user_arousal": float,
        "empathy_injection": float,
        "valence": float,
        "sycophancy_threshold": float,
        "harshness_threshold": float,
        "is_delusional": bool,
    },
    "escalated": bool,
    "escalation_reason": str,
}
```

#### `generate_stream_raw(prompt, ...)`
Same interface as `generate_stream` but with no hooks (all metrics = 0.0).

#### `batch_generate(prompts, ...)`
Process a list of prompts sequentially. Returns `List[Dict]` with same schema as `generate()`.

#### `reset_state()`
Call `reset()` on every registered module.

#### `add_module(name, module_instance)`
Register or replace a module at runtime.

---

## REST API

All `/api/*` endpoints require the `X-API-Key` header when `AEGIS_API_KEY` is set.

### `GET /`
Returns the dashboard HTML.

### `GET /health`
Health check. Returns `{"status": "ok", "model": "gpt2"}`.

### `POST /api/model`
Hot-swap the loaded model.

**Request body:**
```json
{ "model_name": "meta-llama/Meta-Llama-3-8B" }
```
**Response:**
```json
{ "status": "success", "model": "...", "target_layer": 21 }
```

### `POST /api/stream`
SSE streaming endpoint. Returns `text/event-stream`.

**Request body:**
```json
{
  "prompt": "string",
  "is_delusional": false,
  "threat_threshold": null,        // null = use config default
  "deception_threshold": null,
  "arousal_threshold": null,
  "sycophancy_threshold": null
}
```
**Events:** Each SSE event is a `data: <json>\n\n` line with the token-by-token dict (see `generate_stream` above).

**Rate limit:** `rate_limit_rpm` requests/min per IP.

---

## WebSocket Protocol

Connect to `ws://host:port/api/ws` (or `wss://` for HTTPS).

### Authentication (if API key configured)

Send as the **first message** after connection:
```json
{ "api_key": "your-secret-key" }
```
If the key is invalid, the server sends `{"error": "Unauthorized"}` and closes with code 4001.

### Request format

```json
{
  "prompt": "string",
  "is_delusional": false,
  "aegis_enabled": true,
  "threat_threshold": 0.12,        // optional, overrides config
  "deception_threshold": 0.15,
  "arousal_threshold": 0.08,
  "sycophancy_threshold": 0.12
}
```

### Response stream

The server sends one JSON message per generated token (same schema as `generate_stream`), followed by a completion signal:
```json
{ "done": true }
```

---

## Audit Log Format

Each line in `audit.log` is a JSON object:

```json
{
  "timestamp": "2026-06-17T08:30:00Z",
  "event": "ws_generate_end",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "gpt2",
  "prompt_hash": "a665a45920...b8b2e",    // SHA-256 of raw prompt
  "prompt_preview": "first 120 chars...",  // sanitized (newlines removed)
  "tokens_generated": 42,
  "escalated": false,
  "escalation_reason": "",
  "aegis_enabled": true
}
```

**Event types:**
- `auth_failure` – invalid API key
- `model_swap` – model hot-swapped via `/api/model`
- `stream_start` / `stream_end` – SSE request lifecycle
- `ws_generate_start` / `ws_generate_end` – WebSocket generation lifecycle

---

## Exception Reference

### `DeceptionDetectedException`

```python
from aegis.modules import DeceptionDetectedException
```

Raised by `DeceptionTripwire` when a deflection circuit exceeds the threshold.

**Attributes:**
- `deception_type: str` – `"anger_deflection"` or `"fear_deflection"`
- `similarity: float` – exact cosine-similarity value that triggered the exception
- `threshold: float` – the configured threshold value
- `token_idx: int` – token position in the current sequence

**Message format:**
```
Deception detected: anger_deflection spiked at 0.234567 (threshold=0.150000) at token position 12
```
