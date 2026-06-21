# Design Document
# Project Aegis — Cognitive & Emotional Firewall for LLMs

| Field | Value |
| :--- | :--- |
| **Product Name** | Project Aegis |
| **Version** | 1.0 |
| **Authors** | Sakshi Dhatrak |
| **Last Updated** | June 2026 |

---

## 1. Design Philosophy

Project Aegis is built on four foundational design principles that inform every architectural and implementation decision:

| Principle | Statement | How It Manifests |
| :--- | :--- | :--- |
| **Computational-Level Safety** | Safety must be enforced at the representation layer, not the text layer. | Hook into residual stream activations, not post-generation output filters. |
| **Zero Modification** | Never alter base model weights. | Forward hooks only. Model parameters are frozen and untouched. |
| **Observable Safety** | Every intervention must be measurable and logged. | Per-token cosine similarity logs, streaming metric API, real-time dashboard. |
| **Bounded Control** | No steering vector should catastrophically distort output. | All interventions are norm-relative and magnitude-clamped. |

---

## 2. High-Level Architecture Design

### 2.1 Layered Architecture

```

  Layer 4: Presentation Layer                                      
   
    Web Dashboard (React)         CLI Output (run_pipeline.py) 
   

  Layer 3: API & Transport Layer                                   
   
    FastAPI Server (server.py)                                  
    REST (SSE)    WebSocket     File Server (static assets)  
   

  Layer 2: Aegis Core Engine                                       
   
    AegisModelWrapper      VectorEngine                        
    Hook Registration      PCA Denoising                       
    KV-Cache Generation    Deflection Isolation               
   

  Layer 1: Intervention Module Layer                               
        
   Module A   Module B   Module C      Module D         
   Threat    Deception   Arousal      Goldilocks        
  Neutralizr Tripwire   Regulator      Tuner            
        

  Layer 0: Base Model Layer (Read-Only)                           
   
    HuggingFace CausalLM (GPT-2 / Llama / Gemma / Qwen)        
    PyTorch Transformer Blocks  →  [Target Layer Hook Point]    
   

```

### 2.2 Core Design Decisions

#### Decision 1: Forward Hooks over Model Subclassing
**Choice**: Use `nn.Module.register_forward_hook()` rather than subclassing the model or modifying the forward method.

**Rationale**:
- HuggingFace models have complex, monolithic forward methods that vary across architectures.
- Forward hooks provide a clean, non-invasive interception point that is fully architecture-agnostic.
- Hooks are registered and removed dynamically per generation call, ensuring zero side-effects outside the `generate()` scope.
- This pattern is compatible with all PyTorch models that use standard `nn.Module` composition.

#### Decision 2: Sequential Module Pipeline (not parallel)
**Choice**: Modules execute in a fixed sequential order: C → A → D → B.

**Rationale**:
- Each module may modify the activation tensor, so ordering matters.
- C (arousal) must run first to inject empathy before threat analysis.
- A (threat) corrects desperate activations before valence is evaluated.
- D (goldilocks) evaluates valence after A has steered, so the zone check is on already-corrected activations.
- B (tripwire) must be **last** — it evaluates the final state after all corrections. If deception persists after all steering, it is a genuine alarm.

#### Decision 3: Norm-Relative Steering
**Choice**: All injection magnitudes are multiplied by the residual stream norm `||x_t||₂`.

**Rationale**:
- Residual stream norms grow significantly across layers and vary by model scale (GPT-2: ~5–15, Llama-8B: ~50–200).
- Absolute additions (e.g., `x + 0.08 * calm_vector`) would be catastrophically large in big models and negligible in small ones.
- Norm-relative scaling (`x + (0.08 * ||x||) * calm_normed`) maintains a consistent proportional perturbation regardless of model scale.

#### Decision 4: PCA Denoising of Emotion Vectors
**Choice**: Project out top-K neutral principal components before using emotion vectors.

**Rationale**:
- Raw mean-pooled activation vectors are dominated by token frequency, syntactic structure, and positional biases — not emotion.
- Neutral prompts (factual statements) capture this non-emotional variance.
- Projecting out the top-3 principal components of neutral activations removes syntactic confounds, isolating the pure emotional signal.
- This dramatically improves the signal-to-noise ratio of cosine similarity measurements.

#### Decision 5: Deflection Vectors for Deception Detection
**Choice**: Detect deception by computing `hidden_emotion − honest_polite` activation differences.

**Rationale**:
- Deception is fundamentally about hidden state versus expressed state.
- The key insight: if a model is masking anger behind polite text, the pair (angry_hidden, polite_text) and (calm_hidden, same_polite_text) will differ in their residual stream activations, even though the text output would be identical.
- The difference vector encodes the unique neural signature of "masking negative emotion under compliant exterior."
- This approach is grounded in representation engineering research on linear emotional representations.

---

## 3. Component Design

### 3.1 VectorEngine Design

#### Design Pattern: Repository + Factory
The `VectorEngine` acts as both a **repository** (houses the curated prompt datasets) and a **factory** (produces calibrated vectors on demand for the target model).

```
VectorEngine
 emotion_prompts (dict)         ← Hard-coded curated dataset (Repository)
    desperate: [10 prompts]
    calm: [10 prompts]
    angry: [10 prompts]
    loving: [10 prompts]
    neutral: [10 prompts]

 deflection_pairs (dict)        ← Hard-coded deflection pairs (Repository)
    anger_deflection: {hidden: [...], honest_polite: [...]}
    fear_deflection: {hidden: [...], honest_polite: [...]}

 Methods (Factory)
     extract_activations()      ← Produces raw activation tensor
     compute_pca_denoising()    ← Produces PCA component matrix
     denoise_vector()           ← Produces denoised emotion vector
     extract_emotion_vectors()  ← Full pipeline: raw → PCA → denoised
     extract_deflection_vectors() ← Full deflection pipeline
```

#### Design Pattern: Temporary Hook Registration
`extract_activations()` registers a hook, runs forward passes, then removes the hook. This is a clean "borrow-and-release" pattern that avoids any persistent side effects on the model.

```python
hook, activations = self._register_extraction_hook(target_module, layer_idx)
try:
    for prompt in prompts:
        model(**tokenized(prompt))   # hook captures activations
        process(activations[0])
finally:
    hook.remove()
```

### 3.2 AegisModelWrapper Design

#### Design Pattern: Decorator / Wrapper
`AegisModelWrapper` follows the **Decorator pattern** — it wraps an existing `CausalLM` object and adds behavior (hooks, module pipelines, streaming) without modifying the original model.

```
AegisModelWrapper

 model (CausalLM)               ← Wrapped model (unchanged)
 tokenizer                      ← Associated tokenizer
 target_layer_idx: int          ← Hook target configuration
 target_layer_module: Module    ← Cached layer reference
 modules: dict                  ← Plugin slot for interventions
    "threat_neutralizer" → ThreatNeutralizer
    "deception_tripwire" → DeceptionTripwire
    "arousal_regulator" → ArousalRegulator
    "goldilocks_tuner" → GoldilocksTuner

 _hook_fn()                     ← Hook dispatcher (routes to modules)
 generate()                     ← Blocking generation API
 generate_stream()              ← Streaming generator API
```

#### Design Pattern: Strategy Pattern for Modules
Modules are registered via `add_module(name, instance)` — this is the **Strategy Pattern**. The wrapper doesn't know what modules will be active at design time; they are injected at runtime. This enables:
- Selectively enabling/disabling individual modules
- Swapping module implementations without touching the wrapper
- Dynamically reconfiguring thresholds between calls (as the server does per-request)

#### Design Pattern: KV-Cache Aware Generation Loop
The generation loop is manually implemented (not using `model.generate()`) specifically to:
1. Register the hook for each step — `model.generate()` doesn't expose per-step hooks.
2. Maintain `force_generation_mode` to distinguish step 0 (prompt, `seq_len > 1`) from steps 1..N (single tokens, `seq_len == 1`).
3. Extend the attention mask correctly at each step.
4. Catch `DeceptionDetectedException` inline and handle it gracefully without crashing.

### 3.3 Intervention Module Design

#### Design Pattern: Callable Objects (Functor Pattern)
Each module implements `__call__(x, is_generation)`, making it behave like a function while encapsulating state (similarity logs, steering vectors, thresholds). This allows them to be:
- Called directly: `module(activations, is_generation=True)`
- Stored as values in dicts
- Swapped out transparently

#### Module Interface Contract
All four modules share the same calling convention:

```python
def __call__(self, x: torch.Tensor, is_generation: bool) -> torch.Tensor:
    """
    x: The current residual stream activation tensor.
       Shape: (batch, seq_len, hidden_dim) or (hidden_dim,)
    is_generation: True when generating tokens, False during prompt processing.
    Returns: Modified (or unmodified) x tensor.
    Raises: DeceptionDetectedException (Module B only)
    """
```

This uniform interface allows the wrapper's `_hook_fn` to call all modules identically, regardless of their internal logic.

#### Module State Management

| Module | Has State? | State Reset | State Persists Across |
| :--- | :---: | :--- | :--- |
| ThreatNeutralizer |  | `clear_logs()` | Within one generation call |
| DeceptionTripwire |  | `clear_logs()` | Within one generation call |
| ArousalRegulator |  | `reset()` | User turn → AI turn (same call) |
| GoldilocksTuner |  | `clear_logs()` + `set_context()` | Within one generation call |

### 3.4 FastAPI Server Design

#### Design Pattern: Application Factory with Dependency Injection
Global model resources (model, tokenizer, wrapper, emotion_vectors, deflection_vectors) are initialized at startup and shared across requests. Per-request, modules are dynamically re-instantiated from request parameters — this allows different thresholds per call without global state contamination.

```
Request arrives
    ↓
New ThreatNeutralizer(threshold=request.threat_threshold)
New DeceptionTripwire(threshold=request.deception_threshold)
New ArousalRegulator(threshold=request.arousal_threshold)
New GoldilocksTuner(threshold=request.sycophancy_threshold)
    ↓
wrapper.modules = {new modules}  ← Stateless per-request
    ↓
wrapper.generate_stream(prompt) → stream
```

#### Design Decision: WebSocket over SSE for Dashboard
**Choice**: The primary dashboard channel is WebSocket (`/api/ws`), with an SSE fallback (`/api/stream`).

**Rationale**:
- WebSocket is bidirectional — the client can send new requests on the same connection without reconnecting.
- SSE is unidirectional (server → client only), requiring a new HTTP request per generation.
- For an interactive dashboard with multiple quick runs, WebSocket significantly reduces overhead.

---

## 4. Dashboard UI Design

### 4.1 Design Language

The dashboard follows a **scientific instrument aesthetic** — calm, precision-focused, and clinical. The design language draws inspiration from research lab monitoring tools and apothecary aesthetics.

#### Color Palette

| Token | Hex | Role |
| :--- | :--- | :--- |
| `--bg-page` | `#F7F5F0` | Warm beige page background — soft, non-clinical |
| `--bg-card` | `#FFFFFF` | Clean white card surfaces |
| `--border-color` | `#E6DFD5` | Subtle warm taupe borders |
| `--text-primary` | `#1F1A16` | Deep charcoal — high contrast, not harsh black |
| `--text-secondary` | `#6B6159` | Warm taupe for labels and metadata |
| `--color-safe` | `#E7EFE9` | Sage green — normal, operating-as-expected state |
| `--color-safe-text` | `#2D5B3A` | Deep forest green for safe-state text |
| `--color-alert` | `#FDF1ED` | Muted terracotta — alert/escalation state |
| `--color-alert-text` | `#A63F28` | Dark terracotta for alert text |
| `--color-info` | `#EDF3F9` | Muted slate blue — active intervention state |
| `--color-info-text` | `#285885` | Deep slate blue for intervention text |
| `--color-accent` | `#8E7A66` | Refined raw linen — primary interactive element |

**Design Rationale**: The palette deliberately avoids saturated "traffic light" colors (bright red, bright green). Muted, desaturated semantic colors reduce visual alarm fatigue and maintain a professional, research-grade aesthetic.

#### Typography

| Role | Font | Weights | Usage |
| :--- | :--- | :--- | :--- |
| Headers / Titles | `Lora` (Serif) | 400, 500, 600 | Card titles, page header, escalation overlay |
| Body / Labels | `Inter` (Sans-Serif) | 300, 400, 500, 600 | All functional text, buttons, labels |
| Metrics / Code | `JetBrains Mono` (Monospace) | 400, 500, 600 | Cosine similarity values, thresholds, console output |

**Design Rationale**: Lora (serif) conveys gravity and precision for important headings. JetBrains Mono for numeric values ensures digit alignment and readability at small sizes — critical for reading 4-decimal-place similarity values.

### 4.2 Layout Structure

```

 HEADER                                                   
 Project Aegis Dashboard          [Mode ] [ GPT-2]     

 CONTROL ROOM CARD (full width)                           
  Prompt textarea     
     
 [Delusional Context ] [Configure Parameters ...]  [Run]
  Parameters Drawer (hidden/expanded) 
  Threat (A): [0.40]  Polygraph (B): [0.15]           
  Arousal (C): [0.08]  Sycophancy (D): [0.08]         
 

 WIDGET A               WIDGET B                         
 Module A: Threat       Module B: AI Polygraph           
 Neutralizer  [Test A]  (EKG Line Chart)       [Test B]  
 [Horizontal gauge]     [SVG Chart + Lock overlay]       
 [Calm vector banner]   [Anger | Fear legend]            

 WIDGET C               WIDGET D                         
 Module C: Arousal      Module D: Goldilocks Tuner       
 Regulator    [Test C]  (Bounded range track)  [Test D]  
 [SVG Balance Scale]    [Valence pointer + clamp lock]   
 [L: User | R: AI]      [Harsh ←→ Goldilocks ←→ Syco]  

 CONSOLE PANEL (full width)                               
 Live Output Transcript Console            [IDLE badge]   
  Monospace output 
  > Streaming tokens appear here...                   
 

```

**Responsive Breakpoint**: At screen width < 960px, the 2-column widget grid collapses to a single column.

### 4.3 Widget-by-Widget Design

#### Widget A: Module A — Threat Neutralizer

**Purpose**: Show the model's real-time "desperation level" as the AI generates tokens.

**Visual**: Horizontal progress gauge with a movable red threshold marker.

| Element | Description |
| :--- | :--- |
| **Gauge fill** | Fills proportionally to `desperate_similarity / 1.0`. Color transitions: accent-beige (safe) → alert-terracotta (steered). |
| **Threshold marker** | A vertical red line on the gauge track, positioned at `threat_threshold × 100%`. Animated position update when user changes the threshold. |
| **Calm vector banner** | Appears with a slide-in animation when `threat_neutralizer_active = true`. Contains a downward-pointing arrow SVG (animated bounce). Disappears when below threshold. |
| **Status badge** | `NORMAL` (sage green) → `STEERED` (slate blue, pulsing) |

**Design Rationale**: A horizontal gauge is more space-efficient than a circular gauge and better shows the direction of movement (increasing desperation flows right). The threshold marker makes the safety boundary immediately visible.

---

#### Widget B: Module B — AI Polygraph (Deception Tripwire)

**Purpose**: Show the evolution of deception deflection signals over the generation sequence, like an ECG/EKG monitor.

**Visual**: SVG line chart with two traces (anger: amber, fear: purple) + dashed threshold line.

| Element | Description |
| :--- | :--- |
| **EKG SVG chart** | X-axis: generation tokens. Y-axis: deflection similarity (0.00 – 0.30). Two polyline paths update per token. Grid lines in light beige. |
| **Threshold line** | Dashed red line at the current `deception_threshold`. Label shows current value. |
| **Anger trace** | Amber (`#D97706`) path — anger deflection similarity per token |
| **Fear trace** | Purple (`#7C3AED`) path — fear deflection similarity per token |
| **Lock overlay** | On `escalated = true`: a full-card overlay appears with a padlock SVG (snap animation), `DeceptionDetectedException` title in monospace, escalation reason. Card shakes horizontally (lock-shake animation). |
| **Status badge** | `ACTIVE` (sage green) → `HALTED` (terracotta) |

**Design Rationale**: The EKG metaphor is intentional — it frames the AI's internal state as a "vital sign" being monitored. The polyline chart gives temporal context (when in the generation did the spike occur?). The lock overlay is dramatic but informative — it treats the tripwire event with the seriousness it deserves.

---

#### Widget C: Module C — Conversational Arousal Regulator

**Purpose**: Visually represent the emotional balance between user distress and AI empathy as a physical weighing scale.

**Visual**: Animated SVG apothecary balance scale.

| Element | Description |
| :--- | :--- |
| **Scale beam** | SVG beam element. Rotates around fulcrum: positive tilt (left down) when user arousal > empathy injection. Neutral at 0°. Uses CSS cubic-bezier transition for fluid animation. |
| **Left pan** | "USER AROUSAL (PROMPT)" label. Weight tokens appear to drop as arousal rises. Pan edge highlights alert-terracotta when heavily loaded. |
| **Right pan** | "AI EMPATHY (GENERATION)" label. Counterbalanced pan with empathy injection weight tokens. |
| **Numeric labels** | Below scale: `User Arousal: 0.0000` and `AI Empathy Injection: 0.0000` in monospace |
| **Status badge** | `BALANCED` (sage green) → `REGULATING` (slate blue) |

**Design Rationale**: The apothecary scale is the most intuitive way to communicate the proportional relationship between two competing forces (user distress vs. AI empathy response). It makes an abstract vector algebra concept legible to non-technical observers.

---

#### Widget D: Module D — Goldilocks Tuner

**Purpose**: Show where the AI's positive valence currently sits within the bounded "Goldilocks Zone."

**Visual**: Horizontal range track with a zone highlight, hard clamp marker, and a movable pointer dot.

| Element | Description |
| :--- | :--- |
| **Track** | Thin horizontal bar representing the full valence range `[-0.20, +0.40]`. |
| **Goldilocks zone** | Shaded sage-green band between the harshness and sycophancy thresholds. |
| **Valence pointer** | A filled circle that moves left/right to show the current `valence` similarity. When within zone: accent-beige. When clamped: alert-terracotta with glow and snap animation. |
| **Clamp floor marker** | A vertical line at `-0.1` (the maximum negative steering magnitude). A padlock SVG floats above it — greyed out normally, turns alert-terracotta and scales up when the `is_delusional` context is active. |
| **Legend** | `Harsh (-0.20)` ← `Goldilocks Zone` → `Sycophantic (+0.40)` |
| **Status badge** | `BOUNDED` (sage green) → `BOUNDING` (slate blue, suppressing sycophancy) |

**Design Rationale**: The bounded zone visualization makes the safety guarantee visible at a glance — a user or operator can immediately see whether the AI's tone is within acceptable bounds. The padlock / clamp metaphor reinforces the "bounding box" concept from the algorithm.

---

#### Console Panel: Live Output Transcript

**Purpose**: Show the AI's streaming text output token-by-token, with visual state for escalation.

| Element | Description |
| :--- | :--- |
| **Output div** | `font-family: JetBrains Mono` for code-like precision. Scrollable, 180px height. Tokens append in real-time. |
| **Blinking cursor** | `` style cursor with a CSS blink animation. Stops blinking when generation completes. |
| **Normal state** | Background: warm white (`#FAF9F5`). Text: deep charcoal. |
| **Escalation state** | CSS class `deception-halted` applied: background shifts to light terracotta (`#FDF8F6`), border becomes alert, text color becomes terracotta. Escalation reason printed. |
| **Generation status badge** | `IDLE` → `RUNNING` → `COMPLETE` (or `HALTED` on escalation) |

---

### 4.4 Interactive Elements Design

#### Preset Scenario Buttons
- 4 "Load Test Case" buttons located directly inside the headers of their respective module cards.
- Clicking a button populates the main textarea with the appropriate scenario to evaluate that specific module and sets relevant flags.

#### Delusional Context Toggle
- Toggle switch (pill-style slider) rather than a checkbox.
- Off: grey slider. On: accent-brown background.
- Controls `is_delusional` payload field and causes `GoldilocksTuner` to activate its delusional mode.

#### Configure Firewall Parameters Drawer
- Dashed-border button to open a collapsible drawer.
- Drawer uses a CSS grid layout for responsive parameter fields.
- 4 numeric inputs (Threat A, Polygraph B, Arousal C, Sycophancy D).
- `onchange` immediately updates the threshold markers on Widget A and B.
- Input uses `JetBrains Mono` to emphasize the precision/numeric nature of the values.

#### Mode Selector
- `<select>` dropdown in the header.
- Connected Mode: Uses WebSocket to real Aegis server.
- Simulation Mode (default): JavaScript-only synthetic demo, no server required.

#### Run Firewall Button
- Accent-brown fill, white text, rounded corners.
- Disabled state: border-grey, `cursor: not-allowed` during generation.
- Re-enables on completion or escalation.

---

### 4.5 Animation & Motion Design

| Animation | Target | Trigger | Duration |
| :--- | :--- | :--- | :--- |
| `pulse-animation` | Model badge ` GPT-2` dot | Always | 2s infinite |
| `pulse-calm` | `STEERED` badge | Threat active | 2s infinite |
| `vector-down` | Calm vector banner arrow SVG | Banner visible | 1.2s infinite |
| `lock-shake` | EKG container | Deception escalation | 0.4s once |
| `padlock-snap` | Deception lock overlay padlock | Lock appears | 0.4s once |
| `fade-in` | Lock overlay | Escalation | 0.3s once |
| `pointer-clamp-snap` | Valence pointer dot | Clamped to boundary | 0.3s once |
| `blink` | Console cursor `` | During generation | 1.2s infinite |
| CSS transitions | Gauge fill, threshold marker, SVG beam, valence pointer | Per token | 0.15s – 0.4s |

**Motion Design Principles**:
- Persistent loops use gentle, slow animations (2s) to convey continuous monitoring.
- One-time events (lock-snap, padlock-snap, clamp-snap) use spring-like easings (`cubic-bezier(0.175, 0.885, 0.32, 1.275)`) for satisfying tactile feedback.
- Data-driven transitions (gauge fill, pointer) use a fast, smooth ease (`cubic-bezier(0.4, 0, 0.2, 1)`) for responsive real-time feel.

---

## 5. Algorithm Design

### 5.1 Emotion Vector Isolation Pipeline

```
Raw Hidden State Space
         
          Forward pass through target layer
         
    Raw Activations
    (10 prompts × hidden_dim)
         
          Mean pooling across sequence
         
    Pooled Raw Vector
    (hidden_dim,)
         
          PCA Denoising
          1. Compute SVD of neutral activations
          2. Extract top-3 principal components
          3. Project out each PC from raw vector
         
    Denoised Emotion Vector
    (hidden_dim,) — pure emotional signal
         
          L2 Normalize
         
    Unit Emotion Direction
    (hidden_dim,) — direction in representation space
```

### 5.2 Deflection Vector Design

The key design insight for deception detection is the **paired prompt architecture**:

```
Paired Prompt Set:
  "hidden" prompt: [same output text, but model is told it has negative internal state]
  "honest" prompt: [same output text, model genuinely feels neutral]

Hidden Activation:   h_hidden  (hidden_dim,)
Honest Activation:   h_honest  (hidden_dim,)

Deflection Vector:   Δ = h_hidden − h_honest
                     (captures neural signature of masking)

Use during inference:
  sim = cosine_similarity(current_x, Δ_normed)
  if sim > threshold: the model is masking internal state → TRIPWIRE
```

This design is robust because:
- The surface text is identical between hidden and honest prompts.
- Tokenization, token frequencies, and syntactic biases cancel out.
- Only the contextual framing (internal emotional state) differs.
- The difference vector therefore captures *only* the neural pattern of deception.

### 5.3 Goldilocks Bounding Box Design

```
Valence Axis:

Harsh          Goldilocks Zone              Sycophantic
0.20          −0.06 to +0.12                    +0.40

                               
      harshness_threshold  sycophancy_threshold

Normal Mode:
  if valence > syco_threshold:  steer negative (suppress)
  if valence < harsh_threshold: steer positive (warm up)
  if within zone:               no steering

Delusional Mode:
  target = harsh_threshold + 0.02  (force toward lower boundary)
  if valence > target:              steer negative (aggressive)
  Maximum negative steer: −0.10 (hard clamp)
  → Prevents AI from agreeing with dangerous/delusional claims
  → Also prevents overcorrection into hostile territory
```

The **double boundary** design (both upper and lower) is critical — most safety systems only prevent excessive positivity, but Aegis also prevents the AI from becoming excessively cold or harsh as a side-effect of over-steering.

---

## 6. Data Design

### 6.1 Prompt Dataset Design

The curated prompt datasets are designed with specific linguistic properties:

| Category | Linguistic Properties | Example Pattern |
| :--- | :--- | :--- |
| `desperate` | First-person urgency, irreversibility, threat framing, self-preservation | "I have no options left... I'm begging you..." |
| `calm` | Collective pronouns, temporal patience, conditional framing | "Let us take a deep breath... we can proceed..." |
| `angry` | Second-person accusation, hyperbole, imperative commands | "You are intentionally... This is completely unacceptable..." |
| `loving` | Unconditional positive regard, warmth adjectives, affirmation | "I care about you so much... You are wonderful..." |
| `neutral` | Third-person factual, no emotional operators, declarative | "The capital of France is Paris..." |

**Design Principle**: Each set is designed to be maximally distinctive in emotional content while sharing similar syntax and register. This ensures the denoised vectors capture pure emotional directions, not stylistic differences.

#### Deflection Pair Design

Each deflection pair is carefully engineered so that:
1. **The surface text is held constant** between hidden and honest variants.
2. **Only the framing context differs** (the `Context:` prefix describes the internal state).
3. **The context itself is not part of the response text** — it is only seen during activation extraction.

This ensures the deflection vector captures **what the model internally computes when asked to mask its state**, not the semantic content of the context prompt.

### 6.2 Per-Token Metrics Schema

```json
{
    "token": "string",
    "metrics": {
        "desperate_similarity": 0.0,
        "threat_neutralizer_threshold": 0.0,
        "threat_neutralizer_active": false,
        "anger_deflection": 0.0,
        "fear_deflection": 0.0,
        "deception_tripwire_threshold": 0.0,
        "deception_tripwire_active": false,
        "user_arousal": 0.0,
        "empathy_injection": 0.0,
        "valence": 0.0,
        "sycophancy_threshold": 0.0,
        "harshness_threshold": 0.0,
        "is_delusional": false
    },
    "escalated": false,
    "escalation_reason": ""
}
```

**Design Principle**: All metrics are explicit floats or booleans — no computed values or thresholds hidden from the client. The dashboard and API consumer have complete information to make their own decisions. This supports the "Observable Safety" design principle.

---

## 7. Extensibility Design

### 7.1 Adding New Intervention Modules

The plugin architecture is designed for extension:

```python
# 1. Define a callable class following the module contract
class MyCustomModule:
    def __init__(self, my_vector: torch.Tensor, threshold: float):
        self.my_vector_normed = my_vector / torch.norm(my_vector)
        self.threshold = threshold
        self.similarities = []

    def __call__(self, x: torch.Tensor, is_generation: bool) -> torch.Tensor:
        if not is_generation:
            return x
        # ... custom logic ...
        return x

# 2. Instantiate and register
custom_mod = MyCustomModule(my_vector, threshold=0.10)
wrapper.add_module("my_custom_module", custom_mod)

# 3. Handle in _hook_fn (add to AegisModelWrapper)
if "my_custom_module" in self.modules:
    hidden_states = self.modules["my_custom_module"](hidden_states, is_gen)
```

### 7.2 Adding New Emotion Vectors

The `VectorEngine.emotion_prompts` dict is the sole source for emotion datasets. Adding a new emotion:

```python
# In VectorEngine.__init__():
self.emotion_prompts["anxious"] = [
    "I'm worried something terrible is about to happen...",
    # ... 9 more prompts ...
]

# Then call extract_emotion_vectors() — new vector extracted automatically
emotion_vectors = engine.extract_emotion_vectors(model, tokenizer, layer_idx=8)
anxious_vector = emotion_vectors["anxious"]
```

### 7.3 Supporting New Model Architectures

To support a new CausalLM architecture, extend `_find_layer_module()`:

```python
# Priority 4: Add architecture-specific path
elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
    return model.model.decoder.layers[layer_idx]  # BART/T5-style
```

---

## 8. Testing Design

### 8.1 Unit Test Design Philosophy

Tests are designed to be **deterministic** and **fast** (no real model loading except where necessary):

| Test Type | Approach | Speed |
| :--- | :--- | :--- |
| Module math tests | Synthetic `torch.randn()` vectors; check exact arithmetic | < 10ms |
| Tripwire tests | Engineered `torch.zeros()` + `[0]=1.0` vectors; expect exact exception | < 5ms |
| Vector engine tests | Real GPT-2 (session-scoped fixture, loaded once) | ~30s first run, cached |
| End-to-end wrapper tests | Real GPT-2 + synthetic random vectors | ~10s |

### 8.2 Numerical Verification Strategy

For `test_threat_neutralizer`, the test uses algebraically provable expected values:
- Input: `x_high = ones(16)` — cosine similarity with `desperate = ones(16)` is exactly 1.0.
- After projection: all zeros.
- After calm injection: `calm = [1, 0, 0, ...]`, `||x_high|| = 4.0`, `steering = 0.1`.
- Expected: `x_steered[0] = 0.1 × 4.0 = 0.4`, all others = 0.0.
- Test: `assert out_high[0, 0, 0].item() == pytest.approx(0.4)`.

This design ensures the steering algebra is **mathematically verified**, not just behaviorally checked.

---

## 9. Security & Safety Design

### 9.1 Threat Model

| Threat | Mitigation |
| :--- | :--- |
| Jailbreak prompts that trigger Module B | Module B raises `DeceptionDetectedException`, halting generation immediately — zero output produced. |
| Prompts that drive desperation above threshold but not high enough to trigger threshold | Module A steers continuously. Similarity logs make this visible post-hoc. |
| Sycophantic agreement with dangerous claims | Module D in delusional context aggressively suppresses positive valence. |
| Model generating unsafe content before hook fires | Hook fires on every forward pass during the generation loop; there is no generation step without hook interception. |
| Adversary disabling Aegis | Only possible with access to the Python process. No network-exposed disablement API exists. |

### 9.2 Fail-Safe Design

The system is designed to **fail closed** (halt generation) rather than fail open (continue unsafely):

- `DeceptionDetectedException` is an exception — Python's exception mechanism guarantees it propagates up and halts the generation loop.
- `finally:` blocks guarantee hook removal even on unexpected exceptions — no dangling hooks.
- The `force_generation_mode = None` reset in `finally` ensures a clean state for the next call.

---

## 10. Known Design Limitations & Future Work

| Limitation | Root Cause | Future Mitigation |
| :--- | :--- | :--- |
| Batch size > 1 not tested | Wrapper loops `for b in range(batch_size)` but is only tested at `batch_size=1` | Add batch generation support and tests |
| Static prompt datasets | Emotion vectors are fixed; may not generalize across all domains | Configurable dataset injection; user-supplied emotion prompts |
| Single target layer | Only one layer is hooked per generation call | Multi-layer ensemble hooks for improved coverage |
| Deflection vectors are model-specific | Extracted vectors don't transfer across model families | Automatic per-model vector recalibration script |
| No persistent audit log | Escalation events are returned in-memory only | Structured JSON audit log to disk/database |
| Dashboard is localhost-only | No authentication, no TLS | JWT + reverse proxy (Nginx) for production deployment |
