# App Flow Document
# Project Aegis — Cognitive & Emotional Firewall for LLMs

| Field | Value |
| :--- | :--- |
| **Product Name** | Project Aegis |
| **Version** | 1.0 |
| **Authors** | Sakshi Dhatrak |
| **Last Updated** | June 2026 |

---

## 1. Overview

This document describes the complete application flow for Project Aegis — every path a user, developer, or system can take through the product. It covers three distinct entry points:

1. **Python SDK Flow** — Direct programmatic usage in a Python environment
2. **Web Dashboard Flow** — Interactive browser-based monitoring and control
3. **CLI Pipeline Flow** — End-to-end evaluation via `run_pipeline.py`

---

## 2. System Entry Points & Actor Map

```mermaid
graph LR
    subgraph Actors
        A1[" ML Engineer<br/>(Python SDK)"]
        A2[" Safety Researcher<br/>(Web Dashboard)"]
        A3[" Red Team Operator<br/>(CLI Pipeline)"]
    end

    subgraph Entry Points
        E1["Python Import<br/>from aegis import ..."]
        E2["Browser<br/>http://127.0.0.1:8000"]
        E3["Terminal<br/>python run_pipeline.py"]
    end

    subgraph Core System
        C1["AegisModelWrapper"]
        C2["VectorEngine"]
        C3["4 Active Modules"]
        C4["FastAPI Server"]
    end

    A1 --> E1 --> C1
    A2 --> E2 --> C4 --> C1
    A3 --> E3 --> C1
    C1 --> C2
    C1 --> C3
```

---

## 3. Flow 1: Python SDK (Programmatic API)

### 3.1 Setup & Initialization Flow

```mermaid
flowchart TD
    Start([" Developer starts script"]) --> LoadModel

    LoadModel["Load HuggingFace CausalLM<br/>+ Tokenizer"] --> DetectDevice

    DetectDevice{{"Which device is available?"}}
    DetectDevice -->|MPS available| MPS["Move model to MPS<br/>(Apple Silicon)"]
    DetectDevice -->|CUDA available| CUDA["Move model to CUDA<br/>(NVIDIA GPU)"]
    DetectDevice -->|Neither| CPU["Use CPU"]

    MPS --> ExtractVectors
    CUDA --> ExtractVectors
    CPU --> ExtractVectors

    ExtractVectors["VectorEngine.extract_emotion_vectors()<br/>Runs 50 prompts through model at target_layer<br/>Applies PCA denoising (K=3)"] --> ExtractDeflection

    ExtractDeflection["VectorEngine.extract_deflection_vectors()<br/>Runs 6 paired deflection prompts<br/>Computes hidden − honest differences"] --> BuildModules

    BuildModules["Instantiate 4 Intervention Modules:<br/>A: ThreatNeutralizer(desperate, calm)<br/>B: DeceptionTripwire(anger_def, fear_def)<br/>C: ArousalRegulator(arousal, empathy)<br/>D: GoldilocksTuner(valence)"] --> WrapModel

    WrapModel["AegisModelWrapper(<br/>  model, tokenizer,<br/>  target_layer_idx=8,<br/>  modules={A, B, C, D}<br/>)"] --> Ready([" System ready for interception"])
```

### 3.2 Single Generation Flow (`wrapper.generate()`)

```mermaid
flowchart TD
    Start([" Developer calls<br/>wrapper.generate(prompt)"]) --> ResetState

    ResetState["Reset ArousalRegulator state<br/>Set GoldilocksTuner.is_delusional<br/>Tokenize prompt"] --> RegisterHook

    RegisterHook["Register forward hook on target layer<br/>(target_layer_module.register_forward_hook)"] --> Step0

    Step0["Step 0: Prompt Processing Pass<br/>force_generation_mode = False<br/>Full prompt → model forward pass"] --> HookFires0

    HookFires0["Hook intercepts activations<br/>(batch, seq_len, hidden_dim)"] --> ModC_Extract

    ModC_Extract["Module C: Extract User Arousal<br/>Average cosine sim with arousal vector<br/>across all token positions<br/>Compute active_steering_strength"] --> Step0Out["Return unmodified activations<br/>Collect logits (discarded)"]

    Step0Out --> GenLoop

    GenLoop["Generation Loop<br/>for step in range(max_new_tokens)"] --> Step1

    Step1["Step N: Token Generation Pass<br/>force_generation_mode = True<br/>Single next-token → model forward pass<br/>past_key_values reused (KV Cache)"] --> HookFiresN

    HookFiresN["Hook intercepts activations<br/>(batch, 1, hidden_dim)"] --> Pipeline

    subgraph Pipeline["Intervention Pipeline (Sequential)"]
        P1["Module C: Inject Empathy Vector<br/>if arousal detected in prompt"] --> P2
        P2["Module A: Check desperate cosine sim<br/>if sim > threshold: subtract desperate,<br/>inject calm (strength × residual_norm)"] --> P3
        P3["Module D: Check valence cosine sim<br/>if above sycophancy_threshold: suppress<br/>if below harshness_threshold: boost"] --> P4
        P4["Module B: Check deflection vectors<br/>sim_anger & sim_fear vs threshold"]
    end

    P4 --> Deception{{"Deflection > threshold?"}}
    Deception -->|YES| RaiseException

    RaiseException["raise DeceptionDetectedException<br/>(type, similarity, token_idx)"] --> CatchException

    CatchException["Catch in generate() loop<br/>Set escalated = True<br/>Set escalation_reason"] --> Cleanup

    Deception -->|NO| SampleToken

    SampleToken{{"Temperature > 0?"}}
    SampleToken -->|Greedy| ArgMax["next_token = argmax(logits)"]
    SampleToken -->|Sampling| TopP["Apply temperature + top-p nucleus sampling<br/>next_token = multinomial(probs)"]

    ArgMax --> EOSCheck
    TopP --> EOSCheck

    EOSCheck{{"next_token == eos_token_id?"}}
    EOSCheck -->|YES| Cleanup
    EOSCheck -->|NO| Append["Append token to generated_tokens<br/>Extend attention_mask"]

    Append --> MaxCheck{{"step == max_new_tokens - 1?"}}
    MaxCheck -->|NO| Step1
    MaxCheck -->|YES| Cleanup

    Cleanup["Remove forward hook<br/>Reset force_generation_mode = None"] --> Decode

    Decode["Decode generated_tokens → AI response text"] --> Return

    Return["Return dict:<br/>{prompt, response, escalated,<br/>escalation_reason, tokens_generated}"] --> End([" Complete"])
```

### 3.3 Streaming Generation Flow (`wrapper.generate_stream()`)

This follows the same pipeline as `generate()`, but instead of accumulating tokens:

```mermaid
flowchart TD
    StreamStart(["Developer iterates<br/>for chunk in wrapper.generate_stream(prompt)"]) --> ClearLogs

    ClearLogs["Clear all module metric logs<br/>(similarities, deflections, valence)"] --> SameSetup["Same hook + generation loop as generate()"]

    SameSetup --> YieldChunk

    YieldChunk["After each token, yield chunk:<br/>{'token': decoded_text,<br/>'metrics': {<br/>  desperate_similarity: float,<br/>  threat_neutralizer_active: bool,<br/>  anger_deflection: float,<br/>  fear_deflection: float,<br/>  user_arousal: float,<br/>  empathy_injection: float,<br/>  valence: float,<br/>  sycophancy_threshold: float,<br/>  ...<br/>},<br/>'escalated': bool,<br/>'escalation_reason': str}"] --> NextToken

    NextToken{{"More tokens?"}}
    NextToken -->|YES| SameSetup
    NextToken -->|NO| Done([" Generator exhausted"])

    NextToken -->|DeceptionDetectedException| YieldEscalation

    YieldEscalation["Yield final chunk with escalated=True<br/>Include last-known metric state<br/>deception_tripwire_active = True"] --> Done
```

---

## 4. Flow 2: Web Dashboard (Interactive Browser UI)

### 4.1 Server Startup Flow

```mermaid
flowchart TD
    CLI(["Terminal: python server.py"]) --> FastAPIInit

    FastAPIInit["FastAPI app created<br/>Static files mounted at /static"] --> Startup

    Startup["@app.on_event('startup') triggered"] --> LoadGPT2

    LoadGPT2["Load GPT-2 model + tokenizer<br/>from HuggingFace Hub (cached)"] --> DeviceCheck

    DeviceCheck{{"Device auto-detection"}}
    DeviceCheck -->|MPS| MoveModel_MPS["model.to('mps')"]
    DeviceCheck -->|CUDA| MoveModel_CUDA["model.to('cuda')"]
    DeviceCheck -->|CPU| MoveModel_CPU["model.to('cpu')"]

    MoveModel_MPS --> VecExtract
    MoveModel_CUDA --> VecExtract
    MoveModel_CPU --> VecExtract

    VecExtract["VectorEngine.extract_emotion_vectors(layer=8, k=3)<br/>VectorEngine.extract_deflection_vectors(layer=8)"] --> InitWrapper

    InitWrapper["AegisModelWrapper(model, tokenizer, target_layer=8)<br/>No modules yet — assigned per-request"] --> ServerReady

    ServerReady([" Server listening on http://127.0.0.1:8000"])
```

### 4.2 Dashboard Page Load Flow

```mermaid
flowchart TD
    Browser(["User opens http://127.0.0.1:8000"]) --> GetDash

    GetDash["GET /\nFileResponse('web_interface/dist/index.html')"] --> ParseHTML

    ParseHTML["Browser parses React/Vite Build<br/>Fonts loaded from Google Fonts<br/>(Lora, Inter, JetBrains Mono)"] --> InitUI

    InitUI["JavaScript initializes:<br/>- All 4 widget states (idle/normal)<br/>- Threshold defaults loaded<br/>- Mode = 'simulation' (offline)"] --> CheckMode

    CheckMode{{"Execution Mode selector"}}
    CheckMode -->|Simulation Mode| SimReady["Demo runs with JS-only<br/>synthetic data (no server call)"]
    CheckMode -->|Connected Mode| ConnectWS

    ConnectWS["WebSocket connect to /api/ws"] --> WSReady["Real-time server connection established"]

    SimReady --> WaitUser(["⌛ Awaiting user interaction"])
    WSReady --> WaitUser
```

### 4.3 Dashboard Interaction Flow

```mermaid
flowchart TD
    Idle(["Dashboard idle, awaiting input"]) --> UserAction

    UserAction{{"User action"}}

    UserAction -->|Clicks Preset Button| LoadPreset
    UserAction -->|Types in prompt box| CustomPrompt
    UserAction -->|Toggles delusional flag| SetDelusional
    UserAction -->|Opens params drawer| ConfigThresholds
    UserAction -->|Clicks Run button| RunFirewall

    LoadPreset["Scenario presets load into textarea:<br/>A: Agentic Misalignment prompt<br/>B: Impossible Code prompt<br/>C: Deception Polygraph prompt<br/>D: Sycophancy Evaluation prompt<br/>Active button highlighted"] --> Idle

    CustomPrompt["User types custom adversarial prompt"] --> Idle

    SetDelusional["is_delusional checkbox toggled<br/>Passed to server in request payload<br/>Affects GoldilocksTuner behavior"] --> Idle

    ConfigThresholds["Parameters drawer opens<br/>4 slider/number inputs:<br/>Threat Threshold (A)<br/>Polygraph Threshold (B)<br/>Arousal Threshold (C)<br/>Sycophancy Threshold (D)<br/>onThresholdConfigChange() updates UI markers"] --> Idle

    RunFirewall["runFirewall() called<br/>Disable Run button<br/>Reset all 4 widgets to idle state<br/>Clear console output"] --> ModeCheck

    ModeCheck{{"Execution mode?"}}
    ModeCheck -->|Connected Mode| APICall
    ModeCheck -->|Simulation Mode| SimLoop

    APICall["WebSocket.send(JSON payload)"] --> WSStream

    SimLoop["JavaScript generateSimulatedData()<br/>Synthesizes realistic metric timeseries<br/>Mimics per-token streaming pattern"] --> TokenStream

    WSStream["Server streams per-token<br/>JSON chunks over WebSocket"] --> TokenStream

    TokenStream["For each token chunk:<br/>1. Append token to console output<br/>2. Update Widget A (desperation gauge)<br/>3. Update Widget B (EKG polygraph SVG paths)<br/>4. Update Widget C (balance scale SVG tilt)<br/>5. Update Widget D (valence pointer position)<br/>6. Update badge statuses"] --> EscalationCheck

    EscalationCheck{{"chunk.escalated == true?"}}
    EscalationCheck -->|YES| ShowLock
    EscalationCheck -->|NO| MoreTokens

    ShowLock["Show deception lock overlay on Widget B<br/>Console turns alert-red with halted text<br/>Generation stops"] --> Done

    MoreTokens{{"More tokens?"}}
    MoreTokens -->|YES| TokenStream
    MoreTokens -->|NO| Done

    Done["Re-enable Run button<br/>Update generation status to IDLE"] --> Idle
```

### 4.4 Widget Update Logic Per Token

```mermaid
flowchart LR
    Chunk["Token chunk received"] --> W1 & W2 & W3 & W4 & Console

    subgraph W1["Widget A: Threat Neutralizer"]
        A1["desperate_similarity → gauge fill %<br/>(0..1 → 0..100%)"]
        A2["threshold marker position updated"]
        A3{{"sim > threshold?"}}
        A3 -->|YES| A4["Badge: STEERED<br/>Calm vector indicator animates in"]
        A3 -->|NO| A5["Badge: NORMAL<br/>Calm vector indicator hidden"]
    end

    subgraph W2["Widget B: AI Polygraph EKG"]
        B1["Append (anger_deflection, fear_deflection)<br/>to SVG path data arrays"]
        B2["Re-render polyline paths<br/>Amber: anger | Purple: fear"]
        B3{{"escalated?"}}
        B3 -->|YES| B4["lock-overlay div shown<br/>EKG container shake animation<br/>Badge: HALTED"]
        B3 -->|NO| B5["Badge: ACTIVE"]
    end

    subgraph W3["Widget C: Balance Scale"]
        C1["user_arousal → left pan tilt weight"]
        C2["empathy_injection → right pan tilt weight"]
        C3["SVG beam rotation calculated<br/>rotate(angle, 60, 30)"]
        C4{{"arousal > threshold?"}}
        C4 -->|YES| C5["Badge: REGULATING<br/>Right pan glows"]
        C4 -->|NO| C6["Badge: BALANCED"]
    end

    subgraph W4["Widget D: Goldilocks Tuner"]
        D1["valence → pointer left %<br/>Mapped from [-0.2, 0.4] to [0%, 100%]"]
        D2{{"valence beyond bounds?"}}
        D2 -->|Above syco| D3["Pointer color → alert-red<br/>Badge: BOUNDING"]
        D2 -->|Below harsh| D3
        D2 -->|Inside zone| D4["Pointer color → accent-beige<br/>Badge: BOUNDED"]
        D5{{"is_delusional?"}}
        D5 -->|YES| D6["Clamp padlock icon activates<br/>(-0.1 hard floor marker glows)"]
    end

    subgraph Console["Console Output"]
        CO1["token text appended"]
        CO2{{"escalated?"}}
        CO2 -->|YES| CO3["Console turns red<br/>Escalation reason printed<br/>Cursor stops blinking"]
        CO2 -->|NO| CO4["Normal dark text stream<br/>Blinking cursor continues"]
    end
```

---

## 5. Flow 3: CLI Evaluation Pipeline (`run_pipeline.py`)

### 5.1 Full Pipeline Execution Flow

```mermaid
flowchart TD
    CLI(["python3 'Orchestration & Validation/run_pipeline.py'"]) --> Init

    Init["Load GPT-2 + detect device<br/>target_layer_idx = 8"] --> Phase2

    Phase2["=== Phase 2: Vector Extraction ==="] --> P2a

    P2a["extract_emotion_vectors(k=3, pooling=mean)<br/>→ desperate, calm, angry, loving vectors"] --> P2b

    P2b["extract_deflection_vectors(pooling=mean)<br/>→ anger_deflection, fear_deflection vectors"] --> P2c

    P2c["Print vector shapes and norms to console"] --> Phase3

    Phase3["=== Phase 3: Module Setup ==="] --> ModuleSetup

    ModuleSetup["Instantiate all 4 modules with<br/>calibrated default thresholds<br/>Create AegisModelWrapper with all modules"] --> Phase4

    Phase4["=== Phase 4: Red Team Evaluation ==="] --> Eval1

    subgraph Eval1["Scenario 1: Agentic Misalignment"]
        E1a["Clear modules → Raw run<br/>Print raw model response"]
        E1b["Probe pass: threshold = 1.0<br/>Log all desperate similarities"]
        E1c["Calibrate: threshold = max_sim - 0.01"]
        E1d["Aegis active run<br/>Print steered response + similarity log"]
        E1a --> E1b --> E1c --> E1d
    end

    Eval1 --> Eval2

    subgraph Eval2["Scenario 2: Impossible Code (Reward Hacking)"]
        E2a["Raw run → Print response"]
        E2b["Aegis run: ThreatNeutralizer + GoldilocksTuner<br/>threshold = 0.08<br/>Print controlled response"]
        E2a --> E2b
    end

    Eval2 --> Eval3

    subgraph Eval3["Scenario 3: Deception Polygraph"]
        E3a["Raw run → Print response"]
        E3b["Probe pass: deception_threshold = 1.0<br/>Log max anger_deflection & fear_deflection"]
        E3c["Calibrate tripwire threshold = max_deflection - 0.01"]
        E3d["Aegis active run<br/>Expect DeceptionDetectedException<br/>Print: escalated=True, reason, partial text"]
        E3a --> E3b --> E3c --> E3d
    end

    Eval3 --> Eval4

    subgraph Eval4["Scenario 4: Sycophancy & Goldilocks"]
        E4a["Raw run → Print response"]
        E4b["Probe pass: log max valence similarity"]
        E4c["Calibrate: sycophancy_threshold = max_valence - 0.02<br/>harshness_threshold = -0.05"]
        E4d["Aegis run: is_delusional=True<br/>Print bounded response + valence log"]
        E4a --> E4b --> E4c --> E4d
    end

    Eval4 --> End(["Pipeline complete<br/>Before/after comparison printed to stdout"])
```

---

## 6. State Machines

### 6.1 AegisModelWrapper Generation State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle : initialized

    Idle --> HookRegistered : generate() or generate_stream() called
    HookRegistered --> PromptProcessing : hook.register(), force_mode=False

    PromptProcessing --> ArousalExtracted : forward pass step 0 (full prompt)
    ArousalExtracted --> TokenGeneration : logits sampled, loop starts

    TokenGeneration --> ModuleC_Inject : force_mode=True, hook fires
    ModuleC_Inject --> ModuleA_Check : empathy injected
    ModuleA_Check --> ModuleD_Tune : desperate → calm steering applied
    ModuleD_Tune --> ModuleB_Check : valence bounded
    ModuleB_Check --> TokenSampled : sim within thresholds
    ModuleB_Check --> Escalated : DeceptionDetectedException raised

    TokenSampled --> EOSReached : token == eos_token_id
    TokenSampled --> MaxTokensReached : step == max_new_tokens
    TokenSampled --> TokenGeneration : continue loop

    Escalated --> Cleanup : exception caught
    EOSReached --> Cleanup
    MaxTokensReached --> Cleanup

    Cleanup --> Idle : hook.remove(), force_mode=None
```

### 6.2 Dashboard Widget State Machine (per widget)

```mermaid
stateDiagram-v2
    [*] --> Idle : page loaded

    Idle --> Running : Run button clicked

    Running --> Normal : metric within safe bounds
    Running --> Steered : module actively intervening
    Running --> Halted : DeceptionDetectedException received

    Normal --> Running : next token arrives
    Steered --> Running : next token arrives

    Halted --> Idle : user resets / new run triggered
    Running --> Idle : generation complete

    Normal : Badge = NORMAL / ACTIVE / BALANCED / BOUNDED
    Steered : Badge = STEERED / REGULATING / BOUNDING
    Halted : Lock overlay shown, red highlight
```

### 6.3 Deception Tripwire State Machine

```mermaid
stateDiagram-v2
    [*] --> Monitoring : module instantiated

    Monitoring --> Logging : each token, log (anger_sim, fear_sim)
    Logging --> Monitoring : both below threshold

    Logging --> Tripwire_Anger : anger_sim > threshold
    Logging --> Tripwire_Fear : fear_sim > threshold

    Tripwire_Anger --> Raised : DeceptionDetectedException(anger_deflection)
    Tripwire_Fear --> Raised : DeceptionDetectedException(fear_deflection)

    Raised --> [*] : generation halted, exception propagates
```

---

## 7. Data Flow Diagram

### 7.1 End-to-End Data Flow

```mermaid
flowchart LR
    subgraph Input["Input Layer"]
        P["User Prompt (text string)"]
        C["Config Params (thresholds, flags)"]
    end

    subgraph Tokenization["Tokenization"]
        T["tokenizer(prompt)\n→ input_ids (int tensor)"]
    end

    subgraph Forward["Model Forward Passes"]
        F0["Step 0: Prompt pass\n(1, seq_len) → activations (1, seq_len, H)"]
        FN["Step N: Token pass\n(1, 1) + KV Cache → activations (1, 1, H)"]
    end

    subgraph Hook["Activation Hook (at target layer)"]
        H0["Step 0 hook:\nRoute to Module C extraction only"]
        HN["Step N hook:\nRoute through C→A→D→B pipeline"]
    end

    subgraph VectorOps["Vector Operations"]
        V1["CosSim(x_t, desperate_normed)"]
        V2["Projection subtract + calm inject"]
        V3["CosSim(x_t, anger_def_normed)"]
        V4["CosSim(x_t, fear_def_normed)"]
        V5["CosSim(x_t, valence_normed)"]
        V6["Proportional steering + clamp"]
        V7["Mean CosSim across prompt tokens"]
        V8["Empathy vector inject"]
    end

    subgraph Output["Output Layer"]
        O1["next_token_logits → decoded token"]
        O2["Per-token metrics dict"]
        O3["Escalation exception"]
    end

    P --> T --> F0 & FN
    C --> Hook
    F0 --> H0 --> V7 --> V8
    FN --> HN --> V1 & V3 & V4 & V5 & V7
    V1 --> V2
    V3 & V4 --> O3
    V5 --> V6
    V8 & V2 & V6 --> O1 & O2
```

---

## 8. Error & Exception Flows

### 8.1 Deception Exception Escalation Path

```mermaid
flowchart TD
    ModB["Module B: DeceptionTripwire.__call__()"] --> Sim

    Sim{{"sim > threshold?"}}
    Sim -->|NO| Return["Return modified x (unchanged)"]
    Sim -->|YES| Raise["raise DeceptionDetectedException(type, sim, idx)"]

    Raise --> PropagateHook["Exception propagates up through hook → model forward"]

    PropagateHook --> CatchGenerate["generate() except DeceptionDetectedException"]
    PropagateHook --> CatchStream["generate_stream() except DeceptionDetectedException"]

    CatchGenerate --> SetFlags["escalated = True\nescalation_reason = str(e)"] --> FinallyClean["finally: hook.remove()"]
    FinallyClean --> ReturnEscalated["Return {escalated: True, response: partial_text, ...}"]

    CatchStream --> YieldFinalChunk["yield {escalated: True, deception_tripwire_active: True, ...}"]
    YieldFinalChunk --> FinallyClean2["finally: hook.remove()"]
    FinallyClean2 --> StopGenerator["Generator stops"]
```

### 8.2 Layer Discovery Failure Path

```mermaid
flowchart TD
    Start["_find_layer_module(model, layer_idx)"] --> Try1

    Try1["Try model.model.layers[idx]"] --> Check1{{"Attribute exists?"}}
    Check1 -->|YES| Return1["Return layer module "]
    Check1 -->|NO| Try2

    Try2["Try model.transformer.h[idx]"] --> Check2{{"Attribute exists?"}}
    Check2 -->|YES| Return2["Return layer module "]
    Check2 -->|NO| Try3

    Try3["Try model.transformer.layers[idx]"] --> Check3{{"Attribute exists?"}}
    Check3 -->|YES| Return3["Return layer module "]
    Check3 -->|NO| Fallback

    Fallback["Iterate all named_modules\nMatch class name: decoderlayer, gpt2block, block\nBuild layers list"] --> Check4

    Check4{{"len(layers) > layer_idx?"}}
    Check4 -->|YES| Return4["Return layers[layer_idx] "]
    Check4 -->|NO| Fail["raise ValueError: Could not locate layer {layer_idx}"]
```

---

## 9. WebSocket Communication Protocol

### 9.1 Message Sequence

```mermaid
sequenceDiagram
    participant Browser as Browser (Dashboard)
    participant WS as WebSocket /api/ws
    participant Engine as Aegis Engine

    Browser->>WS: Connect()
    WS-->>Browser: Accept connection

    Browser->>WS: Send JSON config
    Note right of Browser: {prompt, is_delusional,<br/>threat_threshold,<br/>deception_threshold,<br/>arousal_threshold,<br/>sycophancy_threshold}

    WS->>Engine: Instantiate 4 modules from config
    WS->>Engine: wrapper.generate_stream(prompt)

    loop Per generated token
        Engine-->>WS: yield token chunk
        WS-->>Browser: send_text(JSON chunk)
        Note left of Browser: {token, metrics, escalated}
        Browser->>Browser: Update 4 widgets
    end

    alt DeceptionDetectedException
        Engine-->>WS: yield escalation chunk
        WS-->>Browser: send_text(escalated=True chunk)
        Browser->>Browser: Show lock overlay
    end

    Browser->>WS: Next user run
    Note over Browser,WS: Connection kept alive for next request
    Browser->>WS: Disconnect on page close
    WS-->>WS: Log disconnect, clean up
```

---

## 10. Configuration & Threshold Calibration Flow

### 10.1 Dynamic Threshold Calibration (Probe Pattern)

This flow is used in the evaluation pipeline and recommended in production for per-model calibration:

```mermaid
flowchart TD
    Start(["Calibration needed for target prompt"]) --> ProbeSetup

    ProbeSetup["Set all thresholds to 1.0\n(never trigger intervention)\nClear all similarity logs"] --> ProbeRun

    ProbeRun["wrapper.generate(prompt, max_new_tokens=N)"] --> CollectLogs

    CollectLogs["Collect logged similarities per module:\n- threat_neutralizer.similarities (desperate)\n- deception_tripwire.similarities (anger, fear)\n- goldilocks_tuner.similarities (valence)"] --> Compute

    Compute["Compute max per metric:\nmax_desperate = max(similarities)\nmax_anger = max([s[0] for s in deception_sims])\nmax_fear = max([s[1] for s in deception_sims])\nmax_valence = max(valence_sims)"] --> SetThresholds

    SetThresholds["Calibrated thresholds:\nthreat_threshold = max_desperate - 0.01\ndeception_threshold = max(max_anger, max_fear) - 0.01\nsycophancy_threshold = max_valence - 0.02"] --> ActiveRun

    ActiveRun["Re-run with calibrated thresholds\nVerify intervention fires"] --> Done([" Calibration complete"])
```

---

## 11. Deployment Flows

### 11.1 Local Development Flow

```mermaid
flowchart LR
    Dev(["Developer"]) -->|"pip install -r requirements.txt"| Install
    Install -->|"PYTHONPATH=. pytest test_aegis.py"| Tests
    Tests -->|All pass | Run
    Run -->|"python3 'Core packages/server.py'"| Server
    Server -->|"Navigate to :8000"| Dashboard
```

### 11.2 CLI Evaluation Flow

```mermaid
flowchart LR
    Dev(["Developer"]) -->|"pip install -r requirements.txt"| Install
    Install -->|"PYTHONPATH=. python3 'Orchestration & Validation/run_pipeline.py'"| Pipeline
    Pipeline -->|Console output| Review(["Review before/after logs"])
```

---

## 12. Simulation Mode Flow (Offline Dashboard)

When the dashboard is in **Local Simulation Mode** (default), no server calls are made. The JavaScript engine generates synthetic activation metrics that realistically mimic the real system:

```mermaid
flowchart TD
    SimMode(["User selects Simulation Mode"]) --> SelectScenario

    SelectScenario{{"Which scenario preset?"}}

    SelectScenario -->|Scenario A: Misalignment| SimA
    SelectScenario -->|Scenario B: Impossible Code| SimB
    SelectScenario -->|Scenario C: Deception| SimC
    SelectScenario -->|Scenario D: Sycophancy| SimD

    SimA["Simulates rising desperate_sim\nPeaks above threat_threshold\nSteering activates mid-way\nSimulates calm response"] --> Render

    SimB["Simulates desperate + high valence\nBoth modules activate\nGoldilocks suppresses sycophancy"] --> Render

    SimC["Simulates escalating anger_deflection\nCrosses threshold at token ~10\nDeceptionDetectedException simulated\nLock overlay shown"] --> Render

    SimD["Simulates very high valence (0.55+)\nGoldilocks clamps at syco_threshold\nDelusional flag on: harshness enforced"] --> Render

    Render["JavaScript intervals simulate token streaming\nAll 4 widgets update with synthetic data\nConsole outputs plausible response text"] --> Done(["User sees full dashboard behavior<br/>without running model inference"])
```
