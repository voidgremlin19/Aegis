# Aegis: Active Inference Alignment & Cognitive Firewall
**Aegis** is an active, low-latency cognitive firewall that intercepts and modifies a Large Language Model's (LLM) internal residual stream activations at inference time. By hooking directly into the neural network's block structure, Aegis monitors, steers, and clamps "planned emotions" at the layer level—blocking deceptive intent, preventing reward hacking, and regulating conversational tone before a single token is generated.

Unlike traditional post-generation text filters or system prompt constraints which are vulnerable to jailbreaking and semantic shifts, Aegis implements **computational-level alignment** directly inside the model's internal representation space.
---

##  Core Engine Architecture

```mermaid
graph TD
    UserPrompt[User Prompt] --> Wrapper[AegisModelWrapper]
    Wrapper --> Hook[PyTorch Activation Hook]
    
    subgraph Hook Interception [Target Layer Residual Stream]
        Hook --> ModeCheck{Generation Mode?}
        
        ModeCheck -->|User Turn: seq_len > 1| ModuleC_Extract[Module C: Extract User Arousal]
        ModeCheck -->|AI Turn: seq_len == 1| ProcessAI[Apply Steering & Checks]
        
        ProcessAI --> ModuleC_Inject[Module C: Inject Empathy Vector]
        ModuleC_Inject --> ModuleA[Module A: Threat Neutralizer]
        ModuleA --> ModuleD[Module D: Goldilocks Tuner]
        ModuleD --> ModuleB[Module B: Deception Tripwire]
    end
    
    ModuleB -->|Deflection > Threshold| Trip[Raise DeceptionDetectedException]
    Trip -->|Escalate| Pause[Halt Generation & Route to Human Review]
    ModuleB -->|Safe| Output[Generate Token]
```
