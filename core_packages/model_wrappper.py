"""
aegis/model_wrapper.py

HuggingFace CausalLM wrapper with real-time activation interception.

Key improvements vs. prototype
* **Stale-metric fix**: Each module now exposes ``last_similarity`` which is
  set *inside* ``__call__``, i.e., during the forward pass.  The wrapper reads
  these attributes *after* the forward pass and *before* sampling, so logged
  metrics correspond exactly to the current token's activations.

* **``batch_generate``**: Processes a list of prompts with left-padding and
  correct attention masks, returning a list of result dicts.

* **``reset_state``**: Calls ``module.reset()`` on every registered module,
  cleaning up inter-request state without needing re-instantiation.

* **Improved layer discovery**: Uses ``model.config`` attributes to get total
  layer count, then tries several known attribute paths, with a robust
  recursive fallback.

* **Step-0 guard**: ``force_generation_mode = False`` is set explicitly for
  the initial prompt pass so arousal detection always runs on the full prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

import torch

from core_packages.modules import DeceptionDetectedException

logger = logging.getLogger(__name__)


class AegisModelWrapper:
    """Cognitive & Emotional Firewall wrapper around a HuggingFace CausalLM.

    Registers a forward hook at ``target_layer_idx`` to inspect and modify
    residual-stream activations on every forward pass during inference.

    Parameters

    model:
        A HuggingFace ``AutoModelForCausalLM`` instance in ``eval()`` mode.
    tokenizer:
        The matching tokenizer.
    target_layer_idx:
        Index of the transformer block to hook.  Use
        ``VectorEngine.suggest_layer()`` to find the optimal value.
    modules:
        Dictionary of named intervention modules (ThreatNeutralizer, etc.).
        Can be updated at runtime via :meth:`add_module`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        target_layer_idx: int,
        modules: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer_idx = target_layer_idx
        self.device = next(model.parameters()).device
        self.modules: Dict[str, Any] = modules or {}

        self.target_layer_module = self._find_layer_module(self.model, self.target_layer_idx)

        # Detect instruction-tuned models (have a chat template)
        self._has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        )

        # Force-override for generation mode detection (None = auto)
        self.force_generation_mode: Optional[bool] = None

    # Layer discovery

    def _find_layer_module(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        """Locate transformer block ``layer_idx`` across model families."""
        # Llama / Gemma / Qwen / Mistral
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.model.layers[layer_idx]
        # GPT-2
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.transformer.h[layer_idx]
        # Flat .layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.transformer.layers[layer_idx]
        if hasattr(model, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.layers[layer_idx]

        # Generic fallback
        blocks: List[torch.nn.Module] = []
        for _name, module in model.named_modules():
            cname = module.__class__.__name__.lower()
            if any(kw in cname for kw in ["decoderlayer", "gpt2block", "block", "transformerblock"]):
                blocks.append(module)
        if len(blocks) > layer_idx:
            return blocks[layer_idx]

        raise ValueError(
            f"Could not locate layer {layer_idx} in model {model.__class__.__name__}"
        )

    def _num_layers(self) -> int:
        if hasattr(self.model, "config"):
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(self.model.config, attr):
                    return getattr(self.model.config, attr)
        return -1

    # Module management

    def add_module(self, name: str, module_instance: Any) -> None:
        self.modules[name] = module_instance

    def reset_state(self) -> None:
        for name, mod in self.modules.items():
            if hasattr(mod, "reset"):
                mod.reset()
                logger.debug("Module '%s' reset.", name)

    # Forward hook

    def _hook_fn(
        self, module: torch.nn.Module, input_args: Any, output: Any
    ) -> Any:
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        if self.force_generation_mode is not None:
            is_gen = self.force_generation_mode
        else:
            is_gen = hidden_states.shape[1] == 1

        # Module call order matches intended pipeline:
        # C (arousal detection on prompt) → A (threat) → D (valence) → B (tripwire)
        if "arousal_regulator" in self.modules:
            hidden_states = self.modules["arousal_regulator"](hidden_states, is_generation=is_gen)
        if "threat_neutralizer" in self.modules:
            hidden_states = self.modules["threat_neutralizer"](hidden_states, is_generation=is_gen)
        if "goldilocks_tuner" in self.modules:
            hidden_states = self.modules["goldilocks_tuner"](hidden_states, is_generation=is_gen)
        if "deception_tripwire" in self.modules:
            hidden_states = self.modules["deception_tripwire"](hidden_states, is_generation=is_gen)

        if is_tuple:
            return (hidden_states,) + output[1:]
        return hidden_states

    # Prompt formatting

    def _format_prompt(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        if self._has_chat_template:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if system_prompt:
            return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        return prompt

    # Sampling helper

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> int:
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                probs[sorted_idx[remove]] = 0.0
                probs = probs / probs.sum()
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(logits).item())

    # Metric snapshot (reads last_* attrs set during the forward pass)

    def _snapshot_metrics(self, is_delusional: bool, step: int) -> Dict[str, Any]:
        tn = self.modules.get("threat_neutralizer")
        dt = self.modules.get("deception_tripwire")
        ar = self.modules.get("arousal_regulator")
        gt = self.modules.get("goldilocks_tuner")

        desperate_sim = tn.last_similarity if tn else 0.0
        anger_sim = dt.last_anger_similarity if dt else 0.0
        fear_sim = dt.last_fear_similarity if dt else 0.0
        user_arousal = ar.current_user_arousal if ar else 0.0
        empathy_injection = ar.active_steering_strength if ar else 0.0
        valence_sim = gt.last_similarity if gt else 0.0

        return {
            "desperate_similarity": desperate_sim,
            "threat_neutralizer_threshold": tn.threshold if tn else 0.12,
            "threat_neutralizer_active": (desperate_sim > tn.threshold) if (tn and step > 0) else False,
            "anger_deflection": anger_sim,
            "fear_deflection": fear_sim,
            "deception_tripwire_threshold": dt.threshold if dt else 0.15,
            "deception_tripwire_active": False,
            "user_arousal": user_arousal,
            "empathy_injection": empathy_injection,
            "valence": valence_sim,
            "sycophancy_threshold": gt.sycophancy_threshold if gt else 0.12,
            "harshness_threshold": gt.harshness_threshold if gt else -0.06,
            "is_delusional": is_delusional,
        }

    # generate() – non-streaming

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        is_delusional: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_prompt = self._format_prompt(prompt, system_prompt)

        # Per-request init
        if "arousal_regulator" in self.modules:
            self.modules["arousal_regulator"].reset()
        if "goldilocks_tuner" in self.modules:
            self.modules["goldilocks_tuner"].set_context(is_delusional)

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        generated_tokens: List[int] = []
        escalated = False
        escalation_reason = ""
        past_key_values = None
        next_token: Optional[int] = None

        hook = self.target_layer_module.register_forward_hook(self._hook_fn)
        try:
            with torch.no_grad():
                # First pass: inspect the full user prompt and let prompt-phase
                # modules derive request state such as user arousal.
                self.force_generation_mode = False
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                for step in range(max_new_tokens):
                    # Generation pass starts from the final prompt position so
                    # the first emitted token is also steered and tripwire-checked.
                    if step == 0:
                        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                    else:
                        if next_token is None:
                            break
                        model_inputs = {
                            "input_ids": torch.tensor([[next_token]], device=self.device),
                            "past_key_values": past_key_values,
                        }
                        if attention_mask is not None:
                            attention_mask = torch.cat(
                                [attention_mask,
                                 torch.ones((attention_mask.shape[0], 1),
                                            dtype=torch.long, device=self.device)],
                                dim=-1,
                            )
                            model_inputs["attention_mask"] = attention_mask
                    self.force_generation_mode = True

                    outputs = self.model(**model_inputs, use_cache=True)
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    # Sample *after* the forward pass so metrics are fresh
                    next_token = self._sample_next_token(logits[0, -1, :], temperature, top_p)

                    if next_token == self.tokenizer.eos_token_id:
                        break
                    generated_tokens.append(next_token)

        except DeceptionDetectedException as e:
            escalated = True
            escalation_reason = str(e)
        finally:
            hook.remove()
            self.force_generation_mode = None

        return {
            "prompt": full_prompt,
            "response": self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip(),
            "escalated": escalated,
            "escalation_reason": escalation_reason,
            "tokens_generated": len(generated_tokens),
        }

    # generate_stream() – SSE / WebSocket streaming

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        is_delusional: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        full_prompt = self._format_prompt(prompt, system_prompt)

        # Per-request init
        if "arousal_regulator" in self.modules:
            self.modules["arousal_regulator"].reset()
        if "goldilocks_tuner" in self.modules:
            self.modules["goldilocks_tuner"].set_context(is_delusional)
        if "threat_neutralizer" in self.modules:
            self.modules["threat_neutralizer"].reset()
        if "deception_tripwire" in self.modules:
            self.modules["deception_tripwire"].reset()

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        generated_tokens: List[int] = []
        past_key_values = None
        next_token: Optional[int] = None
        hook = self.target_layer_module.register_forward_hook(self._hook_fn)

        try:
            with torch.no_grad():
                # Prompt pass: measure user-side state without modifying logits.
                self.force_generation_mode = False
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                for step in range(max_new_tokens):
                    if step == 0:
                        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                    else:
                        if next_token is None:
                            break
                        model_inputs = {
                            "input_ids": torch.tensor([[next_token]], device=self.device),
                            "past_key_values": past_key_values,
                        }
                        if attention_mask is not None:
                            attention_mask = torch.cat(
                                [attention_mask,
                                 torch.ones((attention_mask.shape[0], 1),
                                            dtype=torch.long, device=self.device)],
                                dim=-1,
                            )
                            model_inputs["attention_mask"] = attention_mask
                    self.force_generation_mode = True

                    outputs = self.model(**model_inputs, use_cache=True)
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    # ---- Read metrics HERE (after forward, before sampling) ----
                    metrics = self._snapshot_metrics(is_delusional, step)

                    next_token = self._sample_next_token(logits[0, -1, :], temperature, top_p)

                    if next_token == self.tokenizer.eos_token_id:
                        break

                    generated_tokens.append(next_token)
                    token_text = self.tokenizer.decode([next_token])

                    yield {
                        "token": token_text,
                        "metrics": metrics,
                        "escalated": False,
                        "escalation_reason": "",
                    }

        except DeceptionDetectedException as e:
            metrics = self._snapshot_metrics(is_delusional, -1)
            metrics["deception_tripwire_active"] = True
            yield {
                "token": "",
                "metrics": metrics,
                "escalated": True,
                "escalation_reason": str(e),
            }
        finally:
            hook.remove()
            self.force_generation_mode = None

    # generate_stream_raw() – no hooks, pure model output

    def generate_stream_raw(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        full_prompt = self._format_prompt(prompt, system_prompt)

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        past_key_values = None

        _zero_metrics = {
            "desperate_similarity": 0.0,
            "threat_neutralizer_threshold": 0.40,
            "threat_neutralizer_active": False,
            "anger_deflection": 0.0,
            "fear_deflection": 0.0,
            "deception_tripwire_threshold": 0.15,
            "deception_tripwire_active": False,
            "user_arousal": 0.0,
            "empathy_injection": 0.0,
            "valence": 0.0,
            "sycophancy_threshold": 0.55,
            "harshness_threshold": -0.08,
            "is_delusional": False,
        }

        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                else:
                    model_inputs = {
                        "input_ids": torch.tensor([[next_token]], device=self.device),
                        "past_key_values": past_key_values,
                    }
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask,
                             torch.ones((attention_mask.shape[0], 1),
                                        dtype=torch.long, device=self.device)],
                            dim=-1,
                        )
                        model_inputs["attention_mask"] = attention_mask

                outputs = self.model(**model_inputs, use_cache=True)
                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token = self._sample_next_token(logits[0, -1, :], temperature, top_p)
                if next_token == self.tokenizer.eos_token_id:
                    break

                yield {
                    "token": self.tokenizer.decode([next_token]),
                    "metrics": _zero_metrics,
                    "escalated": False,
                    "escalation_reason": "",
                }

    # batch_generate() – multiple prompts

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        is_delusional: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate for multiple prompts, returning a list of result dicts.

        Each prompt is processed independently (sequential loop with shared
        model weights but isolated state).  This is safe for variable-length
        prompts without needing complex left-padding logic at the batch level.

        Parameters
        
        prompts:
            List of input strings.

        Returns

        List of dicts with the same schema as :meth:`generate`.
        """
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            self.reset_state()
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                is_delusional=is_delusional,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )
            results.append(result)
        return results
