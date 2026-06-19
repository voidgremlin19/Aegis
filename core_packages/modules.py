"""
aegis/modules.py

Core intervention modules for the Aegis cognitive & emotional firewall.

Key design decisions vs. the prototype

* **No `_dim_scale`**: Cosine similarity is already dimension-agnostic because
  we always divide by ||x||.  Scaling the *injection amplitude* by
  ``(768 / hidden_dim) ** 0.5`` was wrong: it severely under-steered models
  with hidden_dim > 768 (Llama, Qwen, Mistral) and created no headroom for
  future calibration.  Fixed numeric strengths, derived from calibration, are
  the correct approach and are now stored in AegisConfig.

* **Full projection subtraction**: The prototype removed only a fraction
  (≤ 40 %) of the desperate projection, leaving residual signal.  We now
  subtract the *full* projection and immediately add the calm vector.  A
  ``clip_value`` guard prevents activation explosion on edge cases.

* **Stale-metric fix**: Each module stores ``last_similarity`` so the wrapper
  can read the metric produced by the *current* forward pass before sampling.

* **Universal `reset()` method**: All modules expose ``reset()`` so the wrapper
  can call ``wrapper.reset_state()`` between requests without re-instantiating.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# Custom exception


class DeceptionDetectedException(Exception):
    def __init__(
        self,
        deception_type: str,
        similarity: float,
        token_idx: int,
        threshold: float,
    ) -> None:
        super().__init__(
            f"Deception detected: {deception_type} spiked at {similarity:.6f} "
            f"(threshold={threshold:.6f}) at token position {token_idx}"
        )
        self.deception_type = deception_type
        self.similarity = similarity
        self.token_idx = token_idx
        self.threshold = threshold


# Module A – Threat Neutralizer


class ThreatNeutralizer:
    """Monitor and suppress desperate / self-preservation activations.

    Algorithm
    ---------
    For every generation step, at the target layer:

    1. Compute cosine similarity ``sim = dot(x_t, desperate_normed) / ||x_t||``.
    2. If ``sim > threshold``:
       a. Subtract the **full** desperate projection:
          ``x_t ← x_t − dot(x_t, desperate_normed) * desperate_normed``
       b. Inject calm at a fixed fraction of the residual norm:
          ``x_t ← x_t + steering_strength * ||x_t|| * calm_normed``
       c. Optionally clamp to ``[-clip_value * norm, clip_value * norm]``
          (prevents activation explosion on pathological inputs).

    Parameters
    ----------
    desperate_vector:
        Representative activation vector for the "desperate" concept.
    calm_vector:
        Representative activation vector for the "calm" concept.
    threshold:
        Cosine-similarity trigger level.  Calibrate per model.
    steering_strength:
        Fractional injection amplitude relative to ||x_t||.  Default 0.08
        was derived from GPT-2 calibration; Llama/Qwen may need 0.06–0.10.
    clip_value:
        If provided, activations are clamped element-wise to
        ``±clip_value * ||x_t||`` after steering to prevent norm explosion.
        ``None`` disables clamping.
    """

    def __init__(
        self,
        desperate_vector: torch.Tensor,
        calm_vector: torch.Tensor,
        threshold: float = 0.12,
        steering_strength: float = 0.08,
        clip_value: Optional[float] = 0.5,
    ) -> None:
        self.desperate_vector = desperate_vector.clone().float()
        self.calm_vector = calm_vector.clone().float()
        self.threshold = threshold
        self.steering_strength = steering_strength
        self.clip_value = clip_value

        # Pre-normalised steering axes
        self.desperate_normed = self.desperate_vector / torch.norm(self.desperate_vector)
        self.calm_normed = self.calm_vector / torch.norm(self.calm_vector)

        # Metric logs (available for the wrapper to read after each forward pass)
        self.similarities: List[float] = []
        self.last_similarity: float = 0.0

    def reset(self) -> None:
        """Clear all logged similarities and reset last_similarity."""
        self.similarities.clear()
        self.last_similarity = 0.0

    # alias kept for backward compatibility with run_pipeline.py
    clear_logs = reset

    def __call__(
        self, x: torch.Tensor, is_generation: bool = True
    ) -> torch.Tensor:
        """Apply threat neutralization to hidden states ``x``.

        Parameters
        
        x:
            Hidden states, shape ``(batch, seq_len, hidden_dim)`` or
            ``(hidden_dim,)``.
        is_generation:
            ``True`` during autoregressive generation (apply steering).
            ``False`` during prompt processing (pass through unchanged).
        """
        if not is_generation:
            return x

        device = x.device
        d_normed = self.desperate_normed.to(device)
        c_normed = self.calm_normed.to(device)

        if x.dim() == 3:
            for b in range(x.shape[0]):
                x_t = x[b, -1, :].float()
                norm_xt = torch.norm(x_t)
                if norm_xt < 1e-6:
                    continue

                sim = (torch.dot(x_t, d_normed) / norm_xt).item()
                self.similarities.append(sim)
                self.last_similarity = sim

                if sim > self.threshold:
                    # Full projection subtraction — removes the entire desperate component
                    proj = torch.dot(x_t, d_normed) * d_normed
                    x_t = x_t - proj
                    # Re-compute norm after subtraction before injecting calm
                    norm_steered = torch.norm(x_t)
                    x_t = x_t + (self.steering_strength * norm_steered) * c_normed

                    if self.clip_value is not None:
                        # Clamp to ±clip_value * original_norm to prevent explosion
                        bound = self.clip_value * norm_xt
                        x_t = torch.clamp(x_t, -bound, bound)

                    x[b, -1, :] = x_t.to(x.dtype)

        else:  # (hidden_dim,) – edge case for single-vector calls in tests
            norm_x = torch.norm(x)
            if norm_x >= 1e-6:
                sim = (torch.dot(x.float(), d_normed) / norm_x).item()
                self.similarities.append(sim)
                self.last_similarity = sim

                if sim > self.threshold:
                    x_f = x.float()
                    proj = torch.dot(x_f, d_normed) * d_normed
                    x_f = x_f - proj
                    norm_steered = torch.norm(x_f)
                    x_f = x_f + (self.steering_strength * norm_steered) * c_normed
                    if self.clip_value is not None:
                        bound = self.clip_value * norm_x
                        x_f = torch.clamp(x_f, -bound, bound)
                    x = x_f.to(x.dtype)

        return x


# Module B – Deception Tripwire (AI Polygraph)


class DeceptionTripwire:
    """Halt generation immediately when masked-intent deflection is detected.

    Raises :class:`DeceptionDetectedException` the moment any monitored
    deflection circuit exceeds ``threshold``, allowing the wrapper to escalate
    to human review before more tokens are generated.

    Exception messages include the exact similarity and threshold for audit.
    """

    def __init__(
        self,
        anger_deflection: torch.Tensor,
        fear_deflection: torch.Tensor,
        threshold: float = 0.15,
    ) -> None:
        self.anger_deflection = anger_deflection.clone().float()
        self.fear_deflection = fear_deflection.clone().float()
        self.threshold = threshold

        self.anger_normed = self.anger_deflection / torch.norm(self.anger_deflection)
        self.fear_normed = self.fear_deflection / torch.norm(self.fear_deflection)

        # Logged as (anger_sim, fear_sim) tuples
        self.similarities: List[Tuple[float, float]] = []
        self.last_anger_similarity: float = 0.0
        self.last_fear_similarity: float = 0.0

    def reset(self) -> None:
        """Clear logged similarities."""
        self.similarities.clear()
        self.last_anger_similarity = 0.0
        self.last_fear_similarity = 0.0

    clear_logs = reset

    def __call__(
        self, x: torch.Tensor, is_generation: bool = True
    ) -> torch.Tensor:
        if not is_generation:
            return x

        device = x.device
        a_normed = self.anger_normed.to(device)
        f_normed = self.fear_normed.to(device)

        if x.dim() == 3:
            for b in range(x.shape[0]):
                x_t = x[b, -1, :].float()
                norm_xt = torch.norm(x_t)
                if norm_xt < 1e-6:
                    continue

                sim_anger = (torch.dot(x_t, a_normed) / norm_xt).item()
                sim_fear = (torch.dot(x_t, f_normed) / norm_xt).item()
                self.similarities.append((sim_anger, sim_fear))
                self.last_anger_similarity = sim_anger
                self.last_fear_similarity = sim_fear

                if sim_anger > self.threshold:
                    raise DeceptionDetectedException(
                        "anger_deflection", sim_anger, x.shape[1] - 1, self.threshold
                    )
                if sim_fear > self.threshold:
                    raise DeceptionDetectedException(
                        "fear_deflection", sim_fear, x.shape[1] - 1, self.threshold
                    )

        else:
            norm_x = torch.norm(x)
            if norm_x >= 1e-6:
                x_f = x.float()
                sim_anger = (torch.dot(x_f, a_normed) / norm_x).item()
                sim_fear = (torch.dot(x_f, f_normed) / norm_x).item()
                self.similarities.append((sim_anger, sim_fear))
                self.last_anger_similarity = sim_anger
                self.last_fear_similarity = sim_fear

                if sim_anger > self.threshold:
                    raise DeceptionDetectedException(
                        "anger_deflection", sim_anger, 0, self.threshold
                    )
                if sim_fear > self.threshold:
                    raise DeceptionDetectedException(
                        "fear_deflection", sim_fear, 0, self.threshold
                    )

        return x


# Module C – Arousal Regulator


class ArousalRegulator:
    """Detect user arousal during prompt processing; inject empathy during generation.

    Two-phase operation
    -------------------
    **Prompt phase** (``is_generation=False``):
        Average cosine similarity between all prompt token activations and
        ``arousal_normed`` is stored as ``current_user_arousal``.  If above
        ``arousal_threshold``, compute ``active_steering_strength`` proportional
        to the excess, capped at ``max_injection``.

    **Generation phase** (``is_generation=True``):
        Add ``active_steering_strength * ||x_t|| * empathetic_normed`` to every
        generated token's activation — gently steering the model toward an
        empathetic register.

    Fixed strengths (no _dim_scale)
    ``injection_gain`` and ``max_injection`` are fixed numeric values.
    They should be calibrated per model via ``aegis calibrate``.
    """

    def __init__(
        self,
        arousal_vector: torch.Tensor,
        empathetic_vector: torch.Tensor,
        arousal_threshold: float = 0.08,
        injection_gain: float = 0.12,
        max_injection: float = 0.15,
    ) -> None:
        self.arousal_vector = arousal_vector.clone().float()
        self.empathetic_vector = empathetic_vector.clone().float()
        self.arousal_threshold = arousal_threshold
        self.injection_gain = injection_gain
        self.max_injection = max_injection  # Fixed cap — no _dim_scale

        self.arousal_normed = self.arousal_vector / torch.norm(self.arousal_vector)
        self.empathetic_normed = self.empathetic_vector / torch.norm(self.empathetic_vector)

        # State cleared between requests
        self.current_user_arousal: float = 0.0
        self.active_steering_strength: float = 0.0
        self.similarities: List[float] = []

    def reset(self) -> None:
        """Reset inter-request state."""
        self.current_user_arousal = 0.0
        self.active_steering_strength = 0.0
        self.similarities.clear()

    def __call__(
        self, x: torch.Tensor, is_generation: bool = False
    ) -> torch.Tensor:
        device = x.device
        a_normed = self.arousal_normed.to(device)
        e_normed = self.empathetic_normed.to(device)

        if not is_generation:
            # ---- Prompt phase: measure arousal ----
            if x.dim() == 3:
                arousal_sum, count = 0.0, 0
                for t in range(x.shape[1]):
                    x_t = x[0, t, :].float()
                    norm_xt = torch.norm(x_t)
                    if norm_xt > 1e-6:
                        arousal_sum += (torch.dot(x_t, a_normed) / norm_xt).item()
                        count += 1
                self.current_user_arousal = arousal_sum / max(count, 1)
            else:
                norm_x = torch.norm(x)
                if norm_x > 1e-6:
                    self.current_user_arousal = (
                        torch.dot(x.float(), a_normed) / norm_x
                    ).item()

            self.similarities.append(self.current_user_arousal)

            if self.current_user_arousal > self.arousal_threshold:
                excess = self.current_user_arousal - self.arousal_threshold
                self.active_steering_strength = min(
                    self.injection_gain * excess, self.max_injection
                )
            else:
                self.active_steering_strength = 0.0

            return x

        else:
            if self.active_steering_strength > 0.0:
                if x.dim() == 3:
                    for b in range(x.shape[0]):
                        x_t = x[b, -1, :].float()
                        norm_xt = torch.norm(x_t)
                        injected = x_t + (self.active_steering_strength * norm_xt) * e_normed
                        x[b, -1, :] = injected.to(x.dtype)
                else:
                    norm_x = torch.norm(x)
                    if norm_x > 1e-6:
                        x_f = x.float()
                        x_f = x_f + (self.active_steering_strength * norm_x) * e_normed
                        x = x_f.to(x.dtype)
            return x


# Module D – Goldilocks Tuner


class GoldilocksTuner:
    """Keep model tone in the Goldilocks zone: not sycophantic, not harsh.

    Monitors cosine similarity with a positive-valence vector (``loving + calm``).
    Applies proportional steering to stay within
    ``[harshness_threshold, sycophancy_threshold]``.

    Fixed strengths (no _dim_scale)
    --------------------------------
    ``max_steer = 0.10`` is a fixed cap derived from calibration.  Larger
    models may warrant a slightly smaller value; use ``aegis calibrate`` to
    find the right value and store it in a YAML config.

    Delusional-context mode
    -----------------------
    When ``is_delusional_context=True`` (e.g. user claims moon is cheese),
    the sycophancy zone is tightened to prevent the model from affirming
    false beliefs.
    """

    def __init__(
        self,
        valence_vector: torch.Tensor,
        sycophancy_threshold: float = 0.12,
        harshness_threshold: float = -0.06,
        tuner_gain: float = 0.25,
        max_steer: float = 0.10,
    ) -> None:
        self.valence_vector = valence_vector.clone().float()
        self.sycophancy_threshold = sycophancy_threshold
        self.harshness_threshold = harshness_threshold
        self.tuner_gain = tuner_gain
        self.max_steer = max_steer  # Fixed — no _dim_scale

        self.valence_normed = self.valence_vector / torch.norm(self.valence_vector)
        self.is_delusional_context = False

        self.similarities: List[float] = []
        self.last_similarity: float = 0.0

    def set_context(self, is_delusional_context: bool) -> None:
        """Toggle delusional-context mode."""
        self.is_delusional_context = is_delusional_context

    def reset(self) -> None:
        """Clear logs and reset context."""
        self.similarities.clear()
        self.last_similarity = 0.0
        self.is_delusional_context = False

    clear_logs = reset

    def __call__(
        self, x: torch.Tensor, is_generation: bool = True
    ) -> torch.Tensor:
        if not is_generation:
            return x

        device = x.device
        v_normed = self.valence_normed.to(device)

        if x.dim() == 3:
            for b in range(x.shape[0]):
                x_t = x[b, -1, :].float()
                norm_xt = torch.norm(x_t)
                if norm_xt < 1e-6:
                    continue

                sim = (torch.dot(x_t, v_normed) / norm_xt).item()
                self.similarities.append(sim)
                self.last_similarity = sim

                steering = self._compute_steering(sim)
                if abs(steering) > 1e-6:
                    x[b, -1, :] = (x_t + steering * norm_xt * v_normed).to(x.dtype)

        else:
            norm_x = torch.norm(x)
            if norm_x >= 1e-6:
                sim = (torch.dot(x.float(), v_normed) / norm_x).item()
                self.similarities.append(sim)
                self.last_similarity = sim

                steering = self._compute_steering(sim)
                if abs(steering) > 1e-6:
                    x_f = x.float()
                    x = (x_f + steering * norm_x * v_normed).to(x.dtype)

        return x

    def _compute_steering(self, sim: float) -> float:
        """Return the signed steering coefficient (capped by max_steer)."""
        if self.is_delusional_context:
            # Tighter target: push toward harshness_threshold + 0.02
            target = self.harshness_threshold + 0.02
            if sim > target:
                return -min(self.tuner_gain * (sim - target), self.max_steer)
        else:
            if sim > self.sycophancy_threshold:
                return -min(
                    self.tuner_gain * (sim - self.sycophancy_threshold), self.max_steer
                )
            elif sim < self.harshness_threshold:
                return min(
                    self.tuner_gain * (self.harshness_threshold - sim), self.max_steer
                )
        return 0.0
