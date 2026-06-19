"""
Project Aegis – Cognitive & Emotional Firewall for LLMs.

A production-ready, pip-installable package that intercepts and steers
large language model residual-stream activations in real time to:
  • Neutralize self-preservation / desperation patterns
  • Detect masked alignment-faking (deception tripwire)
  • Regulate conversational arousal (empathy injection)
  • Bound sycophancy and harshness (Goldilocks tuner)
"""

from core_packages.modules import (
    ThreatNeutralizer,
    DeceptionTripwire,
    ArousalRegulator,
    GoldilocksTuner,
    DeceptionDetectedException,
)
from core_packages.vector_engine import VectorEngine
from core_packages.model_wrapper import AegisModelWrapper
from core_packages.config import AegisConfig, load_config, save_config, get_default_config

__version__ = "1.0.0"
__author__ = "Sakshi Dhatrak"

__all__ = [
    "ThreatNeutralizer",
    "DeceptionTripwire",
    "ArousalRegulator",
    "GoldilocksTuner",
    "DeceptionDetectedException",
    "VectorEngine",
    "AegisModelWrapper",
    "AegisConfig",
    "load_config",
    "save_config",
    "get_default_config",
]
