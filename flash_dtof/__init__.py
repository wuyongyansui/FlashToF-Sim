"""Minimal first-photon Flash dToF simulation package."""

from .config import DerivedConfig, UserConfig, derive_config
from .pipeline import SimulationResult, run_simulation

__all__ = [
    "DerivedConfig",
    "SimulationResult",
    "UserConfig",
    "derive_config",
    "run_simulation",
]

__version__ = "0.1.0"

