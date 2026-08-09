"""完整阵列首光子 Flash dToF 仿真包。"""

from .batch import NYUBatchConfig, NYUBatchSummary, run_nyu_batch
from .config import (
    DerivedConfig,
    SceneInputs,
    SensorConfig,
    derive_config,
    make_uniform_scene,
)
from .pipeline import SimulationResult, run_simulation
from .scene import LoadedNYUScene, NYUDepthV2Loader

__all__ = [
    "DerivedConfig",
    "LoadedNYUScene",
    "NYUBatchConfig",
    "NYUBatchSummary",
    "NYUDepthV2Loader",
    "SceneInputs",
    "SensorConfig",
    "SimulationResult",
    "derive_config",
    "make_uniform_scene",
    "run_simulation",
    "run_nyu_batch",
]

__version__ = "0.3.0"
