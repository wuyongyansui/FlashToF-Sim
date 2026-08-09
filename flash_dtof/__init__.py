"""完整阵列首光子 Flash dToF 仿真包。"""

from .batch import NYUBatchConfig, NYUBatchSummary, run_nyu_batch
from .config import (
    CameraGeometryConfig,
    DerivedConfig,
    SceneInputs,
    SensorConfig,
    derive_config,
    make_uniform_scene,
)
from .geometry import RGBIntrinsics, SceneGeometry, load_nyu_rgb_intrinsics
from .irf import MeasuredIRF, load_measured_irf
from .pipeline import SimulationResult, run_simulation
from .output import OutputConfig
from .scene import LoadedNYUScene, NYUDepthV2Loader

__all__ = [
    "CameraGeometryConfig",
    "DerivedConfig",
    "LoadedNYUScene",
    "MeasuredIRF",
    "NYUBatchConfig",
    "NYUBatchSummary",
    "NYUDepthV2Loader",
    "OutputConfig",
    "RGBIntrinsics",
    "SceneInputs",
    "SceneGeometry",
    "SensorConfig",
    "SimulationResult",
    "derive_config",
    "load_measured_irf",
    "load_nyu_rgb_intrinsics",
    "make_uniform_scene",
    "run_simulation",
    "run_nyu_batch",
]

__version__ = "0.6.0"
