"""Flash dToF 的传感器、相机几何与空间场景数据契约。"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
INT32_MAX = np.iinfo(np.int32).max
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_PARAMS_PATH = WORKSPACE_ROOT / "toolbox_nyu_depth_v2" / "camera_params.m"
DEFAULT_MEASURED_IRF_PATH = WORKSPACE_ROOT / "config_IRF_global.txt"


@dataclass(frozen=True)
class SensorConfig:
    """用户设置的时间传感器、通量与瞬态响应参数。

    默认时间轴与 SP-TransientBench 一致：672 个等宽 bin、0.75 ns/bin。
    光子通量均表示经过光学损耗和 PDE 后的期望探测光子数。默认使用外部
    实测 IRF；高斯响应只作为显式后备或受控测试模式。
    """

    image_height: int = 480
    image_width: int = 640
    num_time_bins: int = 672
    bin_width_s: float = 0.75e-9
    num_laser_periods: int = 20_000

    signal_photons_per_pulse_at_reference: float = 0.05
    reference_distance_m: float = 2.5
    background_photons_per_bin: float = 1e-5

    transient_model: str = "measured_irf"
    measured_irf_path: Path = DEFAULT_MEASURED_IRF_PATH
    pulse_fwhm_s: float = 1.0e-9

    random_seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "measured_irf_path", Path(self.measured_irf_path))
        _validate_sensor_config(self)


@dataclass(frozen=True)
class CameraGeometryConfig:
    """用户设置的 NYU RGB 标定文件与固定几何语义。

    ``camera_params_path`` 指向 NYU Toolbox 的 ``camera_params.m``。
    ``depth_semantics`` 当前只允许 RGB 光轴方向轴向深度；它是防止把输入
    NPY 误当斜距的显式契约，不是可切换的近似模型。
    """

    camera_params_path: Path = DEFAULT_CAMERA_PARAMS_PATH
    calibrated_image_size_wh: tuple = (640, 480)
    depth_semantics: str = "rgb_optical_axis_z"

    def __post_init__(self):
        object.__setattr__(self, "camera_params_path", Path(self.camera_params_path))
        size = self.calibrated_image_size_wh
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in size
            )
        ):
            raise ValueError("calibrated_image_size_wh must contain positive [W, H]")
        object.__setattr__(self, "calibrated_image_size_wh", tuple(size))
        if self.depth_semantics != "rgb_optical_axis_z":
            raise ValueError("depth_semantics must be 'rgb_optical_axis_z'")


@dataclass(frozen=True)
class SceneInputs:
    """逐像素场景量。

    ``depth_z_m`` 与 ``reflectivity`` 的 shape 均为 ``[H, W]``。
    ``depth_z_m`` 是 RGB 相机坐标系中沿光轴 ``+Z`` 的米制轴向深度，不是
    斜距。反射率是 ``[0, 1]`` 的无量纲合成系数。数组会规范化为
    C-contiguous float32，且不修改调用方输入。
    """

    depth_z_m: np.ndarray
    reflectivity: np.ndarray
    background_photons_per_bin_hw: Optional[np.ndarray] = None

    def __post_init__(self):
        depth = np.ascontiguousarray(np.asarray(self.depth_z_m, dtype=np.float32))
        reflectivity = np.ascontiguousarray(
            np.asarray(self.reflectivity, dtype=np.float32)
        )
        if depth.ndim != 2:
            raise ValueError("depth_z_m must have shape [H, W]")
        if reflectivity.shape != depth.shape:
            raise ValueError("reflectivity must have the same [H, W] shape as depth_z_m")
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
            raise ValueError("depth_z_m must contain only finite values > 0 metres")
        if not np.all(np.isfinite(reflectivity)) or np.any(reflectivity < 0.0) or np.any(
            reflectivity > 1.0
        ):
            raise ValueError("reflectivity must contain finite values in [0, 1]")

        background = self.background_photons_per_bin_hw
        if background is not None:
            background = np.asarray(background, dtype=np.float32)
            if background.ndim == 0:
                background = np.full(depth.shape, float(background), dtype=np.float32)
            background = np.ascontiguousarray(background)
            if background.shape != depth.shape:
                raise ValueError(
                    "background_photons_per_bin_hw must be scalar or shape [H, W]"
                )
            if not np.all(np.isfinite(background)) or np.any(background < 0.0):
                raise ValueError("background map must contain finite values >= 0")

        object.__setattr__(self, "depth_z_m", depth)
        object.__setattr__(self, "reflectivity", reflectivity)
        object.__setattr__(self, "background_photons_per_bin_hw", background)

    @property
    def shape_hw(self):
        return self.depth_z_m.shape


@dataclass(frozen=True)
class DerivedConfig:
    """由 SensorConfig 推导出的只读时间、距离与 shape 参数。"""

    tensor_shape_hwt: tuple
    image_shape_hw: tuple
    time_window_s: float
    max_unambiguous_distance_m: float
    range_per_bin_m: float
    pulse_sigma_s: float
    hwt_float32_bytes: int
    hwt_int32_bytes: int


def _validate_sensor_config(config):
    integer_fields = (
        ("image_height", config.image_height),
        ("image_width", config.image_width),
        ("num_time_bins", config.num_time_bins),
        ("num_laser_periods", config.num_laser_periods),
    )
    for name, value in integer_fields:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))
    if config.num_laser_periods > INT32_MAX:
        raise ValueError("num_laser_periods must fit in int32 EWH counts")

    positive_fields = (
        ("bin_width_s", config.bin_width_s),
        ("reference_distance_m", config.reference_distance_m),
        ("pulse_fwhm_s", config.pulse_fwhm_s),
    )
    for name, value in positive_fields:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and > 0".format(name))

    nonnegative_fields = (
        (
            "signal_photons_per_pulse_at_reference",
            config.signal_photons_per_pulse_at_reference,
        ),
        ("background_photons_per_bin", config.background_photons_per_bin),
    )
    for name, value in nonnegative_fields:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("{} must be finite and >= 0".format(name))

    if config.transient_model not in ("measured_irf", "gaussian"):
        raise ValueError("transient_model must be 'measured_irf' or 'gaussian'")
    if config.transient_model == "measured_irf" and not str(config.measured_irf_path):
        raise ValueError("measured_irf_path is required for measured_irf mode")
    if (
        not isinstance(config.random_seed, int)
        or isinstance(config.random_seed, bool)
        or config.random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")


def derive_config(config):
    """返回不可变的时间、距离、shape 与单数组内存派生结果。"""

    _validate_sensor_config(config)
    time_window_s = config.num_time_bins * config.bin_width_s
    item_count = config.image_height * config.image_width * config.num_time_bins
    return DerivedConfig(
        tensor_shape_hwt=(config.image_height, config.image_width, config.num_time_bins),
        image_shape_hw=(config.image_height, config.image_width),
        time_window_s=time_window_s,
        max_unambiguous_distance_m=SPEED_OF_LIGHT_M_PER_S * time_window_s / 2.0,
        range_per_bin_m=SPEED_OF_LIGHT_M_PER_S * config.bin_width_s / 2.0,
        pulse_sigma_s=config.pulse_fwhm_s
        / (2.0 * math.sqrt(2.0 * math.log(2.0))),
        hwt_float32_bytes=item_count * np.dtype(np.float32).itemsize,
        hwt_int32_bytes=item_count * np.dtype(np.int32).itemsize,
    )


def validate_scene_for_sensor(sensor, scene):
    """校验场景空间 shape；斜距时间窗约束由几何换算后单独校验。"""

    if not isinstance(sensor, SensorConfig):
        raise TypeError("sensor must be SensorConfig")
    if not isinstance(scene, SceneInputs):
        raise TypeError("scene must be SceneInputs")
    if scene.shape_hw != (sensor.image_height, sensor.image_width):
        raise ValueError(
            "scene shape {} does not match sensor [H,W] {}".format(
                scene.shape_hw, (sensor.image_height, sensor.image_width)
            )
        )


def make_uniform_scene(sensor, depth_z_m, reflectivity=1.0, background=None):
    """为测试和受控实验构造轴向深度均匀的阵列场景。"""

    shape = (sensor.image_height, sensor.image_width)
    return SceneInputs(
        depth_z_m=np.full(shape, depth_z_m, dtype=np.float32),
        reflectivity=np.full(shape, reflectivity, dtype=np.float32),
        background_photons_per_bin_hw=background,
    )


def format_config(sensor, derived, scene, camera_intrinsics=None):
    """分区显示传感器输入、轴向深度输入、相机内参与派生参数。"""

    lines = [
        "USER-SET SENSOR PARAMETERS",
        "  array [H,W]                 : [{},{}]".format(
            sensor.image_height, sensor.image_width
        ),
        "  EWH bins / bin width        : {} / {:.3e} s".format(
            sensor.num_time_bins, sensor.bin_width_s
        ),
        "  laser periods               : {}".format(sensor.num_laser_periods),
        "  transient model             : {}".format(sensor.transient_model),
        "  measured IRF path           : {}".format(sensor.measured_irf_path),
        "  signal at reference         : {:.6g} detected photons/pulse".format(
            sensor.signal_photons_per_pulse_at_reference
        ),
        "  signal reference range      : {:.6f} m".format(sensor.reference_distance_m),
        "  background                  : {:.6g} detected photons/bin/pulse".format(
            sensor.background_photons_per_bin
        ),
        "  random seed                 : {}".format(sensor.random_seed),
        "SCENE ARRAY INPUTS",
        "  depth semantics             : RGB optical-axis z (not slant range)",
        "  depth z shape / dtype       : {} / {}".format(
            scene.depth_z_m.shape, scene.depth_z_m.dtype
        ),
        "  depth z min / max           : {:.6f} / {:.6f} m".format(
            float(np.min(scene.depth_z_m)), float(np.max(scene.depth_z_m))
        ),
        "DERIVED / CONSTRAINT PARAMETERS (read-only)",
        "  transient / EWH shape       : {} [H,W,T]".format(derived.tensor_shape_hwt),
        "  timing window               : {:.3e} s".format(derived.time_window_s),
        "  slant range per bin         : {:.6f} m".format(derived.range_per_bin_m),
        "  max unambiguous slant range : {:.6f} m".format(
            derived.max_unambiguous_distance_m
        ),
        "  one HWT float32/int32 array : {:.2f} / {:.2f} MiB".format(
            derived.hwt_float32_bytes / (1024.0 ** 2),
            derived.hwt_int32_bytes / (1024.0 ** 2),
        ),
    ]
    if camera_intrinsics is not None:
        lines[lines.index("SCENE ARRAY INPUTS"):lines.index("SCENE ARRAY INPUTS")] = [
            "RGB CAMERA INTRINSICS",
            "  fx / fy                     : {:.9f} / {:.9f} px".format(
                camera_intrinsics.fx, camera_intrinsics.fy
            ),
            "  cx / cy                     : {:.9f} / {:.9f} (MATLAB 1-based)".format(
                camera_intrinsics.cx, camera_intrinsics.cy
            ),
        ]
    return "\n".join(lines)
