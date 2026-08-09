"""Flash dToF 的传感器配置与空间场景数据契约。"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True)
class SensorConfig:
    """用户设置的传感器与采集参数。

    默认值描述首个完整阵列目标：120 行、240 列、190 个等宽 bin，bin 宽
    0.5 ns。光子通量字段均表示经过光学损耗和 PDE 后的期望探测光子数。
    参考信号以场景反射率 1.0 归一化，参考反射率不是独立模型参数。
    """

    image_height: int = 120
    image_width: int = 240
    num_time_bins: int = 190
    bin_width_s: float = 0.5e-9
    num_laser_periods: int = 20_000

    signal_photons_per_pulse_at_reference: float = 0.05
    reference_distance_m: float = 2.5
    background_photons_per_bin: float = 1e-5
    pulse_fwhm_s: float = 1.0e-9

    random_seed: int = 0

    def __post_init__(self):
        _validate_sensor_config(self)


@dataclass(frozen=True)
class SceneInputs:
    """逐像素场景量。

    ``depth_m`` 与 ``reflectivity`` 的 shape 均为 ``[H, W]``。深度表示米制
    单程距离；反射率是 ``[0, 1]`` 范围内的无量纲合成系数。可选背景图会
    覆盖传感器中的标量背景值。数组会规范化为 C-contiguous float32，且不
    修改调用方输入。
    """

    depth_m: np.ndarray
    reflectivity: np.ndarray
    background_photons_per_bin_hw: Optional[np.ndarray] = None

    def __post_init__(self):
        depth = np.ascontiguousarray(np.asarray(self.depth_m, dtype=np.float32))
        reflectivity = np.ascontiguousarray(
            np.asarray(self.reflectivity, dtype=np.float32)
        )
        if depth.ndim != 2:
            raise ValueError("depth_m must have shape [H, W]")
        if reflectivity.shape != depth.shape:
            raise ValueError("reflectivity must have the same [H, W] shape as depth_m")
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
            raise ValueError("depth_m must contain only finite values > 0 metres")
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

        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "reflectivity", reflectivity)
        object.__setattr__(self, "background_photons_per_bin_hw", background)

    @property
    def shape_hw(self):
        return self.depth_m.shape


@dataclass(frozen=True)
class DerivedConfig:
    """由 SensorConfig 推导出的只读时间、距离与 shape 参数。"""

    tensor_shape_hwt: tuple
    image_shape_hw: tuple
    time_window_s: float
    max_unambiguous_distance_m: float
    range_per_bin_m: float
    pulse_sigma_s: float


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

    if (
        not isinstance(config.random_seed, int)
        or isinstance(config.random_seed, bool)
        or config.random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")


def derive_config(config):
    """返回不可变的时间、距离与 shape 派生结果。"""

    _validate_sensor_config(config)
    time_window_s = config.num_time_bins * config.bin_width_s
    return DerivedConfig(
        tensor_shape_hwt=(config.image_height, config.image_width, config.num_time_bins),
        image_shape_hw=(config.image_height, config.image_width),
        time_window_s=time_window_s,
        max_unambiguous_distance_m=SPEED_OF_LIGHT_M_PER_S * time_window_s / 2.0,
        range_per_bin_m=SPEED_OF_LIGHT_M_PER_S * config.bin_width_s / 2.0,
        pulse_sigma_s=config.pulse_fwhm_s
        / (2.0 * math.sqrt(2.0 * math.log(2.0))),
    )


def validate_scene_for_sensor(sensor, scene, derived=None):
    """校验场景空间 shape 与传感器时间窗是否兼容。"""

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
    if derived is None:
        derived = derive_config(sensor)
    if float(np.max(scene.depth_m)) >= derived.max_unambiguous_distance_m:
        raise ValueError(
            "scene contains depth outside the {:.6f} m unambiguous range".format(
                derived.max_unambiguous_distance_m
            )
        )


def make_uniform_scene(sensor, distance_m, reflectivity=1.0, background=None):
    """为测试和受控实验构造空间均匀的阵列场景。"""

    shape = (sensor.image_height, sensor.image_width)
    return SceneInputs(
        depth_m=np.full(shape, distance_m, dtype=np.float32),
        reflectivity=np.full(shape, reflectivity, dtype=np.float32),
        background_photons_per_bin_hw=background,
    )


def format_config(sensor, derived, scene):
    """分区显示可编辑传感器输入、场景输入与派生参数。"""

    return "\n".join(
        [
            "USER-SET SENSOR PARAMETERS",
            "  array [H,W]                 : [{},{}]".format(
                sensor.image_height, sensor.image_width
            ),
            "  EWH bins / bin width        : {} / {:.3e} s".format(
                sensor.num_time_bins, sensor.bin_width_s
            ),
            "  laser periods               : {}".format(sensor.num_laser_periods),
            "  signal at reference         : {:.6g} detected photons/pulse".format(
                sensor.signal_photons_per_pulse_at_reference
            ),
            "  signal reference distance   : {:.6f} m".format(
                sensor.reference_distance_m
            ),
            "  background                  : {:.6g} detected photons/bin/pulse".format(
                sensor.background_photons_per_bin
            ),
            "  pulse FWHM                  : {:.3e} s".format(sensor.pulse_fwhm_s),
            "  random seed                 : {}".format(sensor.random_seed),
            "SCENE ARRAY INPUTS",
            "  depth shape / dtype         : {} / {}".format(
                scene.depth_m.shape, scene.depth_m.dtype
            ),
            "  depth min / max             : {:.6f} / {:.6f} m".format(
                float(np.min(scene.depth_m)), float(np.max(scene.depth_m))
            ),
            "  reflectivity min / max      : {:.6f} / {:.6f}".format(
                float(np.min(scene.reflectivity)), float(np.max(scene.reflectivity))
            ),
            "DERIVED / CONSTRAINT PARAMETERS (read-only)",
            "  transient / EWH shape       : {} [H,W,T]".format(
                derived.tensor_shape_hwt
            ),
            "  timing window               : {:.3e} s".format(derived.time_window_s),
            "  range per bin               : {:.6f} m".format(derived.range_per_bin_m),
            "  unambiguous range           : {:.6f} m".format(
                derived.max_unambiguous_distance_m
            ),
        ]
    )
