"""向量化的逐像素理想瞬态生成。"""

from dataclasses import dataclass
import math

import numpy as np

from .config import DerivedConfig, SceneInputs, SensorConfig, validate_scene_for_sensor


@dataclass(frozen=True)
class IdealTransient:
    """每个激光周期的期望探测光子数。

    ``expected_photons_hwt`` 与 ``signal_photons_hwt`` 是 shape 为
    ``[H, W, T]`` 的 float32 数组。背景图 shape 为 ``[H, W]``，单位是
    每 bin、每周期的期望探测光子数。
    """

    expected_photons_hwt: np.ndarray
    signal_photons_hwt: np.ndarray
    background_photons_per_bin_hw: np.ndarray
    effective_signal_photons_per_pulse_hw: np.ndarray
    bin_edges_s: np.ndarray
    bin_centers_s: np.ndarray


def _erf_approx(values):
    """向量化 Abramowitz-Stegun erf 近似，最大误差约 1.5e-7。"""

    values = np.asarray(values, dtype=np.float32)
    sign = np.sign(values)
    absolute = np.abs(values)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    polynomial = (
        (
            (
                ((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                - 0.284496736
            )
            * t
            + 0.254829592
        )
        * t
    )
    return sign * (1.0 - polynomial * np.exp(-(absolute * absolute)))


def _normal_cdf(values):
    return 0.5 * (1.0 + _erf_approx(values / math.sqrt(2.0)))


def generate_ideal_transient(sensor, derived, scene):
    """生成空间变化的高斯信号，并叠加各 bin 均匀的背景。"""

    if not isinstance(sensor, SensorConfig) or not isinstance(derived, DerivedConfig):
        raise TypeError("sensor and derived must be SensorConfig and DerivedConfig")
    if not isinstance(scene, SceneInputs):
        raise TypeError("scene must be SceneInputs")
    validate_scene_for_sensor(sensor, scene, derived)

    edges = np.arange(sensor.num_time_bins + 1, dtype=np.float64) * sensor.bin_width_s
    centers = (np.arange(sensor.num_time_bins, dtype=np.float64) + 0.5) * sensor.bin_width_s
    round_trip_time_hw = (
        2.0 * scene.depth_m.astype(np.float64) / 299_792_458.0
    ).astype(np.float32)
    effective_signal_hw = (
        sensor.signal_photons_per_pulse_at_reference
        * scene.reflectivity
        * (sensor.reference_distance_m / scene.depth_m) ** 2
    ).astype(np.float32)

    signal_hwt = np.empty(derived.tensor_shape_hwt, dtype=np.float32)
    sigma = np.float32(derived.pulse_sigma_s)
    for index in range(sensor.num_time_bins):
        lower = (np.float32(edges[index]) - round_trip_time_hw) / sigma
        upper = (np.float32(edges[index + 1]) - round_trip_time_hw) / sigma
        probability_mass = np.maximum(_normal_cdf(upper) - _normal_cdf(lower), 0.0)
        signal_hwt[..., index] = effective_signal_hw * probability_mass.astype(np.float32)

    if scene.background_photons_per_bin_hw is None:
        background_hw = np.full(
            derived.image_shape_hw,
            sensor.background_photons_per_bin,
            dtype=np.float32,
        )
    else:
        background_hw = scene.background_photons_per_bin_hw
    expected_hwt = signal_hwt + background_hw[..., np.newaxis]

    return IdealTransient(
        expected_photons_hwt=np.ascontiguousarray(expected_hwt, dtype=np.float32),
        signal_photons_hwt=np.ascontiguousarray(signal_hwt, dtype=np.float32),
        background_photons_per_bin_hw=np.ascontiguousarray(background_hw, dtype=np.float32),
        effective_signal_photons_per_pulse_hw=np.ascontiguousarray(
            effective_signal_hw, dtype=np.float32
        ),
        bin_edges_s=edges,
        bin_centers_s=centers,
    )
