"""使用逐像素斜距和实测或高斯时间响应生成理想到达率。"""

from dataclasses import dataclass
import math

import numpy as np

from .config import DerivedConfig, SceneInputs, SensorConfig, validate_scene_for_sensor
from .geometry import SceneGeometry
from .irf import MeasuredIRF, load_measured_irf, shifted_irf_mass_at_bin_centers


@dataclass(frozen=True)
class IdealTransient:
    """每个激光周期的期望探测光子数。

    ``expected_photons_hwt`` 是唯一保留的完整 ``[H,W,T]`` float32 数组，
    其中已包含信号和背景。为控制原生 640×480×672 内存，不再额外保留一
    份同尺寸信号数组；信号总量通过两个 ``[H,W]`` 诊断图暴露。
    """

    expected_photons_hwt: np.ndarray
    background_photons_per_bin_hw: np.ndarray
    effective_signal_photons_per_pulse_hw: np.ndarray
    recorded_signal_photons_per_pulse_hw: np.ndarray
    round_trip_time_s_hw: np.ndarray
    bin_edges_s: np.ndarray
    bin_centers_s: np.ndarray
    response_model: str
    measured_irf: object


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


def generate_ideal_transient(
    sensor,
    derived,
    scene,
    scene_geometry,
    measured_irf=None,
):
    """按真实斜距生成空间变化信号，并叠加逐 bin 均匀背景。

    ToF 和 ``1/r^2`` 通量缩放都使用 ``scene_geometry.slant_range_m_hw``。
    ``scene.depth_z_m`` 只作为相机轴向深度输入，不直接进入时间或距离衰减。
    """

    if not isinstance(sensor, SensorConfig) or not isinstance(derived, DerivedConfig):
        raise TypeError("sensor and derived must be SensorConfig and DerivedConfig")
    if not isinstance(scene, SceneInputs):
        raise TypeError("scene must be SceneInputs")
    if not isinstance(scene_geometry, SceneGeometry):
        raise TypeError("scene_geometry must be SceneGeometry")
    validate_scene_for_sensor(sensor, scene)
    if scene_geometry.slant_range_m_hw.shape != scene.shape_hw:
        raise ValueError("scene geometry must have shape [H,W] matching the scene")
    max_range = float(np.max(scene_geometry.slant_range_m_hw))
    if max_range >= derived.max_unambiguous_distance_m:
        raise ValueError(
            "scene contains slant range outside the {:.6f} m unambiguous range".format(
                derived.max_unambiguous_distance_m
            )
        )

    edges = np.arange(sensor.num_time_bins + 1, dtype=np.float64) * sensor.bin_width_s
    centers = (np.arange(sensor.num_time_bins, dtype=np.float64) + 0.5) * sensor.bin_width_s
    slant_range = scene_geometry.slant_range_m_hw.astype(np.float64)
    round_trip_time = 2.0 * slant_range / 299_792_458.0
    effective_signal = (
        sensor.signal_photons_per_pulse_at_reference
        * scene.reflectivity.astype(np.float64)
        * (sensor.reference_distance_m / slant_range) ** 2
    ).astype(np.float32)

    used_irf = None
    if sensor.transient_model == "measured_irf":
        if measured_irf is None:
            measured_irf = load_measured_irf(
                sensor.measured_irf_path,
                expected_bin_width_s=sensor.bin_width_s,
            )
        if not isinstance(measured_irf, MeasuredIRF):
            raise TypeError("measured_irf must be MeasuredIRF in measured_irf mode")
        if not np.isclose(
            measured_irf.sample_interval_s,
            sensor.bin_width_s,
            rtol=1e-9,
            atol=1e-18,
        ):
            raise ValueError("measured IRF sample interval must equal sensor bin width")
        signal_hwt = shifted_irf_mass_at_bin_centers(
            measured_irf,
            round_trip_time,
            centers,
        )
        signal_hwt *= effective_signal[..., np.newaxis]
        used_irf = measured_irf
    else:
        signal_hwt = np.empty(derived.tensor_shape_hwt, dtype=np.float32)
        sigma = np.float32(derived.pulse_sigma_s)
        round_trip_time_f32 = round_trip_time.astype(np.float32)
        for index in range(sensor.num_time_bins):
            lower = (np.float32(edges[index]) - round_trip_time_f32) / sigma
            upper = (np.float32(edges[index + 1]) - round_trip_time_f32) / sigma
            probability_mass = np.maximum(_normal_cdf(upper) - _normal_cdf(lower), 0.0)
            signal_hwt[..., index] = (
                effective_signal * probability_mass.astype(np.float32)
            )

    recorded_signal = np.sum(signal_hwt, axis=-1, dtype=np.float64).astype(np.float32)
    if scene.background_photons_per_bin_hw is None:
        background_hw = np.full(
            derived.image_shape_hw,
            sensor.background_photons_per_bin,
            dtype=np.float32,
        )
    else:
        background_hw = scene.background_photons_per_bin_hw
    signal_hwt += background_hw[..., np.newaxis]

    return IdealTransient(
        expected_photons_hwt=np.ascontiguousarray(signal_hwt, dtype=np.float32),
        background_photons_per_bin_hw=np.ascontiguousarray(background_hw, dtype=np.float32),
        effective_signal_photons_per_pulse_hw=np.ascontiguousarray(
            effective_signal, dtype=np.float32
        ),
        recorded_signal_photons_per_pulse_hw=np.ascontiguousarray(
            recorded_signal, dtype=np.float32
        ),
        round_trip_time_s_hw=np.ascontiguousarray(round_trip_time, dtype=np.float64),
        bin_edges_s=edges,
        bin_centers_s=centers,
        response_model=sensor.transient_model,
        measured_irf=used_irf,
    )
