"""端到端完整阵列首光子 Flash dToF 链路。"""

from dataclasses import dataclass

import numpy as np

from .config import DerivedConfig, SceneInputs, SensorConfig, derive_config
from .ewh import EquiWidthHistogram, accumulate_ewh
from .first_photon import sample_first_photon_counts
from .reconstruction import MaximumBinReconstruction, reconstruct_maximum_bin
from .transient import IdealTransient, generate_ideal_transient


@dataclass(frozen=True)
class ArrayDiagnostics:
    """逐像素诊断图，所有数组的 shape 均为 ``[H, W]``。"""

    ideal_peak_bin_hw: np.ndarray
    observed_peak_bin_hw: np.ndarray
    peak_shift_bins_hw: np.ndarray
    measured_detection_fraction_hw: np.ndarray
    expected_detection_fraction_hw: np.ndarray
    distance_bias_m_hw: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    sensor_config: SensorConfig
    derived_config: DerivedConfig
    scene_inputs: SceneInputs
    ideal_transient: IdealTransient
    ewh: EquiWidthHistogram
    reconstruction: MaximumBinReconstruction
    diagnostics: ArrayDiagnostics


def run_simulation(sensor_config, scene_inputs):
    """运行固定阵列链路，不加入校正或额外传感器效应。"""

    derived = derive_config(sensor_config)
    transient = generate_ideal_transient(sensor_config, derived, scene_inputs)
    aggregate = sample_first_photon_counts(
        transient.expected_photons_hwt,
        sensor_config.num_laser_periods,
        sensor_config.random_seed,
    )
    histogram = accumulate_ewh(aggregate)
    reconstruction = reconstruct_maximum_bin(histogram, sensor_config.bin_width_s)

    ideal_peak = np.argmax(transient.expected_photons_hwt, axis=-1).astype(np.int16)
    measured_detection = (
        histogram.detected_counts_hw.astype(np.float64)
        / sensor_config.num_laser_periods
    ).astype(np.float32)
    total_expected = np.sum(transient.expected_photons_hwt, axis=-1, dtype=np.float64)
    expected_detection = (-np.expm1(-total_expected)).astype(np.float32)
    peak_shift = np.full(ideal_peak.shape, np.nan, dtype=np.float32)
    peak_shift[reconstruction.valid_hw] = (
        reconstruction.peak_bin_hw[reconstruction.valid_hw]
        - ideal_peak[reconstruction.valid_hw]
    )
    distance_bias = reconstruction.estimated_distance_m_hw - scene_inputs.depth_m

    diagnostics = ArrayDiagnostics(
        ideal_peak_bin_hw=ideal_peak,
        observed_peak_bin_hw=reconstruction.peak_bin_hw,
        peak_shift_bins_hw=peak_shift,
        measured_detection_fraction_hw=measured_detection,
        expected_detection_fraction_hw=expected_detection,
        distance_bias_m_hw=distance_bias.astype(np.float32),
    )
    return SimulationResult(
        sensor_config=sensor_config,
        derived_config=derived,
        scene_inputs=scene_inputs,
        ideal_transient=transient,
        ewh=histogram,
        reconstruction=reconstruction,
        diagnostics=diagnostics,
    )


def format_diagnostics(result):
    """返回简洁的完整阵列通量、探测、pile-up 与距离汇总。"""

    valid = result.reconstruction.valid_hw
    diagnostics = result.diagnostics
    centre = (
        result.sensor_config.image_height // 2,
        result.sensor_config.image_width // 2,
    )
    valid_fraction = float(np.mean(valid))
    if np.any(valid):
        bias = diagnostics.distance_bias_m_hw[valid]
        shift = diagnostics.peak_shift_bins_hw[valid]
        bias_mean = float(np.mean(bias))
        rmse = float(np.sqrt(np.mean(bias.astype(np.float64) ** 2)))
        shift_summary = "mean={:+.3f}, min={:+.0f}, max={:+.0f}".format(
            float(np.mean(shift)), float(np.min(shift)), float(np.max(shift))
        )
    else:
        bias_mean = float("nan")
        rmse = float("nan")
        shift_summary = "no valid pixels"

    return "\n".join(
        [
            "FULL-ARRAY SIMULATION DIAGNOSTICS",
            "  model                        : earliest photon per pixel per period",
            "  EWH shape                    : {}".format(result.ewh.counts_hwt.shape),
            "  EWH dtype / memory           : {} / {:.2f} MiB".format(
                result.ewh.counts_hwt.dtype,
                result.ewh.counts_hwt.nbytes / (1024.0 ** 2),
            ),
            "  valid pixel fraction         : {:.6f}".format(valid_fraction),
            "  detection fraction mean      : {:.6f} (expected {:.6f})".format(
                float(np.mean(diagnostics.measured_detection_fraction_hw)),
                float(np.mean(diagnostics.expected_detection_fraction_hw)),
            ),
            "  peak shift bins               : {}".format(shift_summary),
            "  distance bias mean / RMSE    : {:+.6f} / {:.6f} m".format(
                bias_mean, rmse
            ),
            "  centre pixel [r,c]           : {}".format(centre),
            "    true / estimated distance   : {:.6f} / {:.6f} m".format(
                float(result.scene_inputs.depth_m[centre]),
                float(result.reconstruction.estimated_distance_m_hw[centre]),
            ),
            "    ideal / observed peak bin   : {} / {}".format(
                int(diagnostics.ideal_peak_bin_hw[centre]),
                int(diagnostics.observed_peak_bin_hw[centre]),
            ),
            "    detected / no-detection     : {} / {}".format(
                int(result.ewh.detected_counts_hw[centre]),
                int(result.ewh.no_detection_counts_hw[centre]),
            ),
        ]
    )
