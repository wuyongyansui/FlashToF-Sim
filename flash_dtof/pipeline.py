"""End-to-end orchestration and diagnostics for the fixed MVP chain."""

from dataclasses import dataclass
import math
from typing import List, Optional

from .config import DerivedConfig, UserConfig, derive_config
from .ewh import EquiWidthHistogram, accumulate_ewh
from .first_photon import FirstPhotonSamples, sample_first_photons
from .reconstruction import MaximumBinReconstruction, reconstruct_maximum_bin
from .transient import IdealTransient, generate_ideal_transient


@dataclass(frozen=True)
class PixelDiagnostics:
    """Diagnostics for one pixel; bin shifts are observed minus ideal."""

    row: int
    column: int
    ideal_peak_bin: int
    observed_peak_bin: Optional[int]
    peak_shift_bins: Optional[int]
    detected_periods: int
    no_detection_periods: int
    measured_detection_fraction: float
    expected_detection_fraction: float
    estimated_distance_m: Optional[float]
    distance_bias_m: Optional[float]


@dataclass(frozen=True)
class SimulationResult:
    user_config: UserConfig
    derived_config: DerivedConfig
    ideal_transient: IdealTransient
    first_photon_samples: FirstPhotonSamples
    ewh: EquiWidthHistogram
    reconstruction: MaximumBinReconstruction
    diagnostics: List[PixelDiagnostics]


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def run_simulation(user_config):
    """Run the fixed MVP chain without correction or extra sensor effects."""

    derived = derive_config(user_config)
    transient = generate_ideal_transient(user_config, derived)
    samples = sample_first_photons(
        transient,
        user_config.num_laser_periods,
        user_config.random_seed,
    )
    histogram = accumulate_ewh(samples, user_config.num_time_bins)
    reconstruction = reconstruct_maximum_bin(histogram, user_config.bin_width_s)

    diagnostics = []
    for row_index in range(user_config.image_height):
        for column_index in range(user_config.image_width):
            ideal_profile = transient.expected_photons_hwt[row_index][column_index]
            ideal_peak = _argmax(ideal_profile)
            observed_peak = reconstruction.peak_bin_hw[row_index][column_index]
            no_detection = histogram.no_detection_counts_hw[row_index][column_index]
            detected = user_config.num_laser_periods - no_detection
            total_expected = sum(ideal_profile)
            estimated_distance = reconstruction.estimated_distance_m_hw[row_index][
                column_index
            ]
            diagnostics.append(
                PixelDiagnostics(
                    row=row_index,
                    column=column_index,
                    ideal_peak_bin=ideal_peak,
                    observed_peak_bin=observed_peak,
                    peak_shift_bins=(
                        None if observed_peak is None else observed_peak - ideal_peak
                    ),
                    detected_periods=detected,
                    no_detection_periods=no_detection,
                    measured_detection_fraction=(
                        float(detected) / user_config.num_laser_periods
                    ),
                    expected_detection_fraction=-math.expm1(-total_expected),
                    estimated_distance_m=estimated_distance,
                    distance_bias_m=(
                        None
                        if estimated_distance is None
                        else estimated_distance - user_config.distance_m
                    ),
                )
            )

    return SimulationResult(
        user_config=user_config,
        derived_config=derived,
        ideal_transient=transient,
        first_photon_samples=samples,
        ewh=histogram,
        reconstruction=reconstruction,
        diagnostics=diagnostics,
    )


def format_diagnostics(result):
    """Format reproducibility, flux, detection, pile-up, and range diagnostics."""

    lines = [
        "SIMULATION DIAGNOSTICS",
        "  seed                         : {}".format(result.user_config.random_seed),
        "  model                        : first photon per pixel per laser period",
        "  signal / background flux     : {:.6g} / {:.6g} detected photons per pulse".format(
            result.derived_config.effective_signal_photons_per_pulse,
            result.derived_config.expected_background_photons_per_pulse,
        ),
    ]
    for diagnostic in result.diagnostics:
        distance = (
            "none"
            if diagnostic.estimated_distance_m is None
            else "{:.6f} m".format(diagnostic.estimated_distance_m)
        )
        bias = (
            "none"
            if diagnostic.distance_bias_m is None
            else "{:+.6f} m".format(diagnostic.distance_bias_m)
        )
        lines.extend(
            [
                "  pixel [{},{}]".format(diagnostic.row, diagnostic.column),
                "    detections / periods       : {} / {}".format(
                    diagnostic.detected_periods,
                    result.user_config.num_laser_periods,
                ),
                "    detection fraction meas/exp: {:.6f} / {:.6f}".format(
                    diagnostic.measured_detection_fraction,
                    diagnostic.expected_detection_fraction,
                ),
                "    ideal / observed peak bin  : {} / {}".format(
                    diagnostic.ideal_peak_bin,
                    diagnostic.observed_peak_bin,
                ),
                "    observed - ideal shift     : {} bins".format(
                    diagnostic.peak_shift_bins
                ),
                "    max-bin distance / bias    : {} / {}".format(distance, bias),
            ]
        )
    return "\n".join(lines)

