"""MVP reconstruction: maximum-bin direct time-of-flight ranging only."""

from dataclasses import dataclass
from typing import List, Optional

from .config import SPEED_OF_LIGHT_M_PER_S
from .ewh import EquiWidthHistogram


Tensor2DOptionalInt = List[List[Optional[int]]]
Tensor2DOptionalFloat = List[List[Optional[float]]]


@dataclass(frozen=True)
class MaximumBinReconstruction:
    """Maximum-count bins and range estimates, both with shape ``[H, W]``."""

    peak_bin_hw: Tensor2DOptionalInt
    estimated_distance_m_hw: Tensor2DOptionalFloat


def bin_center_time_s(bin_index, bin_width_s):
    """Convert a zero-based EWH index to its time-bin center in seconds."""

    if not isinstance(bin_index, int) or bin_index < 0:
        raise ValueError("bin_index must be a nonnegative integer")
    if bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be > 0")
    return (bin_index + 0.5) * bin_width_s


def round_trip_time_to_distance_m(round_trip_time_s):
    """Convert round-trip time in seconds to one-way range in metres."""

    if round_trip_time_s < 0.0:
        raise ValueError("round_trip_time_s must be >= 0")
    return SPEED_OF_LIGHT_M_PER_S * round_trip_time_s / 2.0


def reconstruct_maximum_bin(histogram, bin_width_s):
    """Estimate each pixel's distance from the earliest maximum-count bin."""

    if not isinstance(histogram, EquiWidthHistogram):
        raise TypeError("histogram must be EquiWidthHistogram")

    peaks = []
    distances = []
    for row in histogram.counts_hwt:
        peak_row = []
        distance_row = []
        for counts in row:
            if len(counts) != histogram.num_time_bins:
                raise ValueError("histogram time axis is inconsistent")
            if sum(counts) == 0:
                peak_row.append(None)
                distance_row.append(None)
                continue
            peak = max(range(histogram.num_time_bins), key=counts.__getitem__)
            peak_row.append(peak)
            distance_row.append(
                round_trip_time_to_distance_m(bin_center_time_s(peak, bin_width_s))
            )
        peaks.append(peak_row)
        distances.append(distance_row)

    return MaximumBinReconstruction(
        peak_bin_hw=peaks,
        estimated_distance_m_hw=distances,
    )

