"""Equi-width histogram accumulation from first-photon outcomes."""

from dataclasses import dataclass
from typing import List

from .first_photon import FirstPhotonSamples, NO_DETECTION


Tensor3DInt = List[List[List[int]]]
Tensor2DInt = List[List[int]]


@dataclass(frozen=True)
class EquiWidthHistogram:
    """Accumulated EWH counts.

    ``counts_hwt`` has shape ``[H, W, T]`` and units of detected first-photon
    events per bin across ``num_laser_periods``. For every pixel, histogram
    counts plus ``no_detection_counts_hw`` equals the number of periods.
    """

    counts_hwt: Tensor3DInt
    no_detection_counts_hw: Tensor2DInt
    num_laser_periods: int
    num_time_bins: int


def accumulate_ewh(samples, num_time_bins):
    """Accumulate one bin at most per laser period into an EWH."""

    if not isinstance(samples, FirstPhotonSamples):
        raise TypeError("samples must be FirstPhotonSamples")
    if not isinstance(num_time_bins, int) or num_time_bins <= 0:
        raise ValueError("num_time_bins must be a positive integer")

    histogram = []
    no_detection_counts = []
    for row in samples.bin_indices_hwp:
        histogram_row = []
        no_detection_row = []
        for periods in row:
            if len(periods) != samples.num_laser_periods:
                raise ValueError("event tensor period axis is inconsistent")
            counts = [0] * num_time_bins
            no_detection = 0
            for index in periods:
                if index == NO_DETECTION:
                    no_detection += 1
                elif 0 <= index < num_time_bins:
                    counts[index] += 1
                else:
                    raise ValueError("first-photon bin index is outside the EWH")
            if sum(counts) + no_detection != samples.num_laser_periods:
                raise AssertionError("each period must produce exactly one outcome")
            histogram_row.append(counts)
            no_detection_row.append(no_detection)
        histogram.append(histogram_row)
        no_detection_counts.append(no_detection_row)

    return EquiWidthHistogram(
        counts_hwt=histogram,
        no_detection_counts_hw=no_detection_counts,
        num_laser_periods=samples.num_laser_periods,
        num_time_bins=num_time_bins,
    )

