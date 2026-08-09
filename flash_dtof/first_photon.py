"""Per-laser-period first-photon SPAD acquisition."""

from bisect import bisect_left
from dataclasses import dataclass
import math
import random
from typing import List

from .transient import IdealTransient


NO_DETECTION = -1
Tensor3DInt = List[List[List[int]]]


@dataclass(frozen=True)
class FirstPhotonSamples:
    """One outcome for every pixel and laser period.

    ``bin_indices_hwp`` has shape ``[H, W, P]``. Each value is the earliest
    detected time-bin index in that laser period, or ``NO_DETECTION``. It is
    therefore impossible for a period to contribute to multiple EWH bins.
    """

    bin_indices_hwp: Tensor3DInt
    num_laser_periods: int
    random_seed: int


def first_photon_probabilities(expected_photons_per_bin):
    """Return exact first-detection probabilities for independent Poisson bins.

    For expected counts ``lambda[k]`` per period,

    ``P(K=k) = exp(-sum(lambda[:k])) * (1 - exp(-lambda[k]))``.

    The second return value is the probability of no detection in the complete
    timing window. This is the first-photon pile-up forward model, not an
    independent count draw for every bin.
    """

    probabilities = []
    survival = 1.0
    for index, expected in enumerate(expected_photons_per_bin):
        if not math.isfinite(expected) or expected < 0.0:
            raise ValueError(
                "expected photon count at bin {} must be finite and >= 0".format(index)
            )
        detection_in_bin = -math.expm1(-expected)
        probabilities.append(survival * detection_in_bin)
        survival *= math.exp(-expected)
    return probabilities, survival


def _probability_cdf(expected_photons_per_bin):
    probabilities, no_detection_probability = first_photon_probabilities(
        expected_photons_per_bin
    )
    cdf = []
    running = 0.0
    for probability in probabilities:
        running += probability
        cdf.append(running)
    return cdf, no_detection_probability


def sample_first_photons(transient, num_laser_periods, random_seed):
    """Sample exactly one earliest detection (or none) per pixel and period."""

    if not isinstance(transient, IdealTransient):
        raise TypeError("transient must be IdealTransient")
    if not isinstance(num_laser_periods, int) or num_laser_periods <= 0:
        raise ValueError("num_laser_periods must be a positive integer")
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    generator = random.Random(random_seed)
    all_pixels = []
    for row in transient.expected_photons_hwt:
        sampled_row = []
        for expected_profile in row:
            cdf, _ = _probability_cdf(expected_profile)
            num_bins = len(cdf)
            periods = []
            for _ in range(num_laser_periods):
                index = bisect_left(cdf, generator.random())
                periods.append(index if index < num_bins else NO_DETECTION)
            sampled_row.append(periods)
        all_pixels.append(sampled_row)

    return FirstPhotonSamples(
        bin_indices_hwp=all_pixels,
        num_laser_periods=num_laser_periods,
        random_seed=random_seed,
    )
