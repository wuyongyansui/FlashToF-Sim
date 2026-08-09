"""Ideal transient generation on an equi-width timing grid."""

from dataclasses import dataclass
import math
from typing import List

from .config import DerivedConfig, UserConfig


Tensor3DFloat = List[List[List[float]]]


@dataclass(frozen=True)
class IdealTransient:
    """Expected detected photons per laser period.

    All three tensors have shape ``[H, W, T]`` and units of expected detected
    photons per pixel, per laser period, per time bin.
    """

    expected_photons_hwt: Tensor3DFloat
    signal_photons_hwt: Tensor3DFloat
    background_photons_hwt: Tensor3DFloat
    bin_edges_s: List[float]
    bin_centers_s: List[float]


def _normal_cdf(value, mean, sigma):
    return 0.5 * (1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0))))


def _copy_profile_to_hwt(profile, height, width):
    return [[list(profile) for _ in range(width)] for _ in range(height)]


def generate_ideal_transient(user, derived):
    """Generate a homogeneous single-return Gaussian transient.

    Signal bin weights are Gaussian probability mass integrated between exact
    EWH edges. Background is uniform. No stochastic sampling occurs here.
    """

    if not isinstance(user, UserConfig) or not isinstance(derived, DerivedConfig):
        raise TypeError("user and derived must be UserConfig and DerivedConfig")

    edges = [index * user.bin_width_s for index in range(user.num_time_bins + 1)]
    centers = [
        (index + 0.5) * user.bin_width_s for index in range(user.num_time_bins)
    ]
    signal_profile = []
    for index in range(user.num_time_bins):
        mass = _normal_cdf(
            edges[index + 1], derived.round_trip_time_s, derived.pulse_sigma_s
        ) - _normal_cdf(
            edges[index], derived.round_trip_time_s, derived.pulse_sigma_s
        )
        signal_profile.append(derived.effective_signal_photons_per_pulse * mass)

    background_profile = [user.background_photons_per_bin] * user.num_time_bins
    expected_profile = [
        signal + background
        for signal, background in zip(signal_profile, background_profile)
    ]

    return IdealTransient(
        expected_photons_hwt=_copy_profile_to_hwt(
            expected_profile, user.image_height, user.image_width
        ),
        signal_photons_hwt=_copy_profile_to_hwt(
            signal_profile, user.image_height, user.image_width
        ),
        background_photons_hwt=_copy_profile_to_hwt(
            background_profile, user.image_height, user.image_width
        ),
        bin_edges_s=edges,
        bin_centers_s=centers,
    )

