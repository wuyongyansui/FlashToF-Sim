"""Configuration split into user-set basics and read-only derived values."""

from dataclasses import dataclass
import math


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class UserConfig:
    """Basic parameters that a simulator user is expected to edit.

    Photon-rate fields are expected *detected* photon counts, after any optical
    throughput and photon-detection efficiency. The first prototype uses a
    homogeneous ``[H, W]`` scene, but retains explicit spatial axes so later
    scene models do not have to change the timing tensor convention.
    """

    image_height: int = 1
    image_width: int = 1
    num_time_bins: int = 256
    bin_width_s: float = 100e-12
    num_laser_periods: int = 100_000

    distance_m: float = 2.5
    reflectivity: float = 1.0
    signal_photons_per_pulse_at_reference: float = 0.05
    reference_distance_m: float = 2.5
    background_photons_per_bin: float = 1e-5
    pulse_fwhm_s: float = 1.0e-9

    random_seed: int = 0

    def __post_init__(self):
        _validate_user_config(self)


@dataclass(frozen=True)
class DerivedConfig:
    """Read-only values derived from :class:`UserConfig`.

    These values are constraints or consequences of the user inputs and must
    not be independently edited.
    """

    tensor_shape_hwt: tuple
    event_shape_hwp: tuple
    time_window_s: float
    max_unambiguous_distance_m: float
    round_trip_time_s: float
    echo_center_bin_coordinate: float
    pulse_sigma_s: float
    effective_signal_photons_per_pulse: float
    expected_background_photons_per_pulse: float


def _validate_user_config(config):
    integer_fields = (
        ("image_height", config.image_height),
        ("image_width", config.image_width),
        ("num_time_bins", config.num_time_bins),
        ("num_laser_periods", config.num_laser_periods),
    )
    for name, value in integer_fields:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))

    positive_fields = (
        ("bin_width_s", config.bin_width_s),
        ("distance_m", config.distance_m),
        ("reference_distance_m", config.reference_distance_m),
        ("pulse_fwhm_s", config.pulse_fwhm_s),
    )
    for name, value in positive_fields:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and > 0".format(name))

    if not math.isfinite(config.reflectivity) or not 0.0 <= config.reflectivity <= 1.0:
        raise ValueError("reflectivity must be finite and in [0, 1]")

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

    if not isinstance(config.random_seed, int) or isinstance(config.random_seed, bool):
        raise ValueError("random_seed must be an integer")

    round_trip_time_s = 2.0 * config.distance_m / SPEED_OF_LIGHT_M_PER_S
    time_window_s = config.num_time_bins * config.bin_width_s
    if round_trip_time_s >= time_window_s:
        raise ValueError(
            "distance_m places the echo outside the EWH time window; "
            "increase num_time_bins/bin_width_s or reduce distance_m"
        )


def derive_config(config):
    """Validate and derive immutable timing, range, flux, and shape values."""

    # __post_init__ already validates normal construction. Calling it here
    # again keeps this function safe for unusual deserialization workflows.
    _validate_user_config(config)
    time_window_s = config.num_time_bins * config.bin_width_s
    round_trip_time_s = 2.0 * config.distance_m / SPEED_OF_LIGHT_M_PER_S
    pulse_sigma_s = config.pulse_fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # Deliberately simple first-order radiometric law for the MVP. The input is
    # a detected-photon reference level, so no hidden PDE is applied here.
    effective_signal = (
        config.signal_photons_per_pulse_at_reference
        * config.reflectivity
        * (config.reference_distance_m / config.distance_m) ** 2
    )

    return DerivedConfig(
        tensor_shape_hwt=(config.image_height, config.image_width, config.num_time_bins),
        event_shape_hwp=(
            config.image_height,
            config.image_width,
            config.num_laser_periods,
        ),
        time_window_s=time_window_s,
        max_unambiguous_distance_m=SPEED_OF_LIGHT_M_PER_S * time_window_s / 2.0,
        round_trip_time_s=round_trip_time_s,
        echo_center_bin_coordinate=(round_trip_time_s / config.bin_width_s) - 0.5,
        pulse_sigma_s=pulse_sigma_s,
        effective_signal_photons_per_pulse=effective_signal,
        expected_background_photons_per_pulse=(
            config.background_photons_per_bin * config.num_time_bins
        ),
    )


def format_config(user, derived):
    """Return a concise display that visibly separates editable/derived data."""

    return "\n".join(
        [
            "USER-SET BASIC PARAMETERS",
            "  array [H,W]                 : [{},{}]".format(
                user.image_height, user.image_width
            ),
            "  EWH bins / bin width        : {} / {:.3e} s".format(
                user.num_time_bins, user.bin_width_s
            ),
            "  laser periods               : {}".format(user.num_laser_periods),
            "  distance / reflectivity     : {:.6f} m / {:.3f}".format(
                user.distance_m, user.reflectivity
            ),
            "  signal photons at reference : {:.6g} detected/pulse".format(
                user.signal_photons_per_pulse_at_reference
            ),
            "  signal reference distance    : {:.6f} m".format(
                user.reference_distance_m
            ),
            "  background                  : {:.6g} detected/bin/pulse".format(
                user.background_photons_per_bin
            ),
            "  pulse FWHM                  : {:.3e} s".format(user.pulse_fwhm_s),
            "  random seed                 : {}".format(user.random_seed),
            "DERIVED / CONSTRAINT PARAMETERS (read-only)",
            "  transient / EWH shape       : {} [H,W,T]".format(
                derived.tensor_shape_hwt
            ),
            "  first-photon event shape    : {} [H,W,P]".format(
                derived.event_shape_hwp
            ),
            "  timing window               : {:.3e} s".format(derived.time_window_s),
            "  unambiguous range           : {:.6f} m".format(
                derived.max_unambiguous_distance_m
            ),
            "  round-trip time             : {:.3e} s".format(
                derived.round_trip_time_s
            ),
            "  expected signal             : {:.6g} detected/pulse".format(
                derived.effective_signal_photons_per_pulse
            ),
            "  expected background         : {:.6g} detected/pulse".format(
                derived.expected_background_photons_per_pulse
            ),
        ]
    )
