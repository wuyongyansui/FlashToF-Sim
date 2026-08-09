"""User-editable configuration for ``python run_simulation.py``.

Only edit the values in USER_CONFIG for the first prototype. Derived timing,
range, photon-flux, and shape quantities are computed and validated by the
simulator; they are intentionally not duplicated here.
"""

from flash_dtof.config import UserConfig


# ---------------------------------------------------------------------------
# USER-SET BASIC PARAMETERS
# ---------------------------------------------------------------------------
USER_CONFIG = UserConfig(
    # Minimal homogeneous Flash array. Every pixel sees the same first scene.
    image_height=1,
    image_width=1,

    # Uniform EWH timing grid.
    num_time_bins=256,
    bin_width_s=100e-12,
    num_laser_periods=200_000,

    # Single-surface scene.
    distance_m=2.5,
    reflectivity=1.0,

    # Expected DETECTED photons, i.e. after optical loss and PDE.
    # The signal value is defined at unit reflectivity and reference_distance_m.
    signal_photons_per_pulse_at_reference=0.05,
    reference_distance_m=2.5,
    background_photons_per_bin=1e-5,

    # Gaussian system impulse response full width at half maximum.
    pulse_fwhm_s=1.0e-9,

    # Reproducible Monte Carlo sequence.
    random_seed=20260809,
)

