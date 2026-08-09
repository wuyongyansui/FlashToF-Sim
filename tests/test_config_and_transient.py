import unittest

from flash_dtof.config import SPEED_OF_LIGHT_M_PER_S, UserConfig, derive_config
from flash_dtof.transient import generate_ideal_transient


class ConfigAndTransientTests(unittest.TestCase):
    def test_derived_shapes_units_and_signal_law(self):
        config = UserConfig(
            image_height=2,
            image_width=3,
            num_time_bins=200,
            bin_width_s=100e-12,
            num_laser_periods=7,
            distance_m=2.0,
            reflectivity=0.5,
            signal_photons_per_pulse_at_reference=0.8,
            reference_distance_m=1.0,
            background_photons_per_bin=1e-4,
            pulse_fwhm_s=500e-12,
        )
        derived = derive_config(config)
        self.assertEqual(derived.tensor_shape_hwt, (2, 3, 200))
        self.assertEqual(derived.event_shape_hwp, (2, 3, 7))
        self.assertAlmostEqual(derived.time_window_s, 20e-9)
        self.assertAlmostEqual(
            derived.max_unambiguous_distance_m,
            SPEED_OF_LIGHT_M_PER_S * 20e-9 / 2.0,
        )
        self.assertAlmostEqual(derived.effective_signal_photons_per_pulse, 0.1)
        self.assertAlmostEqual(derived.expected_background_photons_per_pulse, 0.02)

    def test_transient_has_hwt_shape_and_expected_components(self):
        config = UserConfig(
            image_height=2,
            image_width=2,
            num_time_bins=128,
            bin_width_s=200e-12,
            num_laser_periods=10,
            distance_m=2.0,
            signal_photons_per_pulse_at_reference=0.2,
            reference_distance_m=2.0,
            background_photons_per_bin=1e-4,
            pulse_fwhm_s=1e-9,
        )
        transient = generate_ideal_transient(config, derive_config(config))
        self.assertEqual(len(transient.expected_photons_hwt), 2)
        self.assertEqual(len(transient.expected_photons_hwt[0]), 2)
        self.assertEqual(len(transient.expected_photons_hwt[0][0]), 128)
        profile = transient.expected_photons_hwt[0][0]
        signal = transient.signal_photons_hwt[0][0]
        background = transient.background_photons_hwt[0][0]
        for total, signal_value, background_value in zip(profile, signal, background):
            self.assertAlmostEqual(total, signal_value + background_value)
        self.assertAlmostEqual(sum(signal), 0.2, places=10)

    def test_echo_outside_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the EWH time window"):
            UserConfig(
                num_time_bins=10,
                bin_width_s=100e-12,
                distance_m=1.0,
            )


if __name__ == "__main__":
    unittest.main()
