import math
import unittest

from flash_dtof.config import UserConfig, derive_config
from flash_dtof.ewh import accumulate_ewh
from flash_dtof.first_photon import (
    NO_DETECTION,
    first_photon_probabilities,
    sample_first_photons,
)
from flash_dtof.transient import generate_ideal_transient


class FirstPhotonAndEwhTests(unittest.TestCase):
    def test_analytic_first_photon_distribution(self):
        probabilities, no_detection = first_photon_probabilities([0.2, 0.3])
        self.assertAlmostEqual(probabilities[0], 1.0 - math.exp(-0.2))
        self.assertAlmostEqual(
            probabilities[1], math.exp(-0.2) * (1.0 - math.exp(-0.3))
        )
        self.assertAlmostEqual(no_detection, math.exp(-0.5))
        self.assertAlmostEqual(sum(probabilities) + no_detection, 1.0)

    def test_seed_is_reproducible_and_each_period_has_one_outcome(self):
        config = UserConfig(
            num_time_bins=64,
            bin_width_s=500e-12,
            num_laser_periods=500,
            distance_m=2.0,
            signal_photons_per_pulse_at_reference=1.0,
            reference_distance_m=2.0,
            background_photons_per_bin=1e-3,
            pulse_fwhm_s=1e-9,
            random_seed=99,
        )
        transient = generate_ideal_transient(config, derive_config(config))
        first = sample_first_photons(transient, 500, 99)
        second = sample_first_photons(transient, 500, 99)
        self.assertEqual(first, second)
        events = first.bin_indices_hwp[0][0]
        self.assertEqual(len(events), 500)
        self.assertTrue(
            all(event == NO_DETECTION or 0 <= event < config.num_time_bins for event in events)
        )

        histogram = accumulate_ewh(first, config.num_time_bins)
        counts = histogram.counts_hwt[0][0]
        no_detection = histogram.no_detection_counts_hw[0][0]
        self.assertEqual(sum(counts) + no_detection, 500)


if __name__ == "__main__":
    unittest.main()

