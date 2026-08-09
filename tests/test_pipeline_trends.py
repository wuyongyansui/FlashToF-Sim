from dataclasses import replace
import unittest

from flash_dtof.config import UserConfig
from flash_dtof.first_photon import first_photon_probabilities
from flash_dtof.pipeline import run_simulation


def _normalized(values):
    total = float(sum(values))
    return [value / total for value in values]


class PipelineTrendTests(unittest.TestCase):
    def setUp(self):
        self.low_flux = UserConfig(
            num_time_bins=256,
            bin_width_s=100e-12,
            num_laser_periods=200_000,
            distance_m=2.5,
            reflectivity=1.0,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=2.5,
            background_photons_per_bin=0.0,
            pulse_fwhm_s=1.0e-9,
            random_seed=12345,
        )

    def test_low_flux_histogram_is_close_to_ideal_shape(self):
        result = run_simulation(self.low_flux)
        ideal = _normalized(result.ideal_transient.expected_photons_hwt[0][0])
        measured = _normalized(result.ewh.counts_hwt[0][0])
        total_variation = 0.5 * sum(
            abs(measured_value - ideal_value)
            for measured_value, ideal_value in zip(measured, ideal)
        )
        diagnostic = result.diagnostics[0]
        self.assertLess(total_variation, 0.08)
        self.assertLessEqual(abs(diagnostic.peak_shift_bins), 1)

    def test_high_flux_produces_first_photon_pileup_and_earlier_peak(self):
        high_flux = replace(
            self.low_flux,
            signal_photons_per_pulse_at_reference=8.0,
            num_laser_periods=100_000,
        )
        low_result = run_simulation(self.low_flux)
        high_result = run_simulation(high_flux)
        low_diagnostic = low_result.diagnostics[0]
        high_diagnostic = high_result.diagnostics[0]

        self.assertGreaterEqual(low_diagnostic.peak_shift_bins, -1)
        self.assertLess(high_diagnostic.peak_shift_bins, -2)
        self.assertLess(
            high_diagnostic.observed_peak_bin,
            low_diagnostic.observed_peak_bin,
        )
        self.assertLess(
            high_diagnostic.estimated_distance_m,
            low_diagnostic.estimated_distance_m,
        )

        # The analytic forward model must predict the same earlier-mode trend.
        low_profile = low_result.ideal_transient.expected_photons_hwt[0][0]
        high_profile = high_result.ideal_transient.expected_photons_hwt[0][0]
        low_probabilities, _ = first_photon_probabilities(low_profile)
        high_probabilities, _ = first_photon_probabilities(high_profile)
        analytic_low_peak = max(
            range(len(low_probabilities)), key=low_probabilities.__getitem__
        )
        analytic_high_peak = max(
            range(len(high_probabilities)), key=high_probabilities.__getitem__
        )
        self.assertLess(analytic_high_peak, analytic_low_peak)


if __name__ == "__main__":
    unittest.main()

