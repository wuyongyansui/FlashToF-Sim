import unittest

from flash_dtof.config import SPEED_OF_LIGHT_M_PER_S
from flash_dtof.ewh import EquiWidthHistogram
from flash_dtof.reconstruction import reconstruct_maximum_bin


class ReconstructionTests(unittest.TestCase):
    def test_maximum_bin_uses_bin_center_and_correct_si_units(self):
        counts = [0] * 16
        counts[9] = 12
        histogram = EquiWidthHistogram(
            counts_hwt=[[counts]],
            no_detection_counts_hw=[[88]],
            num_laser_periods=100,
            num_time_bins=16,
        )
        bin_width_s = 100e-12
        result = reconstruct_maximum_bin(histogram, bin_width_s)
        expected_distance_m = SPEED_OF_LIGHT_M_PER_S * (9.5 * bin_width_s) / 2.0
        self.assertEqual(result.peak_bin_hw, [[9]])
        self.assertAlmostEqual(result.estimated_distance_m_hw[0][0], expected_distance_m)


if __name__ == "__main__":
    unittest.main()

