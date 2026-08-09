import unittest

import numpy as np

from flash_dtof.config import SPEED_OF_LIGHT_M_PER_S
from flash_dtof.ewh import EquiWidthHistogram
from flash_dtof.reconstruction import reconstruct_maximum_bin


class ReconstructionTests(unittest.TestCase):
    def test_vectorized_maximum_bin_uses_centres_and_si_units(self):
        counts = np.zeros((2, 3, 16), dtype=np.int32)
        peak_indices = np.array([[1, 3, 5], [7, 9, 11]])
        for row in range(2):
            for column in range(3):
                counts[row, column, peak_indices[row, column]] = 12
        histogram = EquiWidthHistogram(
            counts_hwt=counts,
            no_detection_counts_hw=np.full((2, 3), 88, dtype=np.int32),
            num_laser_periods=100,
            num_time_bins=16,
        )
        bin_width_s = 100e-12
        result = reconstruct_maximum_bin(histogram, bin_width_s)
        np.testing.assert_array_equal(result.peak_bin_hw, peak_indices)
        expected = SPEED_OF_LIGHT_M_PER_S * (peak_indices + 0.5) * bin_width_s / 2.0
        np.testing.assert_allclose(result.estimated_distance_m_hw, expected, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()

