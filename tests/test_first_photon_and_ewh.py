import math
import unittest

import numpy as np

from flash_dtof.ewh import accumulate_ewh
from flash_dtof.first_photon import (
    first_photon_probabilities,
    sample_first_photon_counts,
)


class FirstPhotonAndEwhTests(unittest.TestCase):
    def test_analytic_first_photon_distribution(self):
        rates = np.array([[[0.2, 0.3]]], dtype=np.float32)
        probabilities, no_detection = first_photon_probabilities(rates)
        self.assertAlmostEqual(probabilities[0, 0, 0], 1.0 - math.exp(-0.2), places=7)
        self.assertAlmostEqual(
            probabilities[0, 0, 1],
            math.exp(-0.2) * (1.0 - math.exp(-0.3)),
            places=7,
        )
        self.assertAlmostEqual(no_detection[0, 0], math.exp(-0.5), places=7)
        self.assertAlmostEqual(
            float(np.sum(probabilities[0, 0]) + no_detection[0, 0]), 1.0, places=7
        )

    def test_seed_reproducibility_and_pixelwise_period_conservation(self):
        # 10,000 是统计单元测试参数，不是用户默认值。
        rates = np.zeros((4, 5, 12), dtype=np.float32)
        rates[..., 3:7] = np.array([0.1, 0.3, 0.2, 0.05], dtype=np.float32)
        first = sample_first_photon_counts(rates, 10_000, 99)
        second = sample_first_photon_counts(rates, 10_000, 99)
        np.testing.assert_array_equal(first.counts_hwt, second.counts_hwt)
        np.testing.assert_array_equal(
            first.no_detection_counts_hw, second.no_detection_counts_hw
        )

        histogram = accumulate_ewh(first)
        totals = histogram.detected_counts_hw + histogram.no_detection_counts_hw
        np.testing.assert_array_equal(totals, np.full((4, 5), 10_000))
        self.assertEqual(histogram.counts_hwt.shape, (4, 5, 12))
        self.assertEqual(histogram.counts_hwt.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
