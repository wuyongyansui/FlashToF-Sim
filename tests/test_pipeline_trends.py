from dataclasses import replace
import unittest

import numpy as np

from flash_dtof.config import SceneInputs, SensorConfig, make_uniform_scene
from flash_dtof.pipeline import run_simulation


class PipelineTrendTests(unittest.TestCase):
    def test_full_120x240_array_smoke_and_shapes(self):
        # 100 个周期可保持完整 shape 冒烟测试快速；正式默认值为 20,000。
        sensor = SensorConfig(
            num_laser_periods=100,
            signal_photons_per_pulse_at_reference=0.03,
            reference_distance_m=4.0,
            background_photons_per_bin=1e-5,
            random_seed=7,
        )
        depth = np.linspace(1.0, 10.0, sensor.image_width, dtype=np.float32)
        depth = np.broadcast_to(depth, (sensor.image_height, sensor.image_width)).copy()
        scene = SceneInputs(depth_m=depth, reflectivity=np.full(depth.shape, 0.5))
        result = run_simulation(sensor, scene)

        self.assertEqual(result.ideal_transient.expected_photons_hwt.shape, (120, 240, 190))
        self.assertEqual(result.ewh.counts_hwt.shape, (120, 240, 190))
        self.assertEqual(result.reconstruction.estimated_distance_m_hw.shape, (120, 240))
        self.assertEqual(result.diagnostics.peak_shift_bins_hw.shape, (120, 240))
        totals = result.ewh.detected_counts_hw + result.ewh.no_detection_counts_hw
        np.testing.assert_array_equal(totals, np.full((120, 240), 100))

    def test_low_flux_is_near_ideal_and_high_flux_piles_up_early(self):
        # 仅在此趋势测试中使用 200,000 个周期，以降低 Monte Carlo 不确定性。
        base = SensorConfig(
            image_height=1,
            image_width=1,
            num_time_bins=190,
            bin_width_s=0.5e-9,
            num_laser_periods=200_000,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=5.0,
            background_photons_per_bin=0.0,
            pulse_fwhm_s=4.0e-9,
            random_seed=12345,
        )
        scene = make_uniform_scene(base, distance_m=5.0, reflectivity=1.0)
        low = run_simulation(base, scene)
        high_sensor = replace(base, signal_photons_per_pulse_at_reference=8.0)
        high = run_simulation(high_sensor, scene)

        low_shift = float(low.diagnostics.peak_shift_bins_hw[0, 0])
        high_shift = float(high.diagnostics.peak_shift_bins_hw[0, 0])
        self.assertLessEqual(abs(low_shift), 1.0)
        self.assertLess(high_shift, -1.0)
        self.assertLess(
            high.reconstruction.estimated_distance_m_hw[0, 0],
            low.reconstruction.estimated_distance_m_hw[0, 0],
        )

    def test_distance_gradient_reconstructs_a_spatial_range_map(self):
        # 在这个小型 4x8 测试中使用 300,000 个周期，以稳定最大 bin 断言。
        sensor = SensorConfig(
            image_height=4,
            image_width=8,
            num_laser_periods=300_000,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=5.0,
            background_photons_per_bin=0.0,
            pulse_fwhm_s=0.6e-9,
            random_seed=11,
        )
        row = np.linspace(1.0, 9.0, sensor.image_width, dtype=np.float32)
        depth = np.broadcast_to(row, (sensor.image_height, sensor.image_width)).copy()
        scene = SceneInputs(depth_m=depth, reflectivity=np.ones_like(depth))
        result = run_simulation(sensor, scene)
        estimated_row = result.reconstruction.estimated_distance_m_hw[0]
        self.assertTrue(np.all(np.diff(estimated_row) > 0.0))
        self.assertLessEqual(
            float(np.max(np.abs(estimated_row - row))),
            result.derived_config.range_per_bin_m,
        )


if __name__ == "__main__":
    unittest.main()
