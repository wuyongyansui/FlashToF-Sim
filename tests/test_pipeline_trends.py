from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from flash_dtof.config import (
    SPEED_OF_LIGHT_M_PER_S,
    SceneInputs,
    SensorConfig,
    make_uniform_scene,
)
from flash_dtof.geometry import RGBIntrinsics
from flash_dtof.pipeline import run_simulation


def _camera(width, height, focal_length=1000.0):
    """构造可控的小阵列 RGB 针孔相机。"""

    return RGBIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=(width + 1.0) / 2.0,
        cy=(height + 1.0) / 2.0,
        image_size_wh=(width, height),
        source_path="synthetic.m",
    )


class PipelineTrendTests(unittest.TestCase):
    @unittest.skipUnless(
        (Path(__file__).resolve().parents[2] / "config_IRF_global.txt").is_file(),
        "本地 STB IRF 文件不可用",
    )
    def test_measured_irf_small_end_to_end_peak_and_conservation(self):
        irf_path = Path(__file__).resolve().parents[2] / "config_IRF_global.txt"
        target_bin = 50
        target_range = (
            SPEED_OF_LIGHT_M_PER_S * (target_bin + 0.5) * 0.75e-9 / 2.0
        )
        sensor = SensorConfig(
            image_height=1,
            image_width=1,
            num_time_bins=672,
            bin_width_s=0.75e-9,
            num_laser_periods=5_000,
            signal_photons_per_pulse_at_reference=0.2,
            reference_distance_m=target_range,
            background_photons_per_bin=0.0,
            transient_model="measured_irf",
            measured_irf_path=irf_path,
            random_seed=21,
        )
        scene = make_uniform_scene(sensor, depth_z_m=target_range)
        result = run_simulation(sensor, scene, _camera(1, 1))
        self.assertEqual(result.ideal_transient.response_model, "measured_irf")
        self.assertEqual(int(result.diagnostics.ideal_peak_bin_hw[0, 0]), target_bin)
        self.assertEqual(
            int(result.ewh.detected_counts_hw[0, 0])
            + int(result.ewh.no_detection_counts_hw[0, 0]),
            sensor.num_laser_periods,
        )

    def test_small_array_complete_pipeline_smoke_and_shapes(self):
        # 小场景和少量周期保持测试快速；正式默认值为 480×640×672、20,000 周期。
        sensor = SensorConfig(
            image_height=8,
            image_width=12,
            num_time_bins=128,
            bin_width_s=0.75e-9,
            num_laser_periods=100,
            signal_photons_per_pulse_at_reference=0.03,
            reference_distance_m=4.0,
            background_photons_per_bin=1e-5,
            transient_model="gaussian",
            random_seed=7,
        )
        depth_z = np.linspace(1.0, 10.0, sensor.image_width, dtype=np.float32)
        depth_z = np.broadcast_to(
            depth_z, (sensor.image_height, sensor.image_width)
        ).copy()
        scene = SceneInputs(depth_z_m=depth_z, reflectivity=np.full(depth_z.shape, 0.5))
        result = run_simulation(sensor, scene, _camera(12, 8))

        self.assertEqual(result.ideal_transient.expected_photons_hwt.shape, (8, 12, 128))
        self.assertEqual(result.ewh.counts_hwt.shape, (8, 12, 128))
        self.assertEqual(result.reconstruction.estimated_distance_m_hw.shape, (8, 12))
        self.assertEqual(result.scene_geometry.slant_range_m_hw.shape, (8, 12))
        self.assertEqual(result.diagnostics.peak_shift_bins_hw.shape, (8, 12))
        self.assertTrue(
            np.all(result.scene_geometry.slant_range_m_hw >= scene.depth_z_m)
        )
        totals = result.ewh.detected_counts_hw + result.ewh.no_detection_counts_hw
        np.testing.assert_array_equal(totals, np.full((8, 12), 100))

    def test_low_flux_is_near_ideal_and_high_flux_piles_up_early(self):
        # 200,000 周期仅用于降低趋势测试的 Monte Carlo 不确定性。
        base = SensorConfig(
            image_height=1,
            image_width=1,
            num_time_bins=190,
            bin_width_s=0.75e-9,
            num_laser_periods=200_000,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=5.0,
            background_photons_per_bin=0.0,
            transient_model="gaussian",
            pulse_fwhm_s=4.5e-9,
            random_seed=12345,
        )
        scene = make_uniform_scene(base, depth_z_m=5.0, reflectivity=1.0)
        camera = _camera(1, 1)
        low = run_simulation(base, scene, camera)
        high_sensor = replace(base, signal_photons_per_pulse_at_reference=8.0)
        high = run_simulation(high_sensor, scene, camera)

        low_shift = float(low.diagnostics.peak_shift_bins_hw[0, 0])
        high_shift = float(high.diagnostics.peak_shift_bins_hw[0, 0])
        self.assertLessEqual(abs(low_shift), 1.0)
        self.assertLess(high_shift, -1.0)
        self.assertLess(
            high.reconstruction.estimated_distance_m_hw[0, 0],
            low.reconstruction.estimated_distance_m_hw[0, 0],
        )

    def test_axis_depth_gradient_reconstructs_spatial_slant_range(self):
        # 小型 4×8 测试使用较多周期，以稳定最大-bin 断言。
        sensor = SensorConfig(
            image_height=4,
            image_width=8,
            num_time_bins=190,
            bin_width_s=0.75e-9,
            num_laser_periods=300_000,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=5.0,
            background_photons_per_bin=0.0,
            transient_model="gaussian",
            pulse_fwhm_s=0.6e-9,
            random_seed=11,
        )
        row = np.linspace(1.0, 9.0, sensor.image_width, dtype=np.float32)
        depth_z = np.broadcast_to(row, (sensor.image_height, sensor.image_width)).copy()
        scene = SceneInputs(depth_z_m=depth_z, reflectivity=np.ones_like(depth_z))
        result = run_simulation(sensor, scene, _camera(8, 4, focal_length=20.0))
        estimated_row = result.reconstruction.estimated_distance_m_hw[0]
        true_slant_row = result.scene_geometry.slant_range_m_hw[0]
        self.assertTrue(np.all(np.diff(estimated_row) > 0.0))
        self.assertLessEqual(
            float(np.max(np.abs(estimated_row - true_slant_row))),
            result.derived_config.range_per_bin_m,
        )


if __name__ == "__main__":
    unittest.main()
