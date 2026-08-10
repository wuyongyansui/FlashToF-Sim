from pathlib import Path
import unittest

import numpy as np

from flash_dtof.config import SceneInputs, SensorConfig, derive_config
from flash_dtof.geometry import RGBIntrinsics, build_scene_geometry
from flash_dtof.scene import (
    NYUDepthV2Loader,
    make_rgb_relative_reflectivity,
    srgb_u8_to_linear_rgb,
)
from flash_dtof.transient import generate_ideal_transient


class RGBRelativeReflectivityTests(unittest.TestCase):
    def test_constant_mode_is_unchanged(self):
        loader = object.__new__(NYUDepthV2Loader)
        loader.reflectivity_mode = "constant"
        loader.constant_reflectivity = 0.37
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(
            loader._make_reflectivity(rgb),
            np.full((2, 3), 0.37, dtype=np.float32),
        )

    def test_srgb_is_linearized_before_rec709_luminance(self):
        rgb = np.array([[[0, 128, 255]]], dtype=np.uint8)
        linear = srgb_u8_to_linear_rgb(rgb)
        expected_mid = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4
        np.testing.assert_allclose(
            linear[0, 0],
            np.array([0.0, expected_mid, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_proxy_preserves_order_target_mean_and_physical_range(self):
        levels = np.array([16, 32, 64, 128, 255], dtype=np.uint8)
        rgb = np.repeat(levels[np.newaxis, :, np.newaxis], 3, axis=2)
        reflectivity = make_rgb_relative_reflectivity(
            rgb,
            target_mean=0.4,
            ratio_min=0.05,
            ratio_max=20.0,
        )
        self.assertTrue(np.all(np.diff(reflectivity[0]) > 0.0))
        self.assertGreaterEqual(float(np.min(reflectivity)), 0.0)
        self.assertLessEqual(float(np.max(reflectivity)), 1.0)
        self.assertAlmostEqual(float(np.mean(reflectivity)), 0.4, places=6)

    def test_all_black_safely_falls_back_to_constant(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        reflectivity = make_rgb_relative_reflectivity(rgb, target_mean=0.42)
        np.testing.assert_array_equal(
            reflectivity,
            np.full((4, 5), 0.42, dtype=np.float32),
        )

    def test_rgb_proxy_changes_signal_but_not_fixed_background(self):
        rgb = np.repeat(
            np.array([[32, 64, 128, 255]], dtype=np.uint8)[..., np.newaxis],
            3,
            axis=2,
        )
        reflectivity = make_rgb_relative_reflectivity(rgb, target_mean=0.5)
        scene = SceneInputs(
            depth_z_m=np.full((1, 4), 2.0, dtype=np.float32),
            reflectivity=reflectivity,
        )
        sensor = SensorConfig(
            image_height=1,
            image_width=4,
            num_time_bins=64,
            bin_width_s=0.75e-9,
            num_laser_periods=100,
            signal_photons_per_pulse_at_reference=0.2,
            reference_distance_m=2.0,
            background_photons_per_bin=0.0123,
            transient_model="gaussian",
            pulse_fwhm_s=1.0e-9,
            random_seed=7,
        )
        intrinsics = RGBIntrinsics(
            fx=1e9,
            fy=1e9,
            cx=2.5,
            cy=1.0,
            image_size_wh=(4, 1),
            source_path=Path("synthetic_camera_params.m"),
        )
        geometry = build_scene_geometry(scene, intrinsics)
        transient = generate_ideal_transient(
            sensor,
            derive_config(sensor),
            scene,
            geometry,
        )
        expected_signal = (
            sensor.signal_photons_per_pulse_at_reference
            * reflectivity
            * (sensor.reference_distance_m / geometry.slant_range_m_hw) ** 2
        )
        np.testing.assert_allclose(
            transient.effective_signal_photons_per_pulse_hw,
            expected_signal,
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertTrue(
            np.all(np.diff(transient.effective_signal_photons_per_pulse_hw[0]) > 0.0)
        )
        np.testing.assert_array_equal(
            transient.background_photons_per_bin_hw,
            np.full((1, 4), 0.0123, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
