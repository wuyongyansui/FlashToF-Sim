import unittest

import numpy as np

from flash_dtof.config import (
    SPEED_OF_LIGHT_M_PER_S,
    SceneInputs,
    SensorConfig,
    derive_config,
    make_uniform_scene,
)
from flash_dtof.geometry import RGBIntrinsics, build_scene_geometry
from flash_dtof.transient import generate_ideal_transient


def _test_intrinsics(width, height, focal_length=100.0):
    """构造主点位于离散中心附近的小型测试相机。"""

    return RGBIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=(width + 1.0) / 2.0,
        cy=(height + 1.0) / 2.0,
        image_size_wh=(width, height),
        source_path="synthetic_camera_params.m",
    )


class ConfigAndTransientTests(unittest.TestCase):
    def test_default_stb_timing_geometry_and_units(self):
        sensor = SensorConfig()
        derived = derive_config(sensor)
        self.assertEqual(derived.tensor_shape_hwt, (480, 640, 672))
        self.assertEqual(sensor.num_laser_periods, 20_000)
        self.assertEqual(sensor.transient_model, "measured_irf")
        self.assertAlmostEqual(derived.time_window_s, 504e-9)
        self.assertAlmostEqual(
            derived.range_per_bin_m, SPEED_OF_LIGHT_M_PER_S * 0.75e-9 / 2.0
        )
        self.assertAlmostEqual(
            derived.max_unambiguous_distance_m,
            SPEED_OF_LIGHT_M_PER_S * 504e-9 / 2.0,
        )
        self.assertEqual(derived.hwt_float32_bytes, 480 * 640 * 672 * 4)
        self.assertEqual(derived.hwt_int32_bytes, 480 * 640 * 672 * 4)

    def test_scene_contract_normalizes_axis_depth_float32(self):
        depth = np.asfortranarray(np.full((3, 4), 2.0, dtype=np.float16))
        scene = SceneInputs(depth_z_m=depth, reflectivity=np.full((3, 4), 0.5))
        self.assertEqual(scene.depth_z_m.dtype, np.float32)
        self.assertTrue(scene.depth_z_m.flags.c_contiguous)
        self.assertEqual(scene.reflectivity.dtype, np.float32)
        self.assertEqual(scene.shape_hw, (3, 4))
        self.assertFalse(hasattr(scene, "depth_m"))

    def test_gaussian_fallback_uses_slant_range_for_tof_and_flux(self):
        sensor = SensorConfig(
            image_height=2,
            image_width=3,
            num_time_bins=96,
            bin_width_s=0.75e-9,
            num_laser_periods=10,
            signal_photons_per_pulse_at_reference=0.2,
            reference_distance_m=2.0,
            background_photons_per_bin=1e-4,
            transient_model="gaussian",
            pulse_fwhm_s=1.5e-9,
        )
        depth_z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        reflectivity = np.array(
            [[0.25, 0.5, 0.75], [1.0, 0.8, 0.6]], dtype=np.float32
        )
        scene = SceneInputs(depth_z_m=depth_z, reflectivity=reflectivity)
        geometry = build_scene_geometry(scene, _test_intrinsics(3, 2, focal_length=5.0))
        transient = generate_ideal_transient(
            sensor, derive_config(sensor), scene, geometry
        )
        self.assertEqual(transient.expected_photons_hwt.shape, (2, 3, 96))
        self.assertEqual(transient.expected_photons_hwt.dtype, np.float32)
        self.assertEqual(transient.response_model, "gaussian")
        peaks = np.argmax(transient.expected_photons_hwt, axis=-1)
        self.assertTrue(np.all(np.diff(peaks[0]) > 0))
        expected_signal = (
            sensor.signal_photons_per_pulse_at_reference
            * reflectivity
            * (sensor.reference_distance_m / geometry.slant_range_m_hw) ** 2
        )
        np.testing.assert_allclose(
            transient.recorded_signal_photons_per_pulse_hw,
            expected_signal,
            rtol=5e-4,
            atol=1e-6,
        )

    def test_scene_outside_slant_range_window_is_rejected(self):
        sensor = SensorConfig(
            image_height=1,
            image_width=1,
            num_time_bins=16,
            transient_model="gaussian",
        )
        scene = make_uniform_scene(sensor, depth_z_m=3.0)
        geometry = build_scene_geometry(scene, _test_intrinsics(1, 1))
        with self.assertRaisesRegex(ValueError, "unambiguous range"):
            generate_ideal_transient(sensor, derive_config(sensor), scene, geometry)


if __name__ == "__main__":
    unittest.main()
