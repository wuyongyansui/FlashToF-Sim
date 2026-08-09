import unittest

import numpy as np

from flash_dtof.config import (
    SPEED_OF_LIGHT_M_PER_S,
    SceneInputs,
    SensorConfig,
    derive_config,
    make_uniform_scene,
)
from flash_dtof.transient import generate_ideal_transient


class ConfigAndTransientTests(unittest.TestCase):
    def test_default_target_geometry_and_units(self):
        sensor = SensorConfig()
        derived = derive_config(sensor)
        self.assertEqual(derived.tensor_shape_hwt, (120, 240, 190))
        self.assertEqual(sensor.num_laser_periods, 20_000)
        self.assertAlmostEqual(derived.time_window_s, 95e-9)
        self.assertAlmostEqual(
            derived.range_per_bin_m, SPEED_OF_LIGHT_M_PER_S * 0.5e-9 / 2.0
        )
        self.assertAlmostEqual(
            derived.max_unambiguous_distance_m,
            SPEED_OF_LIGHT_M_PER_S * 95e-9 / 2.0,
        )

    def test_scene_contract_normalizes_float32_contiguous_arrays(self):
        depth = np.asfortranarray(np.full((3, 4), 2.0, dtype=np.float16))
        scene = SceneInputs(depth_m=depth, reflectivity=np.full((3, 4), 0.5))
        self.assertEqual(scene.depth_m.dtype, np.float32)
        self.assertTrue(scene.depth_m.flags.c_contiguous)
        self.assertEqual(scene.reflectivity.dtype, np.float32)
        self.assertEqual(scene.shape_hw, (3, 4))

    def test_transient_is_spatially_varying_hwt(self):
        sensor = SensorConfig(
            image_height=2,
            image_width=3,
            num_time_bins=190,
            bin_width_s=0.5e-9,
            num_laser_periods=10,
            signal_photons_per_pulse_at_reference=0.2,
            reference_distance_m=2.0,
            background_photons_per_bin=1e-4,
            pulse_fwhm_s=1e-9,
        )
        depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        reflectivity = np.array(
            [[0.25, 0.5, 0.75], [1.0, 0.8, 0.6]], dtype=np.float32
        )
        scene = SceneInputs(depth_m=depth, reflectivity=reflectivity)
        transient = generate_ideal_transient(sensor, derive_config(sensor), scene)
        self.assertEqual(transient.expected_photons_hwt.shape, (2, 3, 190))
        self.assertEqual(transient.expected_photons_hwt.dtype, np.float32)
        peaks = np.argmax(transient.signal_photons_hwt, axis=-1)
        self.assertTrue(np.all(np.diff(peaks[0]) > 0))
        expected_signal = (
            sensor.signal_photons_per_pulse_at_reference
            * reflectivity
            * (sensor.reference_distance_m / depth) ** 2
        )
        np.testing.assert_allclose(
            np.sum(transient.signal_photons_hwt, axis=-1),
            expected_signal,
            rtol=3e-4,
            atol=1e-6,
        )

    def test_scene_outside_timing_window_is_rejected(self):
        sensor = SensorConfig(image_height=1, image_width=1)
        scene = make_uniform_scene(sensor, distance_m=15.0)
        with self.assertRaisesRegex(ValueError, "unambiguous range"):
            generate_ideal_transient(sensor, derive_config(sensor), scene)


if __name__ == "__main__":
    unittest.main()
