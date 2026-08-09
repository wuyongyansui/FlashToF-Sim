from pathlib import Path
import unittest

import numpy as np

from flash_dtof.config import CameraGeometryConfig, SensorConfig
from flash_dtof.geometry import (
    RGBIntrinsics,
    axial_depth_to_slant_range,
    load_nyu_rgb_intrinsics,
    make_rgb_unit_rays,
)
from flash_dtof.irf import load_measured_irf, shifted_irf_mass_at_bin_centers


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CAMERA_PARAMS = WORKSPACE_ROOT / "toolbox_nyu_depth_v2" / "camera_params.m"
IRF_PATH = WORKSPACE_ROOT / "config_IRF_global.txt"


class GeometryAndIrfTests(unittest.TestCase):
    def test_toolbox_rgb_intrinsics_are_parsed_and_validated_for_640x480(self):
        sensor = SensorConfig()
        config = CameraGeometryConfig(camera_params_path=CAMERA_PARAMS)
        intrinsics = load_nyu_rgb_intrinsics(config, sensor)
        self.assertAlmostEqual(intrinsics.fx, 518.85790117450188)
        self.assertAlmostEqual(intrinsics.fy, 519.46961112127485)
        self.assertAlmostEqual(intrinsics.cx, 325.58244941119034)
        self.assertAlmostEqual(intrinsics.cy, 253.73616633400465)
        self.assertEqual(intrinsics.image_size_wh, (640, 480))
        self.assertEqual(intrinsics.pixel_coordinate_convention, "matlab_one_based")

    def test_unit_rays_and_discrete_center_ray(self):
        intrinsics = RGBIntrinsics(
            fx=100.0,
            fy=120.0,
            cx=2.0,
            cy=2.0,
            image_size_wh=(3, 3),
            source_path="synthetic.m",
        )
        rays = make_rgb_unit_rays(intrinsics)
        np.testing.assert_allclose(
            np.linalg.norm(rays, axis=-1), np.ones((3, 3)), rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(rays[1, 1], [0.0, 0.0, 1.0], atol=1e-7)
        self.assertGreater(rays[1, 2, 0], 0.0)
        self.assertGreater(rays[0, 1, 1], 0.0)

    def test_axis_depth_to_slant_range_uses_z_over_dz(self):
        depth_z = np.array([[2.0, 2.0]], dtype=np.float32)
        direction_z = np.array([[1.0, 0.8]], dtype=np.float32)
        slant = axial_depth_to_slant_range(depth_z, direction_z)
        np.testing.assert_allclose(slant, [[2.0, 2.5]], rtol=1e-7)

    @unittest.skipUnless(IRF_PATH.is_file(), "本地 STB IRF 文件不可用")
    def test_stb_irf_load_clip_normalize_and_shift(self):
        measured = load_measured_irf(IRF_PATH, expected_bin_width_s=0.75e-9)
        self.assertEqual(measured.probability_mass.shape, (501,))
        self.assertEqual(measured.peak_index, 250)
        self.assertAlmostEqual(measured.sample_interval_s, 0.75e-9)
        self.assertTrue(np.all(measured.probability_mass >= 0.0))
        self.assertAlmostEqual(float(np.sum(measured.probability_mass)), 1.0, places=14)
        self.assertGreater(measured.negative_mass_fraction_before_clip, 0.0)
        self.assertLess(measured.negative_mass_fraction_before_clip, 0.001)

        centers = (np.arange(672, dtype=np.float64) + 0.5) * 0.75e-9
        target_peak = 300
        tof = np.array([[centers[target_peak]]], dtype=np.float64)
        shifted = shifted_irf_mass_at_bin_centers(measured, tof, centers)
        self.assertEqual(int(np.argmax(shifted[0, 0])), target_peak)
        self.assertAlmostEqual(float(np.sum(shifted)), 1.0, places=6)
        self.assertGreater(float(np.sum(shifted[0, 0, :target_peak])), 0.0)
        self.assertGreater(float(np.sum(shifted[0, 0, target_peak + 1 :])), 0.0)


if __name__ == "__main__":
    unittest.main()
