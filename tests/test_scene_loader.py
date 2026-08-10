from pathlib import Path
import unittest

import numpy as np

from flash_dtof.scene import NYUDepthV2Loader


DATASET_ROOT = Path(__file__).resolve().parents[2] / "nyu-depth"


@unittest.skipUnless(DATASET_ROOT.is_dir(), "本地 NYU 轻量数据集不可用")
class NYUSceneLoaderTests(unittest.TestCase):
    def _make_loader(self):
        return NYUDepthV2Loader(
            DATASET_ROOT,
            expected_size_wh=(640, 480),
            reflectivity_mode="constant",
            constant_reflectivity=0.6,
        )

    def test_split_enumeration_is_paired_sorted_and_unique(self):
        loader = self._make_loader()
        train_ids = loader.list_sample_ids("train")
        val_ids = loader.list_sample_ids("val")
        self.assertEqual(len(train_ids), 795)
        self.assertEqual(len(val_ids), 654)
        self.assertEqual(train_ids, tuple(sorted(train_ids)))
        self.assertEqual(val_ids, tuple(sorted(val_ids)))
        self.assertEqual(len(train_ids), len(set(train_ids)))
        self.assertEqual(len(val_ids), len(set(val_ids)))
        for sample_id in train_ids:
            self.assertTrue((DATASET_ROOT / "images" / "train" / (sample_id + ".jpg")).is_file())
            self.assertTrue((DATASET_ROOT / "depth" / "train" / (sample_id + ".npy")).is_file())

    def test_start_limit_preserve_selection_without_duplicates(self):
        loader = self._make_loader()
        all_ids = loader.list_sample_ids("val")
        selected = loader.select_sample_ids("val", start=3, limit=5)
        self.assertEqual(selected, all_ids[3:8])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(
            loader.select_sample_ids("val", start=650, limit=None),
            all_ids[650:],
        )

    def test_pair_counts_native_identity_dtype_and_metric_depth(self):
        loader = self._make_loader()
        self.assertEqual(len(loader.list_sample_ids("train")), 795)
        self.assertEqual(len(loader.list_sample_ids("val")), 654)

        loaded = loader.load("nyu_0000", split="val")
        self.assertEqual(loaded.source_size_wh, (640, 480))
        self.assertEqual(loaded.geometry_transform, "native_identity")
        self.assertEqual(loaded.rgb_u8_hwc.shape, (480, 640, 3))
        self.assertEqual(loaded.rgb_u8_hwc.dtype, np.uint8)
        self.assertEqual(loaded.scene_inputs.depth_z_m.shape, (480, 640))
        self.assertEqual(loaded.scene_inputs.depth_z_m.dtype, np.float32)
        self.assertTrue(loaded.scene_inputs.depth_z_m.flags.c_contiguous)
        self.assertGreater(float(np.min(loaded.scene_inputs.depth_z_m)), 0.0)
        self.assertLess(float(np.max(loaded.scene_inputs.depth_z_m)), 10.1)
        np.testing.assert_array_equal(
            loaded.scene_inputs.reflectivity,
            np.full((480, 640), 0.6, dtype=np.float32),
        )

    def test_native_size_mismatch_is_rejected_instead_of_resized(self):
        loader = NYUDepthV2Loader(
            DATASET_ROOT,
            expected_size_wh=(240, 120),
            reflectivity_mode="constant",
            constant_reflectivity=0.6,
        )
        with self.assertRaisesRegex(ValueError, "crop/resize is disabled"):
            loader.load("nyu_0000", split="val")

    def test_aligned_rgb_relative_proxy_has_target_mean_and_spatial_variation(self):
        loader = NYUDepthV2Loader(
            DATASET_ROOT,
            expected_size_wh=(640, 480),
            reflectivity_mode="rgb_relative_proxy",
            constant_reflectivity=0.5,
        )
        loaded = loader.load("nyu_0000", split="val")
        reflectivity = loaded.scene_inputs.reflectivity
        self.assertEqual(reflectivity.shape, loaded.scene_inputs.depth_z_m.shape)
        self.assertGreaterEqual(float(np.min(reflectivity)), 0.0)
        self.assertLessEqual(float(np.max(reflectivity)), 1.0)
        self.assertGreater(float(np.std(reflectivity)), 0.01)
        self.assertAlmostEqual(float(np.mean(reflectivity)), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
