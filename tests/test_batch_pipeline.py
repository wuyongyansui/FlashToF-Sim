from pathlib import Path
import unittest

import numpy as np

from flash_dtof.batch import NYUBatchConfig, run_nyu_batch
from flash_dtof.config import SensorConfig


DATASET_ROOT = Path(__file__).resolve().parents[2] / "nyu-depth"


@unittest.skipUnless(DATASET_ROOT.is_dir(), "本地 NYU 轻量数据集不可用")
class NYUBatchPipelineTests(unittest.TestCase):
    def test_multi_sample_streaming_batch_smoke(self):
        # 此处特意使用小阵列和少量周期，使集成测试保持快速。
        # 正式默认值仍为 120x240x190 和 20,000 个激光周期。
        sensor = SensorConfig(
            image_height=8,
            image_width=12,
            num_time_bins=190,
            bin_width_s=0.5e-9,
            num_laser_periods=50,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=2.5,
            background_photons_per_bin=1e-5,
            random_seed=31415,
        )
        batch = NYUBatchConfig(
            dataset_root=DATASET_ROOT,
            split="val",
            start=1,
            limit=3,
            reflectivity_mode="constant",
            constant_reflectivity=0.5,
        )
        progress = []
        summary = run_nyu_batch(
            sensor,
            batch,
            progress_callback=lambda done, total, metrics: progress.append(
                (done, total, metrics.sample_id)
            ),
        )

        self.assertFalse(hasattr(batch, "sample_id"))
        self.assertEqual(summary.num_samples, 3)
        self.assertEqual(len(summary.sample_ids), 3)
        self.assertEqual(len(set(summary.sample_ids)), 3)
        self.assertEqual(len(summary.scene_metrics), 3)
        self.assertEqual(summary.total_pixels, 3 * 8 * 12)
        self.assertEqual(summary.possible_events, 3 * 8 * 12 * 50)
        self.assertEqual(progress[-1][:2], (3, 3))
        self.assertEqual(tuple(item[2] for item in progress), summary.sample_ids)
        self.assertGreaterEqual(summary.valid_pixel_fraction, 0.0)
        self.assertLessEqual(summary.valid_pixel_fraction, 1.0)
        self.assertGreaterEqual(summary.detection_fraction, 0.0)
        self.assertLessEqual(summary.detection_fraction, 1.0)
        self.assertTrue(np.isfinite(summary.expected_detection_fraction))
        self.assertTrue(np.isfinite(summary.root_mean_squared_error_m))


if __name__ == "__main__":
    unittest.main()
