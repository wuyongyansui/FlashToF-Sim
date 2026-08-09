from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import numpy as np
from PIL import Image

from flash_dtof.batch import NYUBatchConfig, run_nyu_batch
from flash_dtof.config import CameraGeometryConfig, SensorConfig


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "_test_runs"


@contextmanager
def _small_native_nyu_dataset():
    """构造小型原生对齐 RGB-D 数据，验证无裁剪/缩放的批量路径。"""

    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_root = TEST_TEMP_ROOT / ("batch_" + uuid4().hex)
    dataset_root = case_root / "nyu-depth"
    for modality in ("images", "depth"):
        for split in ("train", "val"):
            (dataset_root / modality / split).mkdir(parents=True, exist_ok=True)

    ids_by_split = {
        "train": ("nyu_0004",),
        "val": ("nyu_0000", "nyu_0001", "nyu_0002", "nyu_0003"),
    }
    row = np.linspace(1.0, 3.0, 12, dtype=np.float32)
    for split, sample_ids in ids_by_split.items():
        for index, sample_id in enumerate(sample_ids):
            rgb = np.full((8, 12, 3), 40 + index * 20, dtype=np.uint8)
            depth = np.broadcast_to(row + index * 0.1, (8, 12)).astype(np.float16)
            Image.fromarray(rgb, mode="RGB").save(
                str(dataset_root / "images" / split / (sample_id + ".jpg"))
            )
            np.save(
                str(dataset_root / "depth" / split / (sample_id + ".npy")),
                depth,
                allow_pickle=False,
            )
    camera_path = case_root / "camera_params.m"
    camera_path.write_text(
        "fx_rgb = 1.0e+02;\n"
        "fy_rgb = 1.0e+02;\n"
        "cx_rgb = 6.5;\n"
        "cy_rgb = 4.5;\n",
        encoding="utf-8",
    )
    try:
        yield dataset_root, camera_path
    finally:
        resolved = case_root.resolve(strict=False)
        resolved.relative_to(TEST_TEMP_ROOT.resolve(strict=False))
        if case_root.exists():
            shutil.rmtree(str(case_root))


class NYUBatchPipelineTests(unittest.TestCase):
    def test_multi_sample_native_streaming_batch_smoke(self):
        # 此处特意使用小型原生 8×12 数据和少量周期，使集成测试保持快速。
        # 正式默认值仍为 480×640×672 和 20,000 个激光周期。
        sensor = SensorConfig(
            image_height=8,
            image_width=12,
            num_time_bins=96,
            bin_width_s=0.75e-9,
            num_laser_periods=50,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=2.5,
            background_photons_per_bin=1e-5,
            transient_model="gaussian",
            random_seed=31415,
        )
        with _small_native_nyu_dataset() as resources:
            dataset_root, camera_path = resources
            batch = NYUBatchConfig(
                dataset_root=dataset_root,
                split="val",
                start=1,
                limit=3,
                reflectivity_mode="constant",
                constant_reflectivity=0.5,
            )
            camera = CameraGeometryConfig(
                camera_params_path=camera_path,
                calibrated_image_size_wh=(12, 8),
            )
            progress = []
            summary = run_nyu_batch(
                sensor,
                batch,
                camera,
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
