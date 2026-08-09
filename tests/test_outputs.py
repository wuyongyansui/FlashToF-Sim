import csv
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import numpy as np

from flash_dtof.batch import (
    NYUBatchConfig,
    NYUBatchSummary,
    SceneBatchMetrics,
)
from flash_dtof.config import CameraGeometryConfig, SensorConfig, make_uniform_scene
from flash_dtof.geometry import RGBIntrinsics
from flash_dtof.output import (
    OutputConfig,
    create_run_directory,
    initialize_batch_output,
    initialize_debug_output,
    save_batch_results,
    save_debug_results,
)
from flash_dtof.pipeline import run_simulation
from flash_dtof.scene import LoadedNYUScene


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "_test_runs"


@contextmanager
def _workspace_temporary_directory():
    """在工作区内创建可写临时目录，并在测试后安全删除。"""

    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    directory = TEST_TEMP_ROOT / ("case_" + uuid4().hex)
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        resolved = directory.resolve(strict=False)
        resolved.relative_to(TEST_TEMP_ROOT.resolve(strict=False))
        if directory.exists():
            shutil.rmtree(str(directory))


class OutputTests(unittest.TestCase):
    def _sensor(self):
        # 测试使用小阵列和少量周期；正式默认值仍是 480×640×672、20,000 周期。
        return SensorConfig(
            image_height=2,
            image_width=3,
            num_time_bins=64,
            bin_width_s=0.75e-9,
            num_laser_periods=30,
            signal_photons_per_pulse_at_reference=0.05,
            reference_distance_m=2.5,
            background_photons_per_bin=1e-5,
            transient_model="gaussian",
            random_seed=123,
        )

    def _camera_config(self, path):
        return CameraGeometryConfig(
            camera_params_path=path,
            calibrated_image_size_wh=(3, 2),
        )

    def _intrinsics(self):
        return RGBIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=2.0,
            cy=1.5,
            image_size_wh=(3, 2),
            source_path="synthetic.m",
        )

    def _batch_config(self, dataset_root):
        return NYUBatchConfig(
            dataset_root=dataset_root,
            split="val",
            start=0,
            limit=2,
            reflectivity_mode="constant",
            constant_reflectivity=0.5,
        )

    def test_same_name_increments_or_reports_error_without_overwrite(self):
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary) / "results"
            increment = OutputConfig(root, "实验A", "increment")
            first = create_run_directory(increment, "batch")
            second = create_run_directory(increment, "batch")
            self.assertEqual(first.name, "实验A_batch")
            self.assertEqual(second.name, "实验A_batch__002")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

            strict = OutputConfig(root, "实验A", "error")
            with self.assertRaises(FileExistsError):
                create_run_directory(strict, "batch")

    def test_output_inside_dataset_is_rejected_before_any_write(self):
        with _workspace_temporary_directory() as temporary:
            dataset_root = Path(temporary) / "nyu-depth"
            dataset_root.mkdir()
            output = OutputConfig(dataset_root / "generated", "unsafe")
            with self.assertRaisesRegex(ValueError, "protected dataset"):
                initialize_batch_output(
                    output,
                    self._sensor(),
                    self._camera_config(Path(temporary) / "camera_params.m"),
                    self._batch_config(dataset_root),
                )
            self.assertEqual(list(dataset_root.iterdir()), [])

    def test_batch_writes_snapshot_summary_and_one_csv_row_per_scene(self):
        with _workspace_temporary_directory() as temporary:
            base = Path(temporary)
            dataset_root = base / "nyu-depth"
            dataset_root.mkdir()
            output = OutputConfig(base / "outputs", "batch_case")
            run_directory = initialize_batch_output(
                output,
                self._sensor(),
                self._camera_config(base / "camera_params.m"),
                self._batch_config(dataset_root),
            )

            first = SceneBatchMetrics(
                sample_id="nyu_0000",
                random_seed=11,
                total_pixels=6,
                valid_pixels=6,
                detected_events=12,
                possible_events=180,
                detection_fraction=12 / 180,
                expected_detection_fraction=0.07,
                mean_bias_m=0.01,
                mean_absolute_error_m=0.02,
                root_mean_squared_error_m=0.03,
            )
            second = SceneBatchMetrics(
                sample_id="nyu_0001",
                random_seed=12,
                total_pixels=6,
                valid_pixels=5,
                detected_events=10,
                possible_events=180,
                detection_fraction=10 / 180,
                expected_detection_fraction=0.06,
                mean_bias_m=-0.01,
                mean_absolute_error_m=0.02,
                root_mean_squared_error_m=0.025,
            )
            summary = NYUBatchSummary(
                split="val",
                selection_start=0,
                selection_limit=2,
                sample_ids=(first.sample_id, second.sample_id),
                num_samples=2,
                total_pixels=12,
                valid_pixels=11,
                detected_events=22,
                possible_events=360,
                valid_pixel_fraction=11 / 12,
                detection_fraction=22 / 360,
                expected_detection_fraction=0.065,
                mean_bias_m=0.0,
                mean_absolute_error_m=0.02,
                root_mean_squared_error_m=0.028,
                scene_metrics=(first, second),
            )
            save_batch_results(run_directory, summary)

            snapshot = json.loads(
                (run_directory / "config_snapshot.json").read_text(encoding="utf-8")
            )
            aggregate = json.loads(
                (run_directory / "summary_metrics.json").read_text(encoding="utf-8")
            )
            with (run_directory / "scene_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            status = json.loads(
                (run_directory / "run_status.json").read_text(encoding="utf-8")
            )

            self.assertEqual(snapshot["mode"], "nyu_streaming_batch")
            self.assertEqual(snapshot["sensor_config"]["num_laser_periods"], 30)
            self.assertEqual(
                snapshot["camera_geometry_config"]["depth_semantics"],
                "rgb_optical_axis_z",
            )
            self.assertEqual(aggregate["num_samples"], 2)
            self.assertEqual(aggregate["first_sample_id"], "nyu_0000")
            self.assertEqual([row["sample_id"] for row in rows], ["nyu_0000", "nyu_0001"])
            self.assertEqual(status["status"], "completed")
            self.assertFalse(any(run_directory.glob("*.npy")))
            self.assertEqual(list(dataset_root.iterdir()), [])

    def test_single_scene_writes_required_arrays_and_structured_diagnostics(self):
        with _workspace_temporary_directory() as temporary:
            base = Path(temporary)
            dataset_root = base / "nyu-depth"
            dataset_root.mkdir()
            sensor = self._sensor()
            scene = make_uniform_scene(sensor, depth_z_m=1.5, reflectivity=0.5)
            result = run_simulation(sensor, scene, self._intrinsics())
            loaded = LoadedNYUScene(
                sample_id="nyu_0000",
                split="val",
                rgb_u8_hwc=np.zeros((2, 3, 3), dtype=np.uint8),
                scene_inputs=scene,
                source_size_wh=(3, 2),
                geometry_transform="native_identity",
            )
            output = OutputConfig(base / "outputs", "debug_case", save_debug_ewh=True)
            run_directory = initialize_debug_output(
                output,
                sensor,
                self._camera_config(base / "camera_params.m"),
                self._batch_config(dataset_root),
                "val",
                "nyu_0000",
            )
            saved = save_debug_results(run_directory, loaded, result, output)

            depth = np.load(run_directory / "input_depth_z_m.npy", allow_pickle=False)
            reflectivity = np.load(
                run_directory / "input_reflectivity.npy", allow_pickle=False
            )
            ewh = np.load(run_directory / "ewh_counts.npy", allow_pickle=False)
            distance = np.load(
                run_directory / "reconstructed_slant_range_m.npy", allow_pickle=False
            )
            diagnostics = json.loads(
                (run_directory / "diagnostics.json").read_text(encoding="utf-8")
            )

            np.testing.assert_array_equal(depth, scene.depth_z_m)
            np.testing.assert_array_equal(reflectivity, scene.reflectivity)
            np.testing.assert_array_equal(ewh, result.ewh.counts_hwt)
            np.testing.assert_array_equal(
                distance, result.reconstruction.estimated_distance_m_hw
            )
            self.assertEqual(ewh.shape, (2, 3, 64))
            self.assertEqual(diagnostics["sample_id"], "nyu_0000")
            self.assertEqual(diagnostics["geometry_transform"], "native_identity")
            self.assertEqual(diagnostics["depth_semantics"], "rgb_optical_axis_z")
            self.assertTrue(diagnostics["ewh_saved"])
            self.assertIn("ewh_counts.npy", diagnostics["array_files"])
            self.assertIn("true_slant_range_m.npy", diagnostics["array_files"])
            self.assertIn("ray_direction_z.npy", diagnostics["array_files"])
            self.assertEqual(len(saved["arrays"]), 16)
            self.assertEqual(list(dataset_root.iterdir()), [])

    def test_single_scene_can_skip_large_ewh_file(self):
        with _workspace_temporary_directory() as temporary:
            base = Path(temporary)
            dataset_root = base / "nyu-depth"
            dataset_root.mkdir()
            sensor = self._sensor()
            scene = make_uniform_scene(sensor, depth_z_m=1.5, reflectivity=0.5)
            result = run_simulation(sensor, scene, self._intrinsics())
            loaded = LoadedNYUScene(
                sample_id="nyu_0000",
                split="val",
                rgb_u8_hwc=np.zeros((2, 3, 3), dtype=np.uint8),
                scene_inputs=scene,
                source_size_wh=(3, 2),
                geometry_transform="native_identity",
            )
            output = OutputConfig(
                base / "outputs",
                "debug_without_ewh",
                save_debug_ewh=False,
            )
            run_directory = initialize_debug_output(
                output,
                sensor,
                self._camera_config(base / "camera_params.m"),
                self._batch_config(dataset_root),
                "val",
                "nyu_0000",
            )
            saved = save_debug_results(run_directory, loaded, result, output)
            diagnostics = json.loads(
                (run_directory / "diagnostics.json").read_text(encoding="utf-8")
            )

            self.assertFalse((run_directory / "ewh_counts.npy").exists())
            self.assertFalse(diagnostics["ewh_saved"])
            self.assertEqual(len(saved["arrays"]), 15)


if __name__ == "__main__":
    unittest.main()
