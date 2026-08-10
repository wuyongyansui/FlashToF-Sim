"""仿真配置、轻量批量指标与单场景调试数组的安全落盘。"""

import csv
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import CameraGeometryConfig, SensorConfig, derive_config


@dataclass(frozen=True)
class OutputConfig:
    """用户设置的输出目录、实验名称、同名策略与调试 EWH 开关。

    ``existing_run_policy="increment"`` 会在同名目录已存在时追加
    ``__002``、``__003``；``"error"`` 会直接报错。两种策略都不会覆盖
    已有运行结果。
    """

    output_root: Path
    run_name: str
    existing_run_policy: str = "increment"
    save_debug_ewh: bool = True

    def __post_init__(self):
        root = Path(self.output_root).expanduser()
        _validate_path_component(self.run_name, "run_name")
        if self.existing_run_policy not in ("increment", "error"):
            raise ValueError("existing_run_policy must be 'increment' or 'error'")
        if not isinstance(self.save_debug_ewh, bool):
            raise ValueError("save_debug_ewh must be a boolean")
        object.__setattr__(self, "output_root", root)


def create_run_directory(
    output_config: OutputConfig,
    run_kind: str,
    protected_paths: Iterable[Path] = (),
):
    """创建一个全新运行目录，并拒绝写入受保护的数据集目录。

    目录名称为 ``{run_name}_{run_kind}``。默认同名时递增后缀；本函数绝不
    复用或覆盖已有运行目录。
    """

    if not isinstance(output_config, OutputConfig):
        raise TypeError("output_config must be OutputConfig")
    _validate_path_component(run_kind, "run_kind")

    output_root = output_config.output_root
    base = output_root / "{}_{}".format(output_config.run_name, run_kind)
    _ensure_outside_protected_paths(base, protected_paths)

    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError("output_root is not a directory: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    if output_config.existing_run_policy == "error":
        try:
            base.mkdir(exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                "run directory already exists and will not be overwritten: {}".format(base)
            )
        return base

    index = 1
    while True:
        candidate = base if index == 1 else Path("{}__{:03d}".format(base, index))
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1


def initialize_batch_output(output_config, sensor_config, camera_config, batch_config):
    """创建正式批量目录，并先写入配置快照、运行说明和运行中状态。"""

    run_directory = create_run_directory(
        output_config,
        "batch",
        protected_paths=(batch_config.dataset_root,),
    )
    _write_config_snapshot(
        run_directory,
        mode="nyu_streaming_batch",
        sensor_config=sensor_config,
        camera_config=camera_config,
        data_config=batch_config,
        output_config=output_config,
        extra={
            "storage_policy": (
                "仅保存汇总 JSON 与逐场景 CSV；不保存批量瞬态、EWH、距离图或 RGB-D 数组"
            )
        },
    )
    _write_text_new(run_directory / "RUN_README.txt", _batch_readme_text())
    write_run_status(run_directory, "running")
    return run_directory


def initialize_debug_output(
    output_config,
    sensor_config,
    camera_config,
    batch_config,
    split,
    sample_id,
):
    """创建独立单场景调试目录，并写入配置快照与运行说明。"""

    _validate_path_component(split, "split")
    _validate_path_component(sample_id, "sample_id")
    run_kind = "debug_{}_{}".format(split, sample_id)
    run_directory = create_run_directory(
        output_config,
        run_kind,
        protected_paths=(batch_config.dataset_root,),
    )
    _write_config_snapshot(
        run_directory,
        mode="single_scene_debug",
        sensor_config=sensor_config,
        camera_config=camera_config,
        data_config=batch_config,
        output_config=output_config,
        extra={"split": split, "sample_id": sample_id},
    )
    _write_text_new(
        run_directory / "RUN_README.txt",
        _debug_readme_text(output_config.save_debug_ewh, sensor_config),
    )
    write_run_status(run_directory, "running")
    return run_directory


def save_batch_results(run_directory, summary):
    """保存正式批量汇总 JSON 与每场景一行的 UTF-8 CSV。"""

    run_directory = _require_run_directory(run_directory)
    summary_payload = {
        field.name: getattr(summary, field.name)
        for field in fields(summary)
        if field.name not in ("sample_ids", "scene_metrics")
    }
    summary_payload.update(
        {
            "first_sample_id": summary.sample_ids[0],
            "last_sample_id": summary.sample_ids[-1],
        }
    )
    _write_json_new(run_directory / "summary_metrics.json", summary_payload)

    csv_path = run_directory / "scene_metrics.csv"
    field_names = [field.name for field in fields(summary.scene_metrics[0])]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for metrics in summary.scene_metrics:
            writer.writerow(asdict(metrics))
    write_run_status(run_directory, "completed")
    return {
        "summary_metrics": run_directory / "summary_metrics.json",
        "scene_metrics": csv_path,
    }


def save_debug_results(run_directory, loaded_scene, result, output_config):
    """保存单场景输入、EWH、距离图、诊断数组与结构化诊断 JSON。"""

    run_directory = _require_run_directory(run_directory)
    if not isinstance(output_config, OutputConfig):
        raise TypeError("output_config must be OutputConfig")
    arrays = {
        "input_depth_z_m.npy": result.scene_inputs.depth_z_m,
        "input_reflectivity.npy": result.scene_inputs.reflectivity,
        "true_slant_range_m.npy": result.scene_geometry.slant_range_m_hw,
        "ray_direction_z.npy": result.scene_geometry.ray_direction_z_hw,
        "reconstructed_slant_range_m.npy": (
            result.reconstruction.estimated_distance_m_hw
        ),
        "valid_mask.npy": result.reconstruction.valid_hw,
        "detected_counts.npy": result.ewh.detected_counts_hw,
        "no_detection_counts.npy": result.ewh.no_detection_counts_hw,
        "ideal_peak_bin.npy": result.diagnostics.ideal_peak_bin_hw,
        "observed_peak_bin.npy": result.diagnostics.observed_peak_bin_hw,
        "peak_shift_bins.npy": result.diagnostics.peak_shift_bins_hw,
        "measured_detection_fraction.npy": (
            result.diagnostics.measured_detection_fraction_hw
        ),
        "expected_detection_fraction.npy": (
            result.diagnostics.expected_detection_fraction_hw
        ),
        "slant_range_bias_m.npy": result.diagnostics.slant_range_bias_m_hw,
        "depth_to_slant_delta_m.npy": (
            result.diagnostics.depth_to_slant_delta_m_hw
        ),
    }
    if output_config.save_debug_ewh:
        arrays["ewh_counts.npy"] = result.ewh.counts_hwt
    array_metadata = {}
    for filename, array in arrays.items():
        path = run_directory / filename
        if path.exists():
            raise FileExistsError("debug array will not be overwritten: {}".format(path))
        np.save(str(path), array, allow_pickle=False)
        array_metadata[filename] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "unit_or_meaning": _array_meaning(filename),
        }

    valid = result.reconstruction.valid_hw
    if np.any(valid):
        bias = result.diagnostics.slant_range_bias_m_hw[valid].astype(np.float64)
        mean_bias = float(np.mean(bias))
        mean_absolute_error = float(np.mean(np.abs(bias)))
        root_mean_squared_error = float(np.sqrt(np.mean(bias * bias)))
    else:
        mean_bias = float("nan")
        mean_absolute_error = float("nan")
        root_mean_squared_error = float("nan")

    total_pixels = int(valid.size)
    detected_events = int(np.sum(result.ewh.detected_counts_hw, dtype=np.int64))
    possible_events = result.sensor_config.num_laser_periods * total_pixels
    diagnostics = {
        "sample_id": loaded_scene.sample_id,
        "split": loaded_scene.split,
        "source_size_wh": loaded_scene.source_size_wh,
        "geometry_transform": loaded_scene.geometry_transform,
        "depth_semantics": "rgb_optical_axis_z",
        "range_semantics": "slant_range_equals_depth_z_divided_by_unit_ray_d_z",
        "rgb_intrinsics_k": result.camera_intrinsics.matrix_k.tolist(),
        "pixel_coordinate_convention": (
            result.camera_intrinsics.pixel_coordinate_convention
        ),
        "transient_response_model": result.ideal_transient.response_model,
        "output_shape_hw": list(result.scene_inputs.shape_hw),
        "ewh_shape_hwt": list(result.ewh.counts_hwt.shape),
        "ewh_saved": output_config.save_debug_ewh,
        "valid_pixel_fraction": float(np.mean(valid)),
        "detection_fraction": detected_events / possible_events,
        "expected_detection_fraction": float(
            np.mean(result.diagnostics.expected_detection_fraction_hw)
        ),
        "mean_bias_m": mean_bias,
        "mean_absolute_error_m": mean_absolute_error,
        "root_mean_squared_error_m": root_mean_squared_error,
        "array_files": array_metadata,
        "not_saved": [
            "RGB 图像",
            "理想瞬态数组；可由输入数组和配置快照确定性重建",
        ]
        + ([] if output_config.save_debug_ewh else ["完整 EWH；已由 save_debug_ewh=False 关闭"]),
    }
    measured_irf = result.ideal_transient.measured_irf
    if measured_irf is not None:
        diagnostics["measured_irf"] = {
            "source_path": str(measured_irf.source_path),
            "sample_count": int(measured_irf.probability_mass.size),
            "sample_interval_s": measured_irf.sample_interval_s,
            "peak_index": measured_irf.peak_index,
            "negative_mass_fraction_before_clip": (
                measured_irf.negative_mass_fraction_before_clip
            ),
            "normalized_mass_sum": float(
                np.sum(measured_irf.probability_mass, dtype=np.float64)
            ),
        }
    _write_json_new(run_directory / "diagnostics.json", diagnostics)
    write_run_status(run_directory, "completed")
    return {
        "diagnostics": run_directory / "diagnostics.json",
        "arrays": tuple(run_directory / name for name in arrays),
    }


def write_run_status(run_directory, status, error=None):
    """更新当前新建运行目录的状态文件；不会触碰其他运行目录。"""

    run_directory = _require_run_directory(run_directory)
    if status not in ("running", "completed", "failed"):
        raise ValueError("status must be running, completed, or failed")
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    with (run_directory / "run_status.json").open("w", encoding="utf-8") as stream:
        json.dump(_to_jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_config_snapshot(
    run_directory,
    mode,
    sensor_config,
    camera_config,
    data_config,
    output_config,
    extra=None,
):
    if not isinstance(sensor_config, SensorConfig):
        raise TypeError("sensor_config must be SensorConfig")
    if not isinstance(camera_config, CameraGeometryConfig):
        raise TypeError("camera_config must be CameraGeometryConfig")
    payload = {
        "schema_version": 2,
        "mode": mode,
        "run_directory": str(run_directory.resolve()),
        "sensor_config": sensor_config,
        "camera_geometry_config": camera_config,
        "derived_config": derive_config(sensor_config),
        "data_config": data_config,
        "output_config": output_config,
    }
    if extra:
        payload["run_metadata"] = extra
    _write_json_new(run_directory / "config_snapshot.json", payload)


def _write_json_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_to_jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_text_new(path, text):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text.rstrip() + "\n")


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require_run_directory(run_directory):
    path = Path(run_directory)
    if not path.is_dir():
        raise FileNotFoundError("run directory does not exist: {}".format(path))
    return path


def _validate_path_component(value, field_name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("{} must be a non-empty name without surrounding spaces".format(field_name))
    if value in (".", "..") or value.endswith((".", " ")):
        raise ValueError("{} is not a safe directory name".format(field_name))
    forbidden = set('<>:"/\\|?*')
    if any(character in forbidden or ord(character) < 32 for character in value):
        raise ValueError("{} contains an invalid path character".format(field_name))


def _ensure_outside_protected_paths(candidate, protected_paths):
    candidate_resolved = Path(candidate).resolve(strict=False)
    for protected in protected_paths:
        protected_resolved = Path(protected).resolve(strict=False)
        if _is_within(candidate_resolved, protected_resolved):
            raise ValueError(
                "output directory must not be inside protected dataset path: {}".format(
                    protected_resolved
                )
            )


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _batch_readme_text():
    return """正式 NYU 流式批量运行目录

文件：
- config_snapshot.json：本次相机几何、传感器、数据选择、输出配置与派生参数快照。
- summary_metrics.json：跨全部已选场景、按像素加权的汇总指标。
- scene_metrics.csv：每个场景一行的轻量指标，UTF-8 with BOM。
- run_status.json：running、completed 或 failed 状态。

存储边界：正式批量路径不会保存每个场景的 RGB-D、理想瞬态、EWH 或距离图。
这是为了避免完整数据集产生巨量文件；需要检查完整数组时请使用独立单场景调试入口。
"""


def _debug_readme_text(save_debug_ewh, sensor_config):
    ewh_bytes = (
        sensor_config.image_height
        * sensor_config.image_width
        * sensor_config.num_time_bins
        * np.dtype(np.int32).itemsize
    )
    ewh_description = (
        "本次配置会保存完整 ewh_counts.npy；当前 {}×{}×{} 的 int32 EWH "
        "约为 {:.2f} MiB（{:.2f} MB）。".format(
            sensor_config.image_height,
            sensor_config.image_width,
            sensor_config.num_time_bins,
            ewh_bytes / (1024.0 ** 2),
            ewh_bytes / 1e6,
        )
        if save_debug_ewh
        else "本次配置已关闭完整 ewh_counts.npy，以控制原生分辨率调试输出大小。"
    )
    return """独立单场景调试运行目录

本目录保存输入 RGB 轴向 depth z、真实斜距、射线 d_z、reflectivity、重建
斜距图，以及探测、无探测、峰位和斜距偏差等诊断数组。diagnostics.json 记录 shape、dtype、
单位和标量汇总，config_snapshot.json 记录可复现实验所需配置。

{}

RGB 与理想瞬态不重复落盘；理想瞬态可由输入数组和配置快照重新生成。
所有 NPY 均禁用 pickle，可用 numpy.load(path, allow_pickle=False) 读取。
""".format(ewh_description)


def _array_meaning(filename):
    meanings = {
        "input_depth_z_m.npy": "RGB 光轴方向米制轴向深度 z [H,W]",
        "input_reflectivity.npy": "无量纲反射率 [H,W]；来自常数或 RGB 相对代理配置",
        "true_slant_range_m.npy": "由 z/d_z 得到的米制真实斜距 [H,W]",
        "ray_direction_z.npy": "RGB 单位像素射线的 z 分量 d_z [H,W]",
        "ewh_counts.npy": "每 bin 首光子计数 [H,W,T]",
        "reconstructed_slant_range_m.npy": "最大-bin 重建米制斜距 [H,W]",
        "valid_mask.npy": "是否至少有一次探测 [H,W]",
        "detected_counts.npy": "各像素累计探测周期数 [H,W]",
        "no_detection_counts.npy": "各像素无探测周期数 [H,W]",
        "ideal_peak_bin.npy": "理想瞬态最大 bin [H,W]",
        "observed_peak_bin.npy": "EWH 最大 bin [H,W]",
        "peak_shift_bins.npy": "观测峰相对理想峰的 bin 偏移 [H,W]",
        "measured_detection_fraction.npy": "实测探测周期比例 [H,W]",
        "expected_detection_fraction.npy": "理论探测周期概率 [H,W]",
        "slant_range_bias_m.npy": "重建斜距减真实斜距，单位米 [H,W]",
        "depth_to_slant_delta_m.npy": "真实斜距减轴向深度 z，单位米 [H,W]",
    }
    return meanings[filename]
