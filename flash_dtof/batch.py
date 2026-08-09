"""面向完整阵列仿真器的 NYU split 流式评估。"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import SensorConfig
from .pipeline import run_simulation
from .scene import NYUDepthV2Loader


@dataclass(frozen=True)
class NYUBatchConfig:
    """用户设置的 NYU split 选择参数，不包含单样本正式入口。

    ``limit=None`` 表示评估从 ``start`` 起的所有剩余配对样本。运行器始终
    逐场景执行加载、仿真、汇总和释放。
    """

    dataset_root: Path
    split: str = "val"
    start: int = 0
    limit: Optional[int] = None
    reflectivity_mode: str = "constant"
    constant_reflectivity: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        if self.split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if self.limit is not None and (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or self.limit <= 0
        ):
            raise ValueError("limit must be None or a positive integer")
        if self.reflectivity_mode not in ("constant", "luminance_proxy"):
            raise ValueError("invalid reflectivity_mode")
        if not 0.0 <= self.constant_reflectivity <= 1.0:
            raise ValueError("constant_reflectivity must be in [0, 1]")


@dataclass(frozen=True)
class SceneBatchMetrics:
    """轻量逐场景指标，不包含 RGB-D、瞬态或 EWH 数组。"""

    sample_id: str
    random_seed: int
    total_pixels: int
    valid_pixels: int
    detected_events: int
    possible_events: int
    detection_fraction: float
    expected_detection_fraction: float
    mean_bias_m: float
    mean_absolute_error_m: float
    root_mean_squared_error_m: float


@dataclass(frozen=True)
class NYUBatchSummary:
    """数据选择元信息与按像素加权的汇总指标。"""

    split: str
    selection_start: int
    selection_limit: Optional[int]
    sample_ids: tuple
    num_samples: int
    total_pixels: int
    valid_pixels: int
    detected_events: int
    possible_events: int
    valid_pixel_fraction: float
    detection_fraction: float
    expected_detection_fraction: float
    mean_bias_m: float
    mean_absolute_error_m: float
    root_mean_squared_error_m: float
    scene_metrics: tuple


def run_nyu_batch(
    sensor_config: SensorConfig,
    batch_config: NYUBatchConfig,
    progress_callback: Optional[Callable] = None,
):
    """让选定的 NYU split 流式通过完整阵列仿真链路。

    跨迭代仅保留样本 ID 和轻量指标；加载下一场景前释放当前场景的大型数组。
    """

    if not isinstance(sensor_config, SensorConfig):
        raise TypeError("sensor_config must be SensorConfig")
    if not isinstance(batch_config, NYUBatchConfig):
        raise TypeError("batch_config must be NYUBatchConfig")

    loader = NYUDepthV2Loader(
        batch_config.dataset_root,
        output_height=sensor_config.image_height,
        output_width=sensor_config.image_width,
        reflectivity_mode=batch_config.reflectivity_mode,
        constant_reflectivity=batch_config.constant_reflectivity,
    )
    sample_ids = loader.select_sample_ids(
        batch_config.split,
        start=batch_config.start,
        limit=batch_config.limit,
    )
    if not sample_ids:
        raise ValueError("NYU batch selection is empty")

    scene_metrics = []
    total_pixels = 0
    valid_pixels = 0
    detected_events = 0
    possible_events = 0
    expected_detection_sum = 0.0
    bias_sum = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0

    for local_index, sample_id in enumerate(sample_ids):
        global_index = batch_config.start + local_index
        scene_seed = _derive_scene_seed(sensor_config.random_seed, global_index)
        scene_sensor = replace(sensor_config, random_seed=scene_seed)
        loaded = loader.load(sample_id, split=batch_config.split)
        result = run_simulation(scene_sensor, loaded.scene_inputs)
        metrics, sums = _summarize_scene(sample_id, scene_seed, result)
        scene_metrics.append(metrics)

        total_pixels += metrics.total_pixels
        valid_pixels += metrics.valid_pixels
        detected_events += metrics.detected_events
        possible_events += metrics.possible_events
        expected_detection_sum += sums["expected_detection"]
        bias_sum += sums["bias"]
        absolute_error_sum += sums["absolute_error"]
        squared_error_sum += sums["squared_error"]

        if progress_callback is not None:
            progress_callback(local_index + 1, len(sample_ids), metrics)
        del result
        del loaded

    if valid_pixels:
        mean_bias = bias_sum / valid_pixels
        mean_absolute_error = absolute_error_sum / valid_pixels
        root_mean_squared_error = float(np.sqrt(squared_error_sum / valid_pixels))
    else:
        mean_bias = float("nan")
        mean_absolute_error = float("nan")
        root_mean_squared_error = float("nan")

    return NYUBatchSummary(
        split=batch_config.split,
        selection_start=batch_config.start,
        selection_limit=batch_config.limit,
        sample_ids=tuple(sample_ids),
        num_samples=len(sample_ids),
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        detected_events=detected_events,
        possible_events=possible_events,
        valid_pixel_fraction=valid_pixels / total_pixels,
        detection_fraction=detected_events / possible_events,
        expected_detection_fraction=expected_detection_sum / total_pixels,
        mean_bias_m=mean_bias,
        mean_absolute_error_m=mean_absolute_error,
        root_mean_squared_error_m=root_mean_squared_error,
        scene_metrics=tuple(scene_metrics),
    )


def _derive_scene_seed(base_seed, global_index):
    state = np.random.SeedSequence([base_seed, global_index]).generate_state(1)
    return int(state[0])


def _summarize_scene(sample_id, scene_seed, result):
    valid = result.reconstruction.valid_hw
    total_pixels = int(valid.size)
    valid_pixels = int(np.count_nonzero(valid))
    detected_events = int(
        np.sum(result.ewh.detected_counts_hw, dtype=np.int64)
    )
    possible_events = result.sensor_config.num_laser_periods * total_pixels
    expected_detection_sum = float(
        np.sum(
            result.diagnostics.expected_detection_fraction_hw,
            dtype=np.float64,
        )
    )

    if valid_pixels:
        bias = result.diagnostics.distance_bias_m_hw[valid].astype(np.float64)
        bias_sum = float(np.sum(bias, dtype=np.float64))
        absolute_error_sum = float(np.sum(np.abs(bias), dtype=np.float64))
        squared_error_sum = float(np.sum(bias * bias, dtype=np.float64))
        mean_bias = bias_sum / valid_pixels
        mean_absolute_error = absolute_error_sum / valid_pixels
        root_mean_squared_error = float(np.sqrt(squared_error_sum / valid_pixels))
    else:
        bias_sum = 0.0
        absolute_error_sum = 0.0
        squared_error_sum = 0.0
        mean_bias = float("nan")
        mean_absolute_error = float("nan")
        root_mean_squared_error = float("nan")

    metrics = SceneBatchMetrics(
        sample_id=sample_id,
        random_seed=scene_seed,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        detected_events=detected_events,
        possible_events=possible_events,
        detection_fraction=detected_events / possible_events,
        expected_detection_fraction=expected_detection_sum / total_pixels,
        mean_bias_m=mean_bias,
        mean_absolute_error_m=mean_absolute_error,
        root_mean_squared_error_m=root_mean_squared_error,
    )
    sums = {
        "expected_detection": expected_detection_sum,
        "bias": bias_sum,
        "absolute_error": absolute_error_sum,
        "squared_error": squared_error_sum,
    }
    return metrics, sums


def format_batch_summary(summary):
    """格式化正式 split 选择信息与汇总评估结果。"""

    if not isinstance(summary, NYUBatchSummary):
        raise TypeError("summary must be NYUBatchSummary")
    return "\n".join(
        [
            "NYU STREAMING BATCH SUMMARY",
            "  split / samples             : {} / {}".format(
                summary.split, summary.num_samples
            ),
            "  selection start / limit     : {} / {}".format(
                summary.selection_start,
                "all remaining" if summary.selection_limit is None else summary.selection_limit,
            ),
            "  first / last sample         : {} / {}".format(
                summary.sample_ids[0], summary.sample_ids[-1]
            ),
            "  valid pixel fraction        : {:.6f}".format(
                summary.valid_pixel_fraction
            ),
            "  detection fraction          : {:.6f}".format(
                summary.detection_fraction
            ),
            "  expected detection fraction : {:.6f}".format(
                summary.expected_detection_fraction
            ),
            "  distance bias / MAE / RMSE  : {:+.6f} / {:.6f} / {:.6f} m".format(
                summary.mean_bias_m,
                summary.mean_absolute_error_m,
                summary.root_mean_squared_error_m,
            ),
            "  retained dataset arrays     : none (per-scene streaming)",
        ]
    )
