"""仅实现向量化最大 bin 直接飞行时间重建。"""

from dataclasses import dataclass

import numpy as np

from .config import SPEED_OF_LIGHT_M_PER_S
from .ewh import EquiWidthHistogram


@dataclass(frozen=True)
class MaximumBinReconstruction:
    """最大计数 bin 图和单程斜距图，二者 shape 均为 ``[H, W]``。"""

    peak_bin_hw: np.ndarray
    estimated_distance_m_hw: np.ndarray
    valid_hw: np.ndarray


def bin_center_time_s(bin_index, bin_width_s):
    """将零基 EWH 索引转换为以秒计的时间 bin 中心。"""

    indices = np.asarray(bin_index)
    if np.any(indices < 0) or bin_width_s <= 0.0:
        raise ValueError("bin indices must be >= 0 and bin_width_s must be > 0")
    return (indices + 0.5) * bin_width_s


def round_trip_time_to_distance_m(round_trip_time_s):
    """将秒制往返时间转换为米制单程斜距。"""

    times = np.asarray(round_trip_time_s)
    if np.any(times < 0.0):
        raise ValueError("round_trip_time_s must be >= 0")
    return SPEED_OF_LIGHT_M_PER_S * times / 2.0


def reconstruct_maximum_bin(histogram, bin_width_s):
    """使用 ``argmax(axis=-1)`` 估计完整阵列单程斜距图。"""

    if not isinstance(histogram, EquiWidthHistogram):
        raise TypeError("histogram must be EquiWidthHistogram")
    if bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be > 0")

    detected = histogram.detected_counts_hw > 0
    peak = np.argmax(histogram.counts_hwt, axis=-1).astype(np.int16)
    distances = np.full(peak.shape, np.nan, dtype=np.float32)
    distances[detected] = (
        SPEED_OF_LIGHT_M_PER_S
        * ((peak[detected].astype(np.float64) + 0.5) * bin_width_s)
        / 2.0
    ).astype(np.float32)
    peak = np.where(detected, peak, -1).astype(np.int16)

    return MaximumBinReconstruction(
        peak_bin_hw=peak,
        estimated_distance_m_hw=distances,
        valid_hw=np.ascontiguousarray(detected),
    )
