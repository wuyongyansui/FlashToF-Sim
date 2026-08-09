"""聚合首光子采集的等宽直方图数据契约。"""

from dataclasses import dataclass

import numpy as np

from .first_photon import FirstPhotonAggregate


@dataclass(frozen=True)
class EquiWidthHistogram:
    """首光子 EWH 计数与未探测计数。

    ``counts_hwt`` 的 shape 为 ``[H, W, T]``。对每个像素，其计数总和加
    ``no_detection_counts_hw`` 必须等于 ``num_laser_periods``。
    """

    counts_hwt: np.ndarray
    no_detection_counts_hw: np.ndarray
    num_laser_periods: int
    num_time_bins: int

    @property
    def detected_counts_hw(self):
        return np.sum(self.counts_hwt, axis=-1, dtype=np.int64)


def accumulate_ewh(aggregate):
    """校验聚合首光子计数，并将其封装为 EWH。"""

    if not isinstance(aggregate, FirstPhotonAggregate):
        raise TypeError("aggregate must be FirstPhotonAggregate")
    counts = np.asarray(aggregate.counts_hwt)
    no_detection = np.asarray(aggregate.no_detection_counts_hw)
    if counts.ndim != 3:
        raise ValueError("counts_hwt must have shape [H, W, T]")
    if no_detection.shape != counts.shape[:2]:
        raise ValueError("no_detection_counts_hw must have shape [H, W]")
    if np.any(counts < 0) or np.any(no_detection < 0):
        raise ValueError("EWH counts cannot be negative")
    totals = np.sum(counts, axis=-1, dtype=np.int64) + no_detection.astype(np.int64)
    if not np.all(totals == aggregate.num_laser_periods):
        raise AssertionError("each pixel must have exactly one outcome per laser period")

    return EquiWidthHistogram(
        counts_hwt=np.ascontiguousarray(counts, dtype=np.int32),
        no_detection_counts_hw=np.ascontiguousarray(no_detection, dtype=np.int32),
        num_laser_periods=aggregate.num_laser_periods,
        num_time_bins=counts.shape[-1],
    )
