"""SP-TransientBench 实测 IRF 的只读解析、校验与离散质量映射。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeasuredIRF:
    """以相对峰时刻为零的离散 IRF 概率质量模板。"""

    source_path: Path
    relative_time_s: np.ndarray
    probability_mass: np.ndarray
    irf_std: np.ndarray
    sample_interval_s: float
    peak_index: int
    negative_mass_fraction_before_clip: float


def load_measured_irf(path, expected_bin_width_s=None):
    """读取逗号分隔的 ``t_ps,irf,irf_std`` 并返回非负归一化质量。

    原始文件不会被修改。允许拟合产生的微小负残留，但其绝对质量不得超过
    正质量的 0.5%；使用前统一截断为零并按离散样本和重新归一化。
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("measured IRF file not found: {}".format(source_path))
    try:
        data = np.genfromtxt(
            str(source_path),
            delimiter=",",
            names=True,
            dtype=np.float64,
            encoding="ascii",
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("failed to parse measured IRF CSV: {}".format(error))
    if data.dtype.names != ("t_ps", "irf", "irf_std"):
        raise ValueError("measured IRF header must be exactly t_ps,irf,irf_std")
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    if data.shape[0] < 3:
        raise ValueError("measured IRF must contain at least three samples")

    time_ps = np.asarray(data["t_ps"], dtype=np.float64)
    raw_irf = np.asarray(data["irf"], dtype=np.float64)
    irf_std = np.asarray(data["irf_std"], dtype=np.float64)
    if not np.all(np.isfinite(time_ps)) or not np.all(np.isfinite(raw_irf)):
        raise ValueError("IRF time and value columns must be finite")
    if not np.all(np.isfinite(irf_std)) or np.any(irf_std < 0.0):
        raise ValueError("irf_std must be finite and non-negative")
    intervals_ps = np.diff(time_ps)
    if np.any(intervals_ps <= 0.0):
        raise ValueError("IRF time coordinates must be strictly increasing")
    interval_ps = float(intervals_ps[0])
    if not np.allclose(intervals_ps, interval_ps, rtol=1e-10, atol=1e-9):
        raise ValueError("IRF time coordinates must be uniformly sampled")
    sample_interval_s = interval_ps * 1e-12
    if expected_bin_width_s is not None and not np.isclose(
        sample_interval_s, expected_bin_width_s, rtol=1e-9, atol=1e-18
    ):
        raise ValueError(
            "IRF sample interval {:.12g} s does not match bin width {:.12g} s".format(
                sample_interval_s, expected_bin_width_s
            )
        )

    zero_indices = np.flatnonzero(np.isclose(time_ps, 0.0, rtol=0.0, atol=1e-9))
    if zero_indices.size != 1:
        raise ValueError("IRF time coordinates must contain exactly one 0 ps sample")
    peak_index = int(np.argmax(raw_irf))
    if peak_index != int(zero_indices[0]):
        raise ValueError("IRF maximum must occur at the 0 ps reference sample")

    positive_sum = float(np.sum(np.clip(raw_irf, 0.0, None), dtype=np.float64))
    negative_sum = float(np.sum(np.clip(-raw_irf, 0.0, None), dtype=np.float64))
    if positive_sum <= 0.0:
        raise ValueError("IRF must contain positive mass")
    negative_fraction = negative_sum / positive_sum
    if negative_fraction > 0.005:
        raise ValueError("IRF negative residual mass exceeds the 0.5% tolerance")
    mass = np.clip(raw_irf, 0.0, None)
    mass /= np.sum(mass, dtype=np.float64)

    relative_time_s = time_ps * 1e-12
    return MeasuredIRF(
        source_path=source_path,
        relative_time_s=np.ascontiguousarray(relative_time_s, dtype=np.float64),
        probability_mass=np.ascontiguousarray(mass, dtype=np.float64),
        irf_std=np.ascontiguousarray(irf_std, dtype=np.float64),
        sample_interval_s=sample_interval_s,
        peak_index=peak_index,
        negative_mass_fraction_before_clip=negative_fraction,
    )


def shifted_irf_mass_at_bin_centers(measured_irf, round_trip_time_hw, bin_centers_s):
    """把相对 IRF 平移到逐像素 ToF，并在目标 bin 中返回离散质量。

    源 IRF 样本与目标 bin 均解释为等间隔 bin 中心处的离散质量。两者步长
    相同；非整数 bin 的平移用相邻质量线性分配。完整支撑落在时间窗内时，
    该规则严格保持总质量；越出记录窗的尾部自然丢失，不重新归一化。
    """

    if not isinstance(measured_irf, MeasuredIRF):
        raise TypeError("measured_irf must be MeasuredIRF")
    tof = np.asarray(round_trip_time_hw, dtype=np.float64)
    centers = np.asarray(bin_centers_s, dtype=np.float64)
    if tof.ndim != 2 or centers.ndim != 1:
        raise ValueError("round_trip_time_hw and bin_centers_s must be 2-D and 1-D")
    if not np.all(np.isfinite(tof)) or np.any(tof < 0.0):
        raise ValueError("round-trip times must be finite and non-negative")
    if centers.size < 1 or not np.all(np.isfinite(centers)):
        raise ValueError("bin centers must be a finite non-empty vector")
    if centers.size > 1:
        target_step = float(centers[1] - centers[0])
        if not np.allclose(np.diff(centers), target_step, rtol=1e-10, atol=1e-18):
            raise ValueError("target bin centers must be uniformly spaced")
        if not np.isclose(
            target_step,
            measured_irf.sample_interval_s,
            rtol=1e-9,
            atol=1e-18,
        ):
            raise ValueError("measured IRF and target bin spacing must match")

    source_start = measured_irf.relative_time_s[0]
    step = measured_irf.sample_interval_s
    mass = measured_irf.probability_mass
    output = np.zeros(tof.shape + (centers.size,), dtype=np.float32)
    for index, center in enumerate(centers):
        coordinate = (center - tof - source_start) / step
        lower = np.floor(coordinate).astype(np.int32)
        fraction = coordinate - lower
        valid = (lower >= 0) & (lower < mass.size - 1)
        values = np.zeros(tof.shape, dtype=np.float64)
        if np.any(valid):
            lower_valid = lower[valid]
            fraction_valid = fraction[valid]
            values[valid] = (
                (1.0 - fraction_valid) * mass[lower_valid]
                + fraction_valid * mass[lower_valid + 1]
            )
        exact_last = (lower == mass.size - 1) & np.isclose(
            fraction, 0.0, rtol=0.0, atol=1e-8
        )
        values[exact_last] = mass[-1]
        output[..., index] = values.astype(np.float32)
    return output
