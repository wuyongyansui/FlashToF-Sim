"""面向完整 SPAD 阵列的内存安全首光子聚合。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FirstPhotonAggregate:
    """不创建 ``[H, W, P]`` 张量的首光子聚合结果。"""

    counts_hwt: np.ndarray
    no_detection_counts_hw: np.ndarray
    num_laser_periods: int
    random_seed: int


def first_photon_probabilities(expected_photons_hwt):
    """返回首个探测 bin 与未探测事件的解析概率。

    对速率为 ``lambda[k]`` 的独立 Poisson bin：

    ``P(K=k) = exp(-sum(lambda[:k])) * (1 - exp(-lambda[k]))``.
    """

    expected = _validate_expected_photons(expected_photons_hwt)
    cumulative_inclusive = np.cumsum(expected, axis=-1, dtype=np.float64)
    cumulative_exclusive = cumulative_inclusive - expected
    conditional_detection = -np.expm1(-expected)
    probabilities = np.exp(-cumulative_exclusive) * conditional_detection
    no_detection = np.exp(-cumulative_inclusive[..., -1])
    return probabilities, no_detection


def sample_first_photon_counts(expected_photons_hwt, num_laser_periods, random_seed):
    """使用条件二项分布精确采样聚合首光子 EWH。

    ``remaining`` 是未在更早 bin 探测、因而到达当前 bin 的激光周期数。
    执行抽样

    ``counts[k] ~ Binomial(remaining, 1-exp(-lambda[k]))``

    并扣除这些探测，与逐周期独立仿真并在最早光子处停止的统计分布完全
    等价。内存复杂度为 ``O(H*W*T)``，不包含脉冲维度。
    """

    expected = _validate_expected_photons(expected_photons_hwt)
    if not isinstance(num_laser_periods, int) or isinstance(num_laser_periods, bool):
        raise ValueError("num_laser_periods must be an integer")
    if num_laser_periods <= 0 or num_laser_periods > np.iinfo(np.int32).max:
        raise ValueError("num_laser_periods must be in the positive int32 range")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")

    height, width, num_bins = expected.shape
    generator = np.random.default_rng(random_seed)
    remaining = np.full((height, width), num_laser_periods, dtype=np.int64)
    counts = np.zeros((height, width, num_bins), dtype=np.int32)

    for index in range(num_bins):
        if not np.any(remaining):
            break
        conditional_probability = np.clip(
            -np.expm1(-expected[..., index]), 0.0, 1.0
        )
        detected = generator.binomial(remaining, conditional_probability)
        counts[..., index] = detected.astype(np.int32)
        remaining -= detected

    return FirstPhotonAggregate(
        counts_hwt=counts,
        no_detection_counts_hw=remaining.astype(np.int32),
        num_laser_periods=num_laser_periods,
        random_seed=random_seed,
    )


def _validate_expected_photons(expected_photons_hwt):
    expected = np.asarray(expected_photons_hwt)
    if expected.ndim != 3 or expected.shape[-1] <= 0:
        raise ValueError("expected_photons_hwt must have shape [H, W, T]")
    if not np.issubdtype(expected.dtype, np.floating):
        expected = expected.astype(np.float32)
    if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise ValueError("expected photon rates must be finite and >= 0")
    return expected
