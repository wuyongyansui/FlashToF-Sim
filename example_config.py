"""用户可编辑的传感器参数与正式 NYU 批量数据配置。"""

from pathlib import Path

from flash_dtof.batch import NYUBatchConfig
from flash_dtof.config import SensorConfig


# ---------------------------------------------------------------------------
# 用户可调：传感器与采集参数
# ---------------------------------------------------------------------------
SENSOR_CONFIG = SensorConfig(
    # SPAD 阵列行数 H（像素）；输出深度图、诊断图的高度。
    image_height=120,
    # SPAD 阵列列数 W（像素）；输出深度图、诊断图的宽度。
    image_width=240,
    # 等宽时间直方图的 bin 数 T（个）；总时间窗=T*bin_width_s。
    num_time_bins=190,
    # 单个时间 bin 的宽度（秒）；0.5e-9 s = 0.5 ns。
    # 单程距离分辨间隔约为 c*bin_width_s/2，此处约 0.074948 m。
    bin_width_s=0.5e-9,
    # 每个像素累计的激光发射周期数（次）；每周期最多保留一个最早光子。
    num_laser_periods=20_000,
    # 高斯系统脉冲的半高全宽 FWHM（秒）；描述时间响应展宽，不是周期长度。
    pulse_fwhm_s=1.0e-9,
    # 在 reference_distance_m、参考反射率=1.0 时，每像素每激光周期
    # 期望探测到的信号光子数（photons/pixel/period）；已包含光学损耗和 PDE。
    # 当前模型的“参考反射率”固定归一化为 1.0，不是另一个配置字段。
    signal_photons_per_pulse_at_reference=0.05,
    # 上述参考信号光子数对应的单程参考距离（米）；其他距离按 1/r^2 缩放。
    reference_distance_m=2.5,
    # 每像素、每时间 bin、每激光周期的期望探测背景光子数
    # （photons/pixel/bin/period）；当前基线在所有 bin 上均匀加入背景。
    background_photons_per_bin=1e-5,
    # 非负随机种子；固定后可复现实验。批量运行会据此为各场景派生稳定种子。
    random_seed=20260809,
)


# ---------------------------------------------------------------------------
# 用户可调：正式 NYU split 数据源与选择范围
# ---------------------------------------------------------------------------
# 正式路径不配置 sample_id；单张调试请使用 debug_single_scene.py。
# 所有场景都按 load -> simulate -> summarize -> release 逐张处理，不整批驻留内存。
NYU_BATCH_CONFIG = NYUBatchConfig(
    # NYU 轻量 RGB-D 数据根目录；程序只读 images/ 与 depth/，不移动或改写数据。
    dataset_root=Path(__file__).resolve().parent.parent / "nyu-depth",
    # 数据划分名称：只能是 "train" 或 "val"；会枚举该 split 中全部配对 ID。
    split="val",
    # 在按 sample ID 排序后的配对列表中，从第几个样本开始（零基索引）。
    start=0,
    # 最多处理的场景数；None 表示从 start 起处理该 split 的所有剩余场景。
    # 开发时可临时设为较小正整数，正式全量评估应保持 None。
    limit=None,
    # 反射率来源："constant" 为全像素常数；"luminance_proxy" 仅供明确的
    # 合成实验。NYU JPG 是可见光 RGB，不应视为经过标定的 NIR 反射率。
    reflectivity_mode="constant",
    # 仅在 constant 模式下生效的无量纲相对反射率，范围 [0,1]。
    # 0.5 表示每个像素采用参考反射率 1.0 的一半；因此在参考距离处，
    # 当前默认期望信号为 0.05*0.5=0.025 photons/pixel/period。
    constant_reflectivity=0.5,
)

# NYU 几何适配语义（由 loader 固定执行，不是额外配置项）：
# RGB 与米制深度先使用完全相同的中心裁剪以匹配 240:120=2:1 的目标宽高比；
# 对 640x480 源图即裁剪 [left,top,right,bottom]=[0,80,640,400] 得 640x320。
# 随后 RGB 用双线性缩放至 240x120；深度用最近邻缩放，避免跨物体边界混合距离。
