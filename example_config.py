"""用户可编辑的 NYU 几何、STB 时间传感器、数据选择与输出配置。"""

from pathlib import Path

from flash_dtof.batch import NYUBatchConfig
from flash_dtof.config import CameraGeometryConfig, SensorConfig
from flash_dtof.output import OutputConfig


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 用户可调：NYU RGB 相机几何
# ---------------------------------------------------------------------------
CAMERA_CONFIG = CameraGeometryConfig(
    # NYU Depth V2 Toolbox 的标定脚本路径；程序只读解析其中 fx_rgb、fy_rgb、
    # cx_rgb、cy_rgb，不执行 MATLAB，不读取 depth K，也不应用 RGB-depth 外参。
    camera_params_path=WORKSPACE_ROOT / "toolbox_nyu_depth_v2" / "camera_params.m",
    # 标定对应的 RGB 图像尺寸 [W,H]（像素）；必须与传感器和轻量 NYU 数据
    # 的原生 640×480 网格一致，不裁剪、不缩放、不重投影。
    calibrated_image_size_wh=(640, 480),
    # 输入深度固定解释为 RGB 光轴方向轴向深度 z（米），不是斜距。
    # 程序按单位 RGB 射线计算真实斜距 range=z/d_z；当前只支持此语义。
    depth_semantics="rgb_optical_axis_z",
)


# ---------------------------------------------------------------------------
# 用户可调：STB 一致的时间传感器、通量与响应
# ---------------------------------------------------------------------------
SENSOR_CONFIG = SensorConfig(
    # SPAD 阵列行数 H（像素）；480 与 NYU 原生 RGB/深度高度一致。
    image_height=480,
    # SPAD 阵列列数 W（像素）；640 与 NYU 原生 RGB/深度宽度一致。
    image_width=640,
    # EWH 等宽时间 bin 数 T（个）；STB 时间轴配置为 672 个 bin。
    num_time_bins=672,
    # 单 bin 宽度（秒）；0.75e-9 s=0.75 ns，与实测 IRF 的 750 ps 采样一致。
    # 单程斜距 bin 间隔约 c*0.75 ns/2=0.112422 m。
    bin_width_s=0.75e-9,
    # 每像素累计的激光发射周期数（次）；每周期只保留最早探测光子。
    num_laser_periods=20_000,
    # 默认时间响应模式。"measured_irf" 使用下方实测模板；"gaussian" 只用于
    # 显式后备或受控测试，不是正式默认模型。
    transient_model="measured_irf",
    # SP-TransientBench IRF 文件路径。文件虽为 .txt 后缀，内容必须是逗号
    # 分隔的 t_ps,irf,irf_std；程序只读、将微小负残留 clip 到 0 后归一化。
    measured_irf_path=WORKSPACE_ROOT / "config_IRF_global.txt",
    # 仅在 transient_model="gaussian" 时生效的高斯 FWHM（秒）。
    pulse_fwhm_s=1.0e-9,
    # 在 reference_distance_m、反射率 1.0 时，每像素每周期的期望信号
    # 探测光子数；已包含光学损耗和 PDE。
    signal_photons_per_pulse_at_reference=0.05,
    # 参考信号对应的真实斜距（米）；其他像素按斜距的 1/r² 缩放。
    reference_distance_m=2.5,
    # 每像素、每时间 bin、每周期的均匀背景期望探测光子数。
    background_photons_per_bin=1e-5,
    # 非负随机种子；固定后可复现。批量入口为各场景派生稳定种子。
    random_seed=20260809,
)


# ---------------------------------------------------------------------------
# 用户可调：正式 NYU split 数据源与选择范围
# ---------------------------------------------------------------------------
# 正式配置不含 sample_id；单张调试使用 debug_single_scene.py。
# 所有场景按 load -> geometry -> simulate -> summarize -> release 逐张处理。
NYU_BATCH_CONFIG = NYUBatchConfig(
    # NYU 轻量 RGB-D 根目录；程序只读 images/ 与 depth/，不改写数据。
    dataset_root=WORKSPACE_ROOT / "nyu-depth",
    # 数据划分：只能是 "train" 或 "val"。
    split="val",
    # 按 sample ID 排序后从第几个配对样本开始，零基索引。
    start=0,
    # 最多处理多少场景；None 表示从 start 起处理该 split 的全部剩余场景。
    # 开发时可临时设为较小正整数，正式全量评估保持 None。
    limit=None,
    # 反射率模式："constant" 为全图常数；"rgb_relative_proxy" 对 NYU
    # sRGB 线性化后用 Rec.709 亮度构造相对明暗代理。RGB 只调制主动回波
    # 信号，背景光仍由 SensorConfig 的固定背景参数决定。
    reflectivity_mode="constant",
    # 无量纲目标反射率，范围 [0,1]。constant 模式下是每个像素的固定值；
    # rgb_relative_proxy 模式下是整幅有效场景的目标平均值。0.5 表示目标
    # 平均回波系数为参考反射率 1.0 的一半。
    constant_reflectivity=0.5,
    # RGB 相对亮度比例下限（无量纲），避免黑暗像素被压到严格零回波。
    relative_proxy_ratio_min=0.05,
    # RGB 相对亮度比例上限（无量纲），抑制局部高光产生极端信号倍率。
    relative_proxy_ratio_max=20.0,
    # 线性 Rec.709 亮度的有效判定阈值（无量纲）；整图没有超过该阈值的
    # 亮度时安全回退为 constant_reflectivity 常数图。
    relative_proxy_luminance_epsilon=1e-6,
)


# ---------------------------------------------------------------------------
# 用户可调：仿真结果输出位置与实验名称
# ---------------------------------------------------------------------------
OUTPUT_CONFIG = OutputConfig(
    # 所有运行目录的父目录；程序拒绝把结果写进 NYU 数据集目录。
    output_root=Path(__file__).resolve().parent / "outputs",
    # 实验名称。批量与单场景目录会在其后追加模式和样本信息。
    run_name="nyu_geometry_stb_timing_baseline",
    # 同名策略："increment" 追加 __002 等后缀；"error" 直接报错。
    # 两种策略都不会覆盖已有结果。
    existing_run_policy="increment",
    # 单场景调试是否保存完整 int32 EWH。480×640×672 单文件约 787.50 MiB
    # （825.75 MB）；True 保存，False 跳过。正式批量始终不保存逐场景大数组。
    save_debug_ewh=True,
)


# 固定几何约定：轻量 NPY 已与 RGB 像素对齐，所以不使用 depth K 或 R,t
# 重新投影。Toolbox 的 MATLAB 一基像素坐标用于生成单位 RGB 射线；输入
# depth_z_m 是 RGB 光轴 z，ToF 和 inverse-square 衰减均使用 z/d_z 斜距。
