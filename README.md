# Flash dToF 完整阵列首光子仿真基线

这是一个独立的、可运行的 Python 项目。当前基线组合了两部分：

- NYU Depth V2 原生 640×480 RGB 像素几何；
- SP-TransientBench 一致的 672×0.75 ns 时间轴和实测全局 IRF。

它实现固定单回波链路：轴向深度 `z` → RGB 单位像素射线 → 真实斜距
`range=z/d_z` → 实测 IRF 理想到达率 → 每像素每激光周期仅保留最早光子
→ 等宽时间直方图 EWH → 最大-bin 斜距重建。高通量下早到光子优先，因此
pile-up 会由采集统计自然产生。

本项目不修改、复制或依赖 SPCSImLib 的上游代码，也不会修改 NYU 数据、
NYU Toolbox 或外部 IRF 文件。

## 当前默认参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 阵列 | 480×640 | 与原生 NYU RGB-D 像素一一对应 |
| EWH bin 数 | 672 | STB 时间轴 |
| bin 宽 | 0.75 ns | 与 IRF 的 750 ps 采样间隔一致 |
| 时间窗 | 504 ns | `672×0.75 ns` |
| 斜距 bin 间隔 | 0.112422 m | `c·bin_width/2` |
| 最大无歧义斜距 | 75.547699 m | `c·time_window/2` |
| 激光周期数 | 20,000 | 每像素、每周期最多记录一次最早探测 |
| 瞬态响应 | `measured_irf` | 默认读取外部 STB IRF |
| 场景回波 | 单回波 | 每像素一个轴向深度和一个反射率 |

默认设置集中在 [example_config.py](./example_config.py)，每个用户可调参数都
有中文注释。`SensorConfig` 是用户输入，`DerivedConfig` 中的时间窗、距离
间隔、最大无歧义距离和 HWT 内存是只读派生值。

## 相机几何与深度语义

默认只读解析：

```text
F:\Master\Job\lidar\toolbox_nyu_depth_v2\camera_params.m
```

从中取得经过本地验证的 RGB 针孔内参：

```text
fx = 518.85790117450188 px
fy = 519.46961112127485 px
cx = 325.58244941119034
cy = 253.73616633400465
K  = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

射线生成遵循 Toolbox `rgb_plane2rgb_world.m`：像素坐标是 MATLAB 一基
`u=1..640, v=1..480`，相机坐标采用 `+X` 向图像右、`+Y` 向图像上、
`+Z` 沿 RGB 光轴向前。对每个像素先构造并归一化：

```text
q = [(u-cx)/fx, -(v-cy)/fy, 1]
d = q / ||q||
```

轻量 NYU 的 NPY 已经与 RGB 像素对齐，因此不再使用 depth 相机内参或
RGB-depth 的 `R,t` 重投影。NPY 数值明确解释为 RGB 光轴方向的轴向深度
`depth_z_m`，真实单程斜距为：

```text
slant_range_m = depth_z_m / d_z
```

ToF `2·range/c`、参考信号的 `1/r²` 缩放、重建真值与误差评估全部使用该
斜距。调试输出会同时保存 `depth_z_m`、`d_z` 和 `slant_range_m`，不会再
把轴向深度误标为斜距。

当前未使用 RGB 畸变参数。这里假设轻量 NYU 的对齐 RGB-D 像素网格可直接
配合 Toolbox 的 RGB 针孔内参；这一点是当前唯一仍需用数据来源文档或实测
投影进一步确认的几何假设。

## 反射率模式

`NYUBatchConfig.reflectivity_mode` 支持两种模式：

- `constant`：所有像素使用 `constant_reflectivity`；
- `rgb_relative_proxy`：从与深度对齐的 NYU JPG 构造可见光相对反射率代理。

代理模式先把 8-bit sRGB 归一化并按标准分段函数线性化，再计算 Rec.709
线性亮度：

```text
Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear
ratio = clip(Y / median(Y_valid), ratio_min, ratio_max)
reflectivity = clip(scale * ratio, 0, 1)
```

其中 `scale` 通过单调二分求解，使全图反射率平均值等于
`constant_reflectivity`。默认比例保护范围为 `[0.05,20]`；全黑、没有有效
亮度或数值异常时回退为目标值的常数图。由于物理上限 `[0,1]` 会截断高光，
缩放不会严格保持每个像素原始亮度比，但目标均值仍在 float32 数值精度内
保持。

这个模式只是可见光明暗的相对代理，不是经过光谱响应、材料属性或主动激光
波长标定的真实 NIR 反射率。RGB 只进入主动回波信号项；均匀环境背景仍完全
由 `background_photons_per_bin` 控制，不会从 RGB 推断。调试输出中的
`input_reflectivity.npy` 保存实际送入瞬态生成器的反射率图。

## 实测 IRF

默认只读文件：

```text
F:\Master\Job\lidar\config_IRF_global.txt
```

虽然扩展名是 `.txt`，内容必须是逗号分隔的：

```text
t_ps,irf,irf_std
```

加载器严格检查列名、有限数值、单调等间隔时间、唯一 0 ps 样本、0 ps 处
峰值、非负 `irf_std` 以及采样间隔与传感器 bin 宽一致。当前文件有 501 个
样本，间隔 750 ps，峰在索引 250 的 0 ps。

原始 `irf` 中拟合产生的微小负残留在内存中 clip 到 0，再按离散样本和归一
化；原文件不修改。模板是以峰为 0 的相对响应，不是 672-bin 的完整观测窗。
生成场景瞬态时，将它平移到每个像素 ToF。源/目标步长相同时，非整数 bin
位移通过相邻质量线性分配：模板完整落在记录窗内时总质量守恒，落出窗口的
尾部自然截断且不重新归一化，峰前与峰后都保留。

`transient_model="gaussian"` 仍可用于小测试或显式后备；正式默认值不是
Gaussian。

## 首光子采集语义

理想到达率 `lambda[H,W,T]` 表示每周期、每 bin 的期望探测光子数。对每个
像素，从 `remaining=P` 开始依次处理时间 bin：

```text
p_k = 1 - exp(-lambda_k)
counts_k ~ Binomial(remaining, p_k)
remaining -= counts_k
```

这与逐周期 Poisson 到达并在最早探测光子处停止严格同分布，但不创建
`[H,W,P]` 张量。逐像素始终满足：

```text
sum_k(EWH_k) + no_detection = num_laser_periods
```

因此它不是“每个 bin 独立计数”。低通量下 EWH 接近理想到达率；高通量下
晚 bin 可用周期被前面的探测消耗，峰自然前移。

## NYU 流式批量入口

环境与依赖：

```powershell
conda activate spcsimlib
python -m pip install -e .
```

正式 split 入口：

```powershell
python run_simulation.py
python run_simulation.py --start 0 --limit 2
```

`limit=None` 表示从 `start` 起处理整个 split。流程严格逐场景执行
`load → geometry → simulate → summarize → release`，不把整个数据集放入
内存，也不保存每个场景的大数组。

每次批量运行创建独立目录：

```text
outputs/<run_name>_batch/
  config_snapshot.json
  summary_metrics.json
  scene_metrics.csv
  run_status.json
  RUN_README.txt
```

CSV 每场景一行；JSON 保存汇总指标。误差指标以真实斜距为真值。

## 独立单场景调试入口

单样本不混入正式批量配置：

```powershell
python debug_single_scene.py --split val --sample-id nyu_0000
```

调试目录至少包含：

```text
input_depth_z_m.npy
input_reflectivity.npy
true_slant_range_m.npy
ray_direction_z.npy
reconstructed_slant_range_m.npy
slant_range_bias_m.npy
depth_to_slant_delta_m.npy
valid_mask.npy
detected_counts.npy
no_detection_counts.npy
ideal_peak_bin.npy
observed_peak_bin.npy
peak_shift_bins.npy
measured_detection_fraction.npy
expected_detection_fraction.npy
diagnostics.json
config_snapshot.json
run_status.json
RUN_README.txt
```

当 `save_debug_ewh=True` 时还保存 `ewh_counts.npy`。所有 NPY 都禁用 pickle，
可用 `numpy.load(path, allow_pickle=False)` 读取。RGB 与理想瞬态不重复落盘，
因为输入数组、内参、IRF 路径和配置快照足以重建它们。

程序禁止覆盖既有运行目录，并拒绝把输出写入 NYU 数据集目录。

## 内存与运行成本

480×640×672 的一个 float32 瞬态或 int32 EWH 各占：

```text
825,753,600 bytes = 787.50 MiB = 825.75 MB
```

当前瞬态只保留一份完整 HWT，不再同时保留 signal 和 signal+background 两份
数组。进入首光子采样后，瞬态与 EWH 两个核心 HWT 至少约占 1,575 MiB，
还需加上随机采样临时数组、几何图和 Python/NumPy 开销。672 个 bin 也意味着
每场景需要 672 次逐 bin 条件二项采样。

所以：

- 正式批量只保留轻量指标，严格逐场景释放；
- 单场景保存完整 EWH 会额外占 787.50 MiB 磁盘；
- 默认原生场景适合在内存充足的机器上运行；
- 单元测试使用小阵列和较少周期，不改变面向用户的默认值。

## 测试

```powershell
conda run -n spcsimlib python -B -m unittest discover -s tests -v
```

测试覆盖：

- 672×0.75 ns 派生时间窗、距离 bin 与最大无歧义斜距；
- Toolbox RGB `K` 解析、尺寸校验、单位射线和中心射线；
- `depth_z_m → slant_range_m`；
- IRF 列结构、采样间隔、clip、归一化、峰平移、峰前/峰后和质量守恒；
- HWT shape、空间变化和斜距 `1/r²` 通量；
- sRGB 线性化、Rec.709 相对代理、目标均值/范围和全黑回退；
- RGB 反射率只调制主动回波、固定背景不随 RGB 变化；
- 固定 seed、逐像素周期守恒、低/高通量 pile-up；
- 最大-bin 斜距单位与空间梯度；
- NYU 配对枚举、原生尺寸与流式多场景；
- 批量/调试输出目录、配置快照和几何诊断数组。

## 与 SPCSImLib 的关键区别

本项目把“每周期最多一个最早探测”作为采集层的核心数据契约。EWH 的不同
bin 不是相互独立采样，因此可自然复现 pile-up。配置明确分开用户输入、相机
几何、场景轴向深度、派生时间/距离量和输出策略；每个模块有小规模可重复
测试。SPCSImLib 只作为历史参考，未被修改或复制。

## 第一版明确未实现

- Coates 或其他 pile-up 反演/校正；
- 白墙实测标定、逐像素 TDC offset、固定图样噪声 FPN；
- TDC jitter、DNL、INL、量化 offset；
- 跨周期 dead time、afterpulsing、crosstalk；
- 前沿、质心、匹配滤波等重建；
- 多径、多回波、复杂瞬态分解；
- RGB 畸变校正或 depth 相机到 RGB 的重新投影；
- RGB 亮度到真实 NIR 反射率的标定。

当前输出是“单回波、首光子 SPAD 阵列、EWH、最大-bin 斜距”的基础可运行
仿真基线，并不代表后续高级物理模块已经完成。
