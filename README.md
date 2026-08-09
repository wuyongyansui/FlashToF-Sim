# 完整阵列 Flash dToF 首光子仿真器

这是一个独立 Python 项目，用于实现 Flash 直接飞行时间仿真平台的第一条
可运行完整阵列链路。`SPCSimLib` 仅作为只读架构参考；本项目既不导入，
也不修改它。

版本 0.3 的固定逐场景链路为：

```text
SensorConfig + SceneInputs(depth_m[H,W], reflectivity[H,W])
  -> 空间理想瞬态 lambda[H,W,T]
  -> 每像素、每激光周期的最早光子
  -> 首光子等宽直方图 counts[H,W,T]
  -> 最大 bin 距离图 [H,W]
  -> 探测、未探测、峰位偏移与距离偏差图
```

## 目标传感器规格

用户可编辑的默认值对应首个完整阵列目标：

| 参数 | 默认值 |
|---|---:|
| SPAD 阵列 | 120 行 × 240 列 |
| EWH bin 数 | 190 |
| bin 宽 | 0.5 ns |
| 激光周期数 | 20,000 |
| 时间窗 | 95 ns |
| 单 bin 单程距离 | 0.074948 m |
| 最大无模糊距离 | 14.240142 m |

95 ns 时间窗可以覆盖本地 NYU Depth V2 轻量数据约 0.71–9.99 m 的
距离范围。

## 安装与运行

已验证 Conda 环境为 `spcsimlib`，Python 版本 3.8.20。运行时依赖仅有 NumPy
与 Pillow。

```powershell
conda activate spcsimlib
cd F:\Master\Job\lidar\flash_dtof_simulator

# 正式路径：流式处理配置 split 中的全部配对场景。
python run_simulation.py

# 开发子集：仍然逐场景流式处理。
python run_simulation.py --start 0 --limit 8

# 显式单场景调试；不能作为正式 benchmark 路径。
python debug_single_scene.py --split val --sample-id nyu_0000

python -m unittest discover -s tests -v
```

请只编辑 `example_config.py` 中明确标出的区域，以修改阵列尺寸、bin、
周期数、脉冲宽度、通量、随机种子、数据根目录、split、样本选择范围或
合成反射率模式。正式配置不包含 `sample_id`。

## 配置与场景边界

`SensorConfig` 保存用户设置的采集参数：

- 阵列高度与宽度；
- 等宽时间 bin 的数量与宽度；
- 激光周期数与随机种子；
- 高斯脉冲 FWHM；
- 参考距离处的期望探测信号光子数；
- 每 bin、每周期的期望探测背景光子数。

`SceneInputs` 保存空间 float32 数组：

| 物理量 | shape | 单位或含义 |
|---|---:|---|
| `depth_m` | `[H,W]` | 米制单程距离 |
| `reflectivity` | `[H,W]` | `[0,1]` 范围的合成无量纲系数 |
| 可选背景图 | `[H,W]` | 探测光子/bin/周期 |

派生的时间、距离、脉冲 sigma 与 shape 参数均为只读。场景数组会依据
传感器 shape 和最大无模糊距离进行校验。

### 中文配置速查

`example_config.py` 已在每个可调字段旁提供中文注释。关键含义如下：

| 配置字段 | 单位/取值 | 物理意义 |
|---|---|---|
| `image_height` / `image_width` | 像素 | SPAD 阵列行数 H 与列数 W |
| `num_time_bins` | 个 | EWH 等宽时间 bin 数 T |
| `bin_width_s` | 秒 | 单 bin 时间宽；单程距离间隔为 `c*bin_width_s/2` |
| `num_laser_periods` | 次 | 每像素累计的激光周期数；每周期最多记录最早一个光子 |
| `pulse_fwhm_s` | 秒 | 高斯系统脉冲的时间半高全宽，不是激光周期长度 |
| `reference_distance_m` | 米 | 参考信号通量对应的单程距离，其他距离按 `1/r²` 缩放 |
| `signal_photons_per_pulse_at_reference` | 探测光子/像素/周期 | 在参考距离且反射率为 1.0 时的期望信号光子数，已包含损耗与 PDE |
| `background_photons_per_bin` | 探测光子/像素/bin/周期 | 每个等宽 bin 的期望背景光子数 |
| `random_seed` | 非负整数 | 固定后可复现；批量运行据此为各场景派生稳定种子 |

参考反射率固定归一化为 `1.0`，不是独立配置字段。场景反射率会乘到
参考信号通量上：例如 `constant_reflectivity=0.5` 表示所有像素都使用
参考目标一半的相对反射率；在当前参考信号 `0.05` 下，参考距离处得到
`0.025` 探测光子/像素/周期。

NYU 批量字段中，`dataset_root` 是只读数据根目录，`split` 只能选择
`train` 或 `val`，`start` 是排序后配对 ID 的零基起点，`limit` 是最多
处理数量。`limit=None` 表示从 `start` 起处理该 split 的全部剩余样本。
`reflectivity_mode="constant"` 使用统一的合成相对反射率；可见光 RGB
不是经标定的 NIR 反射率，不应直接按物理 NIR 反射率解释。

几何适配对 RGB 与深度使用完全相同的中心裁剪。640×480 输入裁成
640×320 后缩放至 240×120；RGB 使用双线性，米制深度使用最近邻，
以免在深度不连续处混合前景和背景距离。

## NYU Depth V2 轻量数据加载

`NYUDepthV2Loader` 以只读方式打开本地配对数据：

```text
nyu-depth/
  images/{train,val}/nyu_XXXX.jpg
  depth/{train,val}/nyu_XXXX.npy
```

对于 640×480 源数据和 240×120 目标阵列，RGB 与深度使用完全相同的
中心裁剪：

```text
裁剪 [left, top, right, bottom] = [0, 80, 640, 400]
640×480 -> 居中裁成 640×320 -> 240×120
```

RGB 使用双线性 resize。米制深度使用最近邻 resize，避免在距离不连续
边界混合前景与背景。源 float16、Fortran-order NPY 数组会转换为
C-contiguous float32 米制数组。

默认反射率模式是 `constant`。可见光 RGB 会返回以供检查，但不会被
暗中当作已标定的 NIR 反射率。`luminance_proxy` 仅用于用户明确选择的
合成实验。

## 正式 split 批量与单场景调试

`NYUBatchConfig` 是正式数据源边界：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `dataset_root` | 本地 `nyu-depth` | 只读配对数据根目录 |
| `split` | `val` | 枚举完整训练集或验证集 split |
| `start` | `0` | 排序后配对 ID 的零基起始位置 |
| `limit` | `None` | 全部剩余配对样本；正式评估默认值 |

加载器先验证 JPG 与 NPY 的 ID 集合完全相同，再对唯一配对 ID 排序和
切片。`run_nyu_batch` 按顺序处理每个 ID：

```text
加载一对 RGB-D
  -> 适配为 [H,W]
  -> 运行完整 [H,W,T] 仿真器
  -> 累计轻量标量指标
  -> 释放当前场景数组
  -> 继续下一个 ID
```

程序不会创建数据集级 RGB、深度、瞬态或 EWH 张量。每个选中场景都会
获得由基础随机种子和其在排序 split 中的全局索引共同派生的确定性种子，
因此无论通过全量运行还是 `start/limit` 子集访问，同一场景的 Monte
Carlo 种子都相同。

最终按像素加权的汇总包括样本数、有效像素比例、实测与理论检测率、
平均距离偏差、MAE 和 RMSE。逐场景结果只保留 ID 与轻量标量指标。

`debug_single_scene.py --sample-id ...` 是刻意分离的调试命令，只用于
检查单个场景的裁剪、shape、直方图与重建行为，不能作为正式 split
benchmark。因此 `example_config.py` 和批量入口中均不存在 `sample_id`。

## 精确聚合首光子采集

对于每个 bin 的期望光子数为 `lambda[k]` 的独立 Poisson 到达过程，
某周期到达 bin `k` 且在此首次探测到光子的概率为：

```text
P(K=k) = exp(-sum(lambda[j], j<k)) * (1 - exp(-lambda[k]))
```

仿真器不会创建规模过大的 `[H,W,P]` 事件张量，而是维护到达每个 bin
时仍未探测的周期数：

```text
remaining = P
for k in range(T):
    q = 1 - exp(-lambda[..., k])
    counts[..., k] ~ Binomial(remaining, q)
    remaining -= counts[..., k]
no_detection = remaining
```

这种条件二项分解与逐周期独立仿真并在最早光子处停止具有完全相同的聚合
分布，能够自然产生高通量 pile-up 与峰位前移。对每个像素都强制满足
`sum(EWH) + no_detection == P`。

在 120×240×190 下，一个 float32 HWT 张量约占 20.87 MiB，一个 int32
EWH 也约占 20.87 MiB。若显式保存配置的 20,000 个周期，则会产生
5.76 亿个元素；即使使用 int16，也约占 1.07 GiB，且尚未包含其他数组。
因此正式路径的内存复杂度为 `O(HWT)`，不存在脉冲维度。

## 模块

- `flash_dtof/config.py`：`SensorConfig`、`SceneInputs` 与派生约束。
- `flash_dtof/scene.py`：NYU 配对加载、裁剪/缩放与反射率策略。
- `flash_dtof/batch.py`：split 选择、流式运行器与汇总指标。
- `flash_dtof/transient.py`：向量化逐像素高斯瞬态。
- `flash_dtof/first_photon.py`：解析概率与条件二项首光子聚合。
- `flash_dtof/ewh.py`：EWH 契约与逐像素周期守恒。
- `flash_dtof/reconstruction.py`：仅实现向量化最大 bin 测距。
- `flash_dtof/pipeline.py`：仿真编排与诊断图。
- `run_simulation.py`：不含 `sample_id` 的正式 split 批量入口。
- `debug_single_scene.py`：显式单场景调试命令。
- `tests/`：完整阵列、shape、守恒、种子、pile-up、距离梯度、单位、
  唯一配对枚举、limit 行为、多场景流式处理和 NYU 加载测试。

## 明确未实现

版本 0.3 不包含 Coates 或其他 pile-up 反演/校正、前沿法、质心法、
匹配滤波、亚 bin 重建、TDC offset/DNL/jitter、跨周期 dead time、
多径或多回波。这些能力保留为后续显式扩展，不会隐藏在当前基线中。
