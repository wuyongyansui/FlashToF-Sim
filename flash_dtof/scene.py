"""只读原生空间场景加载与像素恒等数据契约。"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .config import SceneInputs


_SAMPLE_ID_PATTERN = re.compile(r"^nyu_\d{4}$")


@dataclass(frozen=True)
class LoadedNYUScene:
    """一个保持原生像素网格、不做裁剪或缩放的 RGB-D 样本。"""

    sample_id: str
    split: str
    rgb_u8_hwc: np.ndarray
    scene_inputs: SceneInputs
    source_size_wh: tuple
    geometry_transform: str


class NYUDepthV2Loader:
    """读取成对的 JPEG/float16-NPY NYU Depth V2 轻量样本。

    RGB 与米制轴向深度必须具有同一原生 shape，且必须与 ``expected_size_wh``
    完全一致。加载器只转换数组 dtype/内存布局，不裁剪、不缩放、不重投影，
    因而每个原始 NYU 像素对应一个 SPAD 仿真像素。源文件始终只读打开。
    """

    def __init__(
        self,
        dataset_root,
        expected_size_wh=(640, 480),
        reflectivity_mode="constant",
        constant_reflectivity=1.0,
    ):
        self.dataset_root = Path(dataset_root)
        if (
            not isinstance(expected_size_wh, (tuple, list))
            or len(expected_size_wh) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in expected_size_wh
            )
        ):
            raise ValueError("expected_size_wh must contain positive integer [W, H]")
        self.expected_size_wh = tuple(expected_size_wh)
        self.reflectivity_mode = reflectivity_mode
        self.constant_reflectivity = float(constant_reflectivity)
        if reflectivity_mode not in ("constant", "luminance_proxy"):
            raise ValueError("reflectivity_mode must be 'constant' or 'luminance_proxy'")
        if not 0.0 <= self.constant_reflectivity <= 1.0:
            raise ValueError("constant_reflectivity must be in [0, 1]")
        for modality in ("images", "depth"):
            for split in ("train", "val"):
                path = self.dataset_root / modality / split
                if not path.is_dir():
                    raise FileNotFoundError("missing NYU directory: {}".format(path))

    def list_sample_ids(self, split):
        """返回排序后的配对 ID；任一模态不完整时立即报错。"""

        self._validate_split(split)
        image_ids = {path.stem for path in (self.dataset_root / "images" / split).glob("*.jpg")}
        depth_ids = {path.stem for path in (self.dataset_root / "depth" / split).glob("*.npy")}
        if image_ids != depth_ids:
            raise ValueError(
                "unpaired NYU files in {}: missing depth={}, missing RGB={}".format(
                    split, sorted(image_ids - depth_ids), sorted(depth_ids - image_ids)
                )
            )
        sample_ids = tuple(sorted(image_ids))
        if len(sample_ids) != len(set(sample_ids)):
            raise AssertionError("paired NYU sample IDs must be unique")
        return sample_ids

    def select_sample_ids(self, split, start=0, limit=None):
        """从 ``split`` 的所有配对 ID 中选取确定性切片。

        ``limit=None`` 是正式评估默认值，表示选取从 ``start`` 起的全部
        样本。此方法只保存文件 ID，不加载 RGB-D 数组。
        """

        self._validate_split(split)
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("start must be a non-negative integer")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be None or a positive integer")
        sample_ids = self.list_sample_ids(split)
        stop = None if limit is None else start + limit
        return sample_ids[start:stop]

    def load(self, sample_id, split="val"):
        """为流式处理或显式单场景调试加载一对 RGB-D。"""

        self._validate_split(split)
        if not _SAMPLE_ID_PATTERN.match(sample_id):
            raise ValueError("sample_id must look like nyu_0000")
        image_path = self.dataset_root / "images" / split / (sample_id + ".jpg")
        depth_path = self.dataset_root / "depth" / split / (sample_id + ".npy")
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError("paired NYU sample not found: {}/{}".format(split, sample_id))

        with Image.open(str(image_path)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        depth = np.load(str(depth_path), allow_pickle=False)
        if depth.ndim != 2:
            raise ValueError("NYU depth array must have shape [H, W]")
        if rgb.shape[:2] != depth.shape:
            raise ValueError("NYU RGB and depth source shapes are not aligned")
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
            raise ValueError("NYU depth must be finite, positive metres")

        source_height, source_width = depth.shape
        source_size_wh = (source_width, source_height)
        if source_size_wh != self.expected_size_wh:
            raise ValueError(
                "native NYU size {} does not match sensor size {}; crop/resize is disabled".format(
                    source_size_wh, self.expected_size_wh
                )
            )

        rgb_out = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_out = np.ascontiguousarray(depth, dtype=np.float32)
        reflectivity = self._make_reflectivity(rgb_out)

        return LoadedNYUScene(
            sample_id=sample_id,
            split=split,
            rgb_u8_hwc=rgb_out,
            scene_inputs=SceneInputs(depth_z_m=depth_out, reflectivity=reflectivity),
            source_size_wh=source_size_wh,
            geometry_transform="native_identity",
        )

    def _make_reflectivity(self, rgb_u8_hwc):
        if self.reflectivity_mode == "constant":
            return np.full(
                rgb_u8_hwc.shape[:2],
                self.constant_reflectivity,
                dtype=np.float32,
            )
        rgb = rgb_u8_hwc.astype(np.float32) / 255.0
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return np.ascontiguousarray(np.clip(luminance, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def _validate_split(split):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
