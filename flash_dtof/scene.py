"""只读空间场景加载与几何适配。"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .config import SceneInputs


_SAMPLE_ID_PATTERN = re.compile(r"^nyu_\d{4}$")


@dataclass(frozen=True)
class LoadedNYUScene:
    """一个已对齐并适配到指定 SPAD 阵列的 RGB-D 样本。"""

    sample_id: str
    split: str
    rgb_u8_hwc: np.ndarray
    scene_inputs: SceneInputs
    source_size_wh: tuple
    crop_box_ltrb: tuple


class NYUDepthV2Loader:
    """读取成对的 JPEG/float16-NPY NYU Depth V2 轻量样本。

    RGB 与深度使用相同的中心裁剪以匹配目标宽高比。随后 RGB 使用双线性
    resize，米制深度使用最近邻 resize，避免在前景/背景边界混合出虚假
    距离。源文件始终只读打开，绝不修改。
    """

    def __init__(
        self,
        dataset_root,
        output_height=120,
        output_width=240,
        reflectivity_mode="constant",
        constant_reflectivity=1.0,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.reflectivity_mode = reflectivity_mode
        self.constant_reflectivity = float(constant_reflectivity)
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dimensions must be positive")
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
        crop_box = self._center_crop_box(source_width, source_height)
        left, top, right, bottom = crop_box
        rgb_crop = rgb[top:bottom, left:right]
        depth_crop = np.asarray(depth[top:bottom, left:right], dtype=np.float32)

        resampling = getattr(Image, "Resampling", Image)
        rgb_resized = Image.fromarray(rgb_crop, mode="RGB").resize(
            (self.output_width, self.output_height), resample=resampling.BILINEAR
        )
        depth_resized = Image.fromarray(depth_crop, mode="F").resize(
            (self.output_width, self.output_height), resample=resampling.NEAREST
        )
        rgb_out = np.ascontiguousarray(np.asarray(rgb_resized, dtype=np.uint8))
        depth_out = np.ascontiguousarray(np.asarray(depth_resized, dtype=np.float32))
        reflectivity = self._make_reflectivity(rgb_out)

        return LoadedNYUScene(
            sample_id=sample_id,
            split=split,
            rgb_u8_hwc=rgb_out,
            scene_inputs=SceneInputs(depth_m=depth_out, reflectivity=reflectivity),
            source_size_wh=(source_width, source_height),
            crop_box_ltrb=crop_box,
        )

    def _center_crop_box(self, source_width, source_height):
        source_aspect = float(source_width) / source_height
        target_aspect = float(self.output_width) / self.output_height
        if source_aspect < target_aspect:
            crop_height = max(1, int(round(source_width / target_aspect)))
            top = (source_height - crop_height) // 2
            return (0, top, source_width, top + crop_height)
        crop_width = max(1, int(round(source_height * target_aspect)))
        left = (source_width - crop_width) // 2
        return (left, 0, left + crop_width, source_height)

    def _make_reflectivity(self, rgb_u8_hwc):
        if self.reflectivity_mode == "constant":
            return np.full(
                (self.output_height, self.output_width),
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
