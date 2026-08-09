"""NYU RGB 针孔相机内参解析、单位射线与轴向深度几何换算。"""

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from .config import CameraGeometryConfig, SceneInputs, SensorConfig


_RGB_PARAMETER_NAMES = ("fx_rgb", "fy_rgb", "cx_rgb", "cy_rgb")
_SCALAR_ASSIGNMENT = re.compile(
    r"^\s*(fx_rgb|fy_rgb|cx_rgb|cy_rgb)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;\s*$"
)


@dataclass(frozen=True)
class RGBIntrinsics:
    """NYU RGB 针孔相机内参与其像素坐标约定。"""

    fx: float
    fy: float
    cx: float
    cy: float
    image_size_wh: tuple
    source_path: Path
    pixel_coordinate_convention: str = "matlab_one_based"

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("RGB intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("RGB focal lengths must be > 0")
        if (
            not isinstance(self.image_size_wh, (tuple, list))
            or len(self.image_size_wh) != 2
            or any(value <= 0 for value in self.image_size_wh)
        ):
            raise ValueError("image_size_wh must contain positive [W, H]")
        width, height = self.image_size_wh
        if not 0.0 < self.cx <= width or not 0.0 < self.cy <= height:
            raise ValueError("RGB principal point must lie inside the calibrated image")
        if self.pixel_coordinate_convention != "matlab_one_based":
            raise ValueError("only the Toolbox MATLAB one-based convention is supported")
        object.__setattr__(self, "image_size_wh", tuple(self.image_size_wh))
        object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def matrix_k(self):
        """返回 ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``。"""

        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SceneGeometry:
    """从 RGB 单位射线和轴向深度得到的逐像素斜距诊断。"""

    unit_rays_hwc: np.ndarray
    ray_direction_z_hw: np.ndarray
    slant_range_m_hw: np.ndarray


def load_nyu_rgb_intrinsics(camera_config, sensor_config=None):
    """只读解析 Toolbox ``camera_params.m`` 中四个 RGB 针孔参数。

    解析器只接受独立的标量 MATLAB 赋值，不执行 ``.m`` 文件。相机尺寸由
    配置显式声明，并与传感器的 640×480 空间网格交叉校验。
    """

    if not isinstance(camera_config, CameraGeometryConfig):
        raise TypeError("camera_config must be CameraGeometryConfig")
    if sensor_config is not None and not isinstance(sensor_config, SensorConfig):
        raise TypeError("sensor_config must be SensorConfig")
    path = camera_config.camera_params_path
    if not path.is_file():
        raise FileNotFoundError("NYU camera_params.m not found: {}".format(path))

    assignments = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = _SCALAR_ASSIGNMENT.match(line)
            if match:
                name = match.group(1)
                if name in assignments:
                    raise ValueError("duplicate RGB camera parameter: {}".format(name))
                assignments[name] = float(match.group(2))
    missing = tuple(name for name in _RGB_PARAMETER_NAMES if name not in assignments)
    if missing:
        raise ValueError("camera_params.m is missing RGB parameters: {}".format(missing))

    size_wh = camera_config.calibrated_image_size_wh
    if sensor_config is not None:
        sensor_size_wh = (sensor_config.image_width, sensor_config.image_height)
        if sensor_size_wh != size_wh:
            raise ValueError(
                "sensor size {} does not match calibrated RGB size {}".format(
                    sensor_size_wh, size_wh
                )
            )
    return RGBIntrinsics(
        fx=assignments["fx_rgb"],
        fy=assignments["fy_rgb"],
        cx=assignments["cx_rgb"],
        cy=assignments["cy_rgb"],
        image_size_wh=size_wh,
        source_path=path,
    )


def make_rgb_unit_rays(intrinsics):
    """按 Toolbox 的一基像素坐标生成 shape ``[H,W,3]`` 的单位射线。

    坐标系采用 ``+X`` 向图像右、``+Y`` 向图像上、``+Z`` 沿 RGB 光轴向前，
    与 Toolbox ``rgb_plane2rgb_world.m`` 的 ``[x,-y,z]`` 约定一致。
    """

    if not isinstance(intrinsics, RGBIntrinsics):
        raise TypeError("intrinsics must be RGBIntrinsics")
    width, height = intrinsics.image_size_wh
    columns = np.arange(1, width + 1, dtype=np.float64)
    rows = np.arange(1, height + 1, dtype=np.float64)
    x = (columns[np.newaxis, :] - intrinsics.cx) / intrinsics.fx
    y = -(rows[:, np.newaxis] - intrinsics.cy) / intrinsics.fy
    x = np.broadcast_to(x, (height, width))
    y = np.broadcast_to(y, (height, width))
    z = np.ones((height, width), dtype=np.float64)
    inverse_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
    rays = np.empty((height, width, 3), dtype=np.float32)
    rays[..., 0] = (x * inverse_norm).astype(np.float32)
    rays[..., 1] = (y * inverse_norm).astype(np.float32)
    rays[..., 2] = inverse_norm.astype(np.float32)
    return np.ascontiguousarray(rays)


def axial_depth_to_slant_range(depth_z_m, ray_direction_z_hw):
    """执行 ``range=z/d_z``，把 RGB 轴向深度换成逐像素真实斜距。"""

    depth = np.asarray(depth_z_m, dtype=np.float32)
    direction_z = np.asarray(ray_direction_z_hw, dtype=np.float32)
    if depth.ndim != 2 or direction_z.shape != depth.shape:
        raise ValueError("depth_z_m and ray_direction_z_hw must share shape [H,W]")
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("depth_z_m must be finite and > 0")
    if not np.all(np.isfinite(direction_z)) or np.any(direction_z <= 0.0):
        raise ValueError("ray_direction_z_hw must be finite and > 0")
    return np.ascontiguousarray(depth / direction_z, dtype=np.float32)


def build_scene_geometry(scene, intrinsics):
    """为一个轴向深度场景生成单位射线、``d_z`` 与斜距图。"""

    if not isinstance(scene, SceneInputs):
        raise TypeError("scene must be SceneInputs")
    rays = make_rgb_unit_rays(intrinsics)
    if rays.shape[:2] != scene.shape_hw:
        raise ValueError(
            "camera ray grid {} does not match scene {}".format(
                rays.shape[:2], scene.shape_hw
            )
        )
    direction_z = np.ascontiguousarray(rays[..., 2], dtype=np.float32)
    slant_range = axial_depth_to_slant_range(scene.depth_z_m, direction_z)
    return SceneGeometry(
        unit_rays_hwc=rays,
        ray_direction_z_hw=direction_z,
        slant_range_m_hw=slant_range,
    )
