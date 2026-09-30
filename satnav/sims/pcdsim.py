#!/usr/bin/env python3
"""Point-cloud RGB simulator with SatSim-compatible TIF state and metrics.

``PCDSimWrapper`` keeps the built-in :class:`SatSimWrapper` as its movement,
geographic-coordinate, navigability, and top-down-map implementation.  Only
the RGB observation is replaced with a nadir perspective rendering of the
scene point cloud.

Point-cloud assets are resolved as::

    DATASET.PCD_PATH/<logical scene id>/*.ply

The per-block spatial index is cached beside the point-cloud files as
``pcdsim_index.json``.  Local XYZRGB points and the last rendered frame are
also cached in memory across observations.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig

from satnav.sims.satsim.geoutils import GeoUtils
from satnav.sims.satsim_wrapper import SatSimWrapper


_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}
_COLOR_ALIASES = {
    "red": ("red", "r", "diffuse_red"),
    "green": ("green", "g", "diffuse_green"),
    "blue": ("blue", "b", "diffuse_blue"),
}
_LOCAL_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ],
    align=False,
)
_INTERNAL_RENDER_WIDTH = 1024
_INTERNAL_RENDER_HEIGHT = 1024


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, DictConfig):
        return getattr(config, name, default)
    if isinstance(config, Mapping):
        return config.get(name, default)
    return default


@dataclass(frozen=True)
class _Registration:
    """Fixed EPSG:3857-to-local registration for one point-cloud family."""

    source_width: int
    source_height: int
    local_pixel_size_metres: float
    geotransform: Tuple[float, float, float, float, float, float]

    @property
    def local_origin_epsg3857(self) -> np.ndarray:
        gt = self.geotransform
        return np.asarray(
            [
                gt[0] + gt[2] * self.source_height,
                gt[3] + gt[5] * self.source_height,
            ],
            dtype=np.float64,
        )

    @property
    def local_to_epsg3857_matrix(self) -> np.ndarray:
        gt = self.geotransform
        pixel = self.local_pixel_size_metres
        return np.asarray(
            [
                [gt[1] / pixel, -gt[2] / pixel],
                [gt[4] / pixel, -gt[5] / pixel],
            ],
            dtype=np.float64,
        )

    def epsg3857_to_local(self, easting: float, northing: float) -> np.ndarray:
        return np.linalg.solve(
            self.local_to_epsg3857_matrix,
            np.asarray([easting, northing], dtype=np.float64)
            - self.local_origin_epsg3857,
        )


# These registrations are intrinsic calibration data for the available point
# clouds, not machine-local asset paths.  Scene suffixes such as Cambridge-1
# resolve to the same city registration.
_REGISTRATIONS = {
    "cambridge": _Registration(
        source_width=8613,
        source_height=8549,
        local_pixel_size_metres=0.25,
        geotransform=(
            12150.654131870977,
            0.40775917001922746,
            -0.012173240101837643,
            6839219.157459296,
            -0.012173240101837476,
            -0.40775917001922196,
        ),
    ),
    "birmingham": _Registration(
        source_width=5096,
        source_height=5228,
        local_pixel_size_metres=0.25,
        geotransform=(
            -212523.5778807616,
            0.4105410555483493,
            0.0,
            6895216.529521647,
            0.0,
            -0.41054105554831305,
        ),
    ),
}


@dataclass(frozen=True)
class _PlyLayout:
    path: Path
    data_offset: int
    header_point_count: int
    point_count: int
    dtype: np.dtype
    fields: Mapping[str, str]


@dataclass(frozen=True)
class _PointBlock:
    path: Path
    size: int
    mtime_ns: int
    point_count: int
    header_point_count: int
    bounds: Tuple[float, float, float, float, float, float]

    def intersects_xy(
        self, left: float, bottom: float, right: float, top: float
    ) -> bool:
        min_x, min_y, _, max_x, max_y, _ = self.bounds
        return not (
            max_x < left or min_x > right or max_y < bottom or min_y > top
        )


@dataclass
class _LocalPointCache:
    points: np.ndarray
    bounds: Tuple[float, float, float, float]

    def contains(self, required: Sequence[float]) -> bool:
        left, bottom, right, top = self.bounds
        req_left, req_bottom, req_right, req_top = required
        return (
            left <= req_left
            and bottom <= req_bottom
            and right >= req_right
            and top >= req_top
        )


def _parse_ply(path: Path) -> _PlyLayout:
    properties: List[Tuple[str, str]] = []
    header_count: Optional[int] = None
    current_element = ""
    file_format = ""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"{path.name}: missing PLY signature")
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"{path.name}: missing end_header")
            line = raw_line.decode("ascii").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                file_format = parts[1]
            elif parts[0] == "element":
                current_element = parts[1].lower()
                if current_element == "vertex":
                    header_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError(
                        f"{path.name}: list vertex properties are unsupported"
                    )
                scalar_type = parts[1].lower()
                if scalar_type not in _PLY_TYPES:
                    raise ValueError(
                        f"{path.name}: unsupported PLY type {scalar_type!r}"
                    )
                properties.append((parts[2].lower(), _PLY_TYPES[scalar_type]))
            elif parts[0] == "end_header":
                data_offset = stream.tell()
                break

    if file_format != "binary_little_endian":
        raise ValueError(
            f"{path.name}: only binary_little_endian PLY is supported"
        )
    if not header_count or not properties:
        raise ValueError(f"{path.name}: missing vertex definition")
    dtype = np.dtype(properties, align=False)
    available_count = max(0, (path.stat().st_size - data_offset) // dtype.itemsize)
    point_count = min(header_count, available_count)
    if point_count == 0:
        raise ValueError(f"{path.name}: contains no complete point records")

    names = set(dtype.names or ())
    fields: Dict[str, str] = {}
    for coordinate in ("x", "y", "z"):
        if coordinate not in names:
            raise ValueError(f"{path.name}: missing {coordinate}")
        fields[coordinate] = coordinate
    for canonical, aliases in _COLOR_ALIASES.items():
        match = next((alias for alias in aliases if alias in names), None)
        if match is None:
            raise ValueError(f"{path.name}: missing {canonical} color")
        fields[canonical] = match
    return _PlyLayout(
        path=path,
        data_offset=data_offset,
        header_point_count=header_count,
        point_count=point_count,
        dtype=dtype,
        fields=fields,
    )


def _iter_ply(layout: _PlyLayout, chunk_points: int) -> Iterable[np.ndarray]:
    with layout.path.open("rb") as stream:
        stream.seek(layout.data_offset)
        remaining = layout.point_count
        while remaining:
            count = min(chunk_points, remaining)
            points = np.fromfile(stream, dtype=layout.dtype, count=count)
            if len(points) != count:
                raise OSError(
                    f"{layout.path.name}: expected {count} points, read {len(points)}"
                )
            yield points
            remaining -= count


def _scan_bounds(
    layout: _PlyLayout, chunk_points: int
) -> Tuple[float, float, float, float, float, float]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    for points in _iter_ply(layout, chunk_points):
        xyz = np.column_stack(
            (points[layout.fields["x"]], points[layout.fields["y"]], points[layout.fields["z"]])
        )
        finite = np.all(np.isfinite(xyz), axis=1)
        if finite.any():
            minimum = np.minimum(minimum, xyz[finite].min(axis=0))
            maximum = np.maximum(maximum, xyz[finite].max(axis=0))
    if not np.all(np.isfinite(minimum)):
        raise ValueError(f"{layout.path.name}: no finite XYZ points")
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


def _block_payload(block: _PointBlock) -> Dict[str, Any]:
    return {
        "file": block.path.name,
        "size": block.size,
        "mtime_ns": block.mtime_ns,
        "point_count": block.point_count,
        "header_point_count": block.header_point_count,
        "bounds": list(block.bounds),
    }


def _build_or_load_index(
    pointcloud_dir: Path, chunk_points: int
) -> List[_PointBlock]:
    ply_paths = sorted(pointcloud_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(
            f"No binary PLY point-cloud files found in {pointcloud_dir}"
        )
    index_path = pointcloud_dir / "pcdsim_index.json"
    signatures = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in ply_paths
    }
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            cached = {item["file"]: item for item in payload.get("blocks", [])}
            if set(cached) == set(signatures) and all(
                (cached[name]["size"], cached[name]["mtime_ns"])
                == signatures[name]
                for name in signatures
            ):
                return [
                    _PointBlock(
                        path=pointcloud_dir / name,
                        size=int(cached[name]["size"]),
                        mtime_ns=int(cached[name]["mtime_ns"]),
                        point_count=int(cached[name]["point_count"]),
                        header_point_count=int(cached[name]["header_point_count"]),
                        bounds=tuple(float(value) for value in cached[name]["bounds"]),
                    )
                    for name in sorted(cached)
                ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    blocks: List[_PointBlock] = []
    started = time.perf_counter()
    for path in ply_paths:
        layout = _parse_ply(path)
        blocks.append(
            _PointBlock(
                path=path,
                size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                point_count=layout.point_count,
                header_point_count=layout.header_point_count,
                bounds=_scan_bounds(layout, chunk_points),
            )
        )
    payload = {
        "version": 1,
        "pointcloud_dir": str(pointcloud_dir),
        "created_seconds": time.perf_counter() - started,
        "blocks": [_block_payload(block) for block in blocks],
    }
    temporary = index_path.with_name(
        f".{index_path.name}.{os.getpid()}.partial"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, index_path)
    return blocks


def _colors_as_uint8(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        finite = values[np.isfinite(values)]
        multiplier = 255.0 if finite.size and float(finite.max()) <= 1.0 else 1.0
        return np.clip(np.nan_to_num(values) * multiplier, 0, 255).astype(np.uint8)
    if values.dtype.itemsize > 1 and values.size and int(values.max()) > 255:
        maximum = float(np.iinfo(values.dtype).max)
        return np.clip(
            values.astype(np.float64) * (255.0 / maximum), 0, 255
        ).astype(np.uint8)
    return np.clip(values, 0, 255).astype(np.uint8, copy=False)


def _update_zbuffer(
    indices: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    zbuffer: np.ndarray,
    image: np.ndarray,
) -> int:
    if not len(indices):
        return 0
    order = np.lexsort((depths, indices))
    sorted_indices = indices[order]
    first = np.r_[True, sorted_indices[1:] != sorted_indices[:-1]]
    winners = order[first]
    winner_indices = indices[winners]
    nearer = depths[winners] < zbuffer[winner_indices]
    winner_indices = winner_indices[nearer]
    winners = winners[nearer]
    zbuffer[winner_indices] = depths[winners]
    image[winner_indices] = colors[winners]
    return int(len(winners))


def _fill_one_pixel_holes(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Fill invalid pixels from the mean of valid 8-connected neighbours.

    The validity mask comes from the Z-buffer rather than RGB values, so a
    genuinely black point remains a valid sample.  This is intentionally a
    single pass: it closes small rasterisation gaps without propagating colour
    deeply into regions where the point cloud has no coverage.  At least three
    valid neighbours are required before an invalid pixel is filled.
    """
    height, width = valid_mask.shape
    colour_sum = np.zeros((height, width, 3), dtype=np.uint16)
    neighbour_count = np.zeros((height, width), dtype=np.uint8)

    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)))
    padded_valid = np.pad(valid_mask, ((1, 1), (1, 1)))
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbour_valid = padded_valid[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            neighbour_rgb = padded_image[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            colour_sum += neighbour_rgb * neighbour_valid[..., None]
            neighbour_count += neighbour_valid

    fill_mask = ~valid_mask & (neighbour_count > 2)
    filled_count = int(fill_mask.sum())
    if not filled_count:
        return image, 0

    result = image.copy()
    result[fill_mask] = np.rint(
        colour_sum[fill_mask] / neighbour_count[fill_mask, None]
    ).astype(np.uint8)
    return result, filled_count


class PCDSimWrapper(SatSimWrapper):
    """SatSim-compatible simulator whose RGB sensor renders point clouds."""

    def __init__(
        self,
        config: Union[DictConfig, dict],
        scenes_dir: Optional[str] = None,
    ) -> None:
        super().__init__(config, scenes_dir)
        dataset_config = _config_value(config, "DATASET", {})
        pcd_path = _config_value(dataset_config, "PCD_PATH")
        if not pcd_path:
            raise ValueError(
                "DATASET.PCD_PATH must be configured when SIMULATOR.TYPE=pcdsim"
            )
        self._pcd_root = Path(str(pcd_path)).expanduser()
        if not self._pcd_root.is_dir():
            raise FileNotFoundError(
                f"DATASET.PCD_PATH directory does not exist: {self._pcd_root}"
            )

        sim_config = _config_value(config, "SIMULATOR", config)
        pcd_config = _config_value(sim_config, "PCDSIM", {})
        self._chunk_points = int(
            _config_value(pcd_config, "CHUNK_POINTS", 2_000_000)
        )
        self._point_size = int(_config_value(pcd_config, "POINT_SIZE", 1))
        self._near_clip_metres = float(
            _config_value(pcd_config, "NEAR_CLIP_METRES", 0.05)
        )
        self._cache_padding_metres = float(
            _config_value(pcd_config, "CACHE_PADDING_METRES", 25.0)
        )
        self._ground_percentile = float(
            _config_value(pcd_config, "GROUND_PERCENTILE", 2.0)
        )
        if self._chunk_points <= 0:
            raise ValueError("SIMULATOR.PCDSIM.CHUNK_POINTS must be positive")
        if self._point_size <= 0 or self._point_size % 2 == 0:
            raise ValueError("SIMULATOR.PCDSIM.POINT_SIZE must be a positive odd integer")
        if self._near_clip_metres <= 0:
            raise ValueError("SIMULATOR.PCDSIM.NEAR_CLIP_METRES must be positive")
        if self._cache_padding_metres < 0:
            raise ValueError("SIMULATOR.PCDSIM.CACHE_PADDING_METRES cannot be negative")
        if not 0.0 <= self._ground_percentile <= 100.0:
            raise ValueError("SIMULATOR.PCDSIM.GROUND_PERCENTILE must be in [0, 100]")

        self._pcd_scene_dir: Optional[Path] = None
        self._registration: Optional[_Registration] = None
        self._blocks: List[_PointBlock] = []
        self._local_cache: Optional[_LocalPointCache] = None
        self._last_render_key: Optional[Tuple[Any, ...]] = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_render_stats: Dict[str, Any] = {}

    @staticmethod
    def _registration_key(scene_id: str) -> str:
        lowered = str(scene_id).strip().lower()
        if lowered in _REGISTRATIONS:
            return lowered
        base = lowered.rsplit("-", 1)[0]
        if base in _REGISTRATIONS:
            return base
        matches = [name for name in _REGISTRATIONS if lowered.startswith(name)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"No point-cloud registration is available for scene {scene_id!r}; "
            f"available registrations: {sorted(_REGISTRATIONS)}"
        )

    def _resolve_pcd_scene_dir(self, scene_id: str) -> Path:
        logical_name = str(scene_id).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if logical_name.lower().endswith(".tif"):
            logical_name = logical_name[:-4]
        direct = self._pcd_root / logical_name
        if direct.is_dir():
            return direct
        casefolded = logical_name.casefold()
        matches = [
            path
            for path in self._pcd_root.iterdir()
            if path.is_dir() and path.name.casefold() == casefolded
        ]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Point-cloud scene directory not found: {direct}. Expected "
            "DATASET.PCD_PATH/<scene_id>/"
        )

    def reset(self, scene_id: str) -> Dict[str, Any]:
        observations = super().reset(scene_id)
        pcd_scene_dir = self._resolve_pcd_scene_dir(scene_id)
        registration = _REGISTRATIONS[self._registration_key(scene_id)]
        if pcd_scene_dir != self._pcd_scene_dir:
            self._blocks = _build_or_load_index(
                pcd_scene_dir, chunk_points=self._chunk_points
            )
            self._pcd_scene_dir = pcd_scene_dir
            self._local_cache = None
        self._registration = registration
        self._last_render_key = None
        self._last_rgb = None
        self._last_render_stats = {}
        return observations

    def step(self, action: Union[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        # SatSim remains authoritative for motion and TIF-backed navigability.
        super().step(action)
        return self.get_observations()

    @staticmethod
    def _view_corners(
        center: np.ndarray,
        right_axis: np.ndarray,
        up_axis: np.ndarray,
        width_metres: float,
        height_metres: float,
    ) -> np.ndarray:
        half_right = right_axis * (width_metres / 2.0)
        half_up = up_axis * (height_metres / 2.0)
        return np.asarray(
            [
                center - half_right + half_up,
                center + half_right + half_up,
                center + half_right - half_up,
                center - half_right - half_up,
            ]
        )

    def _load_local_points(
        self, required_bounds: Tuple[float, float, float, float]
    ) -> _LocalPointCache:
        left, bottom, right, top = required_bounds
        padding = self._cache_padding_metres
        cache_bounds = (
            left - padding,
            bottom - padding,
            right + padding,
            top + padding,
        )
        selected_blocks = [
            block
            for block in self._blocks
            if block.intersects_xy(*cache_bounds)
        ]
        if not selected_blocks:
            raise RuntimeError(
                "No point-cloud block intersects the requested camera region"
            )

        retained: List[np.ndarray] = []
        cache_left, cache_bottom, cache_right, cache_top = cache_bounds
        for block in selected_blocks:
            layout = _parse_ply(block.path)
            for points in _iter_ply(layout, self._chunk_points):
                x = points[layout.fields["x"]]
                y = points[layout.fields["y"]]
                z = points[layout.fields["z"]]
                keep = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & (x >= cache_left)
                    & (x <= cache_right)
                    & (y >= cache_bottom)
                    & (y <= cache_top)
                )
                if not keep.any():
                    continue
                source = points[keep]
                compact = np.empty(len(source), dtype=_LOCAL_POINT_DTYPE)
                for coordinate in ("x", "y", "z"):
                    compact[coordinate] = source[layout.fields[coordinate]]
                for channel in ("red", "green", "blue"):
                    compact[channel] = _colors_as_uint8(
                        source[layout.fields[channel]]
                    )
                retained.append(compact)
        if not retained:
            raise RuntimeError("No point-cloud points found in the camera region")
        cache = _LocalPointCache(
            points=np.concatenate(retained), bounds=cache_bounds
        )
        self._local_cache = cache
        return cache

    def _estimate_ground(
        self, points: np.ndarray, center: np.ndarray, radius: float
    ) -> Tuple[float, int]:
        inside = (
            (points["x"] - center[0]) ** 2
            + (points["y"] - center[1]) ** 2
            <= radius * radius
        )
        elevations = points["z"][inside]
        elevations = elevations[np.isfinite(elevations)]
        if not len(elevations):
            raise RuntimeError("No point-cloud samples are available for ground estimation")
        return (
            float(np.percentile(elevations, self._ground_percentile)),
            int(len(elevations)),
        )

    def _render_points(
        self,
        points: np.ndarray,
        center: np.ndarray,
        right_axis: np.ndarray,
        up_axis: np.ndarray,
        camera_z: float,
        physical_width: float,
        physical_height: float,
        render_bounds: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        width = _INTERNAL_RENDER_WIDTH
        height = _INTERNAL_RENDER_HEIGHT
        altitude = float(self.get_agent_state().position[2])
        focal_x = width * altitude / physical_width
        focal_y = height * altitude / physical_height
        zbuffer = np.full(width * height, np.inf, dtype=np.float32)
        image = np.zeros((width * height, 3), dtype=np.uint8)
        radius = (self._point_size - 1) // 2
        offsets = [
            (du, dv)
            for dv in range(-radius, radius + 1)
            for du in range(-radius, radius + 1)
            if du * du + dv * dv <= radius * radius
        ] or [(0, 0)]
        left, bottom, right, top = render_bounds
        projected_count = 0
        zbuffer_updates = 0
        started = time.perf_counter()

        for start in range(0, len(points), self._chunk_points):
            selected = points[start : start + self._chunk_points]
            inside_xy = (
                (selected["x"] >= left)
                & (selected["x"] <= right)
                & (selected["y"] >= bottom)
                & (selected["y"] <= top)
            )
            selected = selected[inside_xy]
            if not len(selected):
                continue

            relative_x = selected["x"].astype(np.float64) - center[0]
            relative_y = selected["y"].astype(np.float64) - center[1]
            relative_z = selected["z"].astype(np.float64) - camera_z
            depth = -relative_z
            in_front = depth > self._near_clip_metres
            if not in_front.any():
                continue
            selected = selected[in_front]
            relative_x = relative_x[in_front]
            relative_y = relative_y[in_front]
            depth = depth[in_front]
            camera_right = relative_x * right_axis[0] + relative_y * right_axis[1]
            camera_up = relative_x * up_axis[0] + relative_y * up_axis[1]
            projected_u = width / 2.0 + camera_right * focal_x / depth
            projected_v = height / 2.0 - camera_up * focal_y / depth
            inside_image = (
                (projected_u >= 0.0)
                & (projected_u < width)
                & (projected_v >= 0.0)
                & (projected_v < height)
            )
            if not inside_image.any():
                continue
            u = np.floor(projected_u[inside_image]).astype(np.int64)
            v = np.floor(projected_v[inside_image]).astype(np.int64)
            depths = depth[inside_image].astype(np.float32)
            visible_points = selected[inside_image]
            colors = np.column_stack(
                [visible_points[channel] for channel in ("red", "green", "blue")]
            )
            projected_count += len(visible_points)
            for du, dv in offsets:
                shifted_u = u + du
                shifted_v = v + dv
                visible = (
                    (shifted_u >= 0)
                    & (shifted_u < width)
                    & (shifted_v >= 0)
                    & (shifted_v < height)
                )
                indices = shifted_v[visible] * width + shifted_u[visible]
                zbuffer_updates += _update_zbuffer(
                    indices,
                    depths[visible],
                    colors[visible],
                    zbuffer,
                    image,
                )

        valid_mask = np.isfinite(zbuffer).reshape(height, width)
        valid_pixels = int(valid_mask.sum())
        raw_image = image.reshape(height, width, 3)
        filled_image, hole_filled_pixels = _fill_one_pixel_holes(
            raw_image, valid_mask
        )
        return filled_image, {
            "render_seconds": time.perf_counter() - started,
            "points_in_memory": int(len(points)),
            "points_projected_inside_image": int(projected_count),
            "zbuffer_updates": int(zbuffer_updates),
            "valid_rendered_pixels": valid_pixels,
            "invalid_rendered_pixels": int(width * height - valid_pixels),
            "hole_filled_pixels": hole_filled_pixels,
            "remaining_unfilled_pixels": int(
                width * height - valid_pixels - hole_filled_pixels
            ),
        }

    def get_observations(self) -> Dict[str, Any]:
        if self._registration is None or self._pcd_scene_dir is None:
            raise RuntimeError("Point-cloud scene is not loaded; call reset() first")
        state = self.get_agent_state()
        longitude, latitude, altitude = state.position
        if altitude <= 0:
            raise ValueError("PCDSim camera altitude must be positive")
        x_3857, y_3857 = GeoUtils.wgs84_to_mercator(longitude, latitude)
        rotation = float(state.rotation) % 360.0
        render_key = (
            self._scene_id,
            float(x_3857),
            float(y_3857),
            float(altitude),
            rotation,
            self.rgb_width,
            self.rgb_height,
            float(self.rgb_hfov),
        )
        if render_key == self._last_render_key and self._last_rgb is not None:
            return {"rgb": self._last_rgb}

        azimuth = math.radians(rotation)
        epsg_right = np.asarray(
            [math.cos(azimuth), -math.sin(azimuth)], dtype=np.float64
        )
        epsg_up = np.asarray(
            [math.sin(azimuth), math.cos(azimuth)], dtype=np.float64
        )
        aspect_ratio = self.rgb_width / self.rgb_height
        physical_width_requested = 2.0 * altitude * math.tan(
            math.radians(self.rgb_hfov / 2.0)
        )
        physical_height_requested = physical_width_requested / aspect_ratio
        epsg_units_per_physical_metre = float(
            math.sqrt(
                abs(np.linalg.det(self._registration.local_to_epsg3857_matrix))
            )
        )
        epsg_center = np.asarray([x_3857, y_3857], dtype=np.float64)
        epsg_corners = self._view_corners(
            epsg_center,
            epsg_right,
            epsg_up,
            physical_width_requested * epsg_units_per_physical_metre,
            physical_height_requested * epsg_units_per_physical_metre,
        )
        local_corners = np.asarray(
            [
                self._registration.epsg3857_to_local(float(x), float(y))
                for x, y in epsg_corners
            ]
        )
        top_left, top_right, _, bottom_left = local_corners
        local_center = local_corners.mean(axis=0)
        local_right = top_right - top_left
        local_right /= np.linalg.norm(local_right)
        local_up = (top_left + top_right) / 2.0 - local_center
        local_up /= np.linalg.norm(local_up)
        physical_width = float(np.linalg.norm(top_right - top_left))
        physical_height = float(np.linalg.norm(top_left - bottom_left))
        render_bounds = (
            float(local_corners[:, 0].min() - 1.0),
            float(local_corners[:, 1].min() - 1.0),
            float(local_corners[:, 0].max() + 1.0),
            float(local_corners[:, 1].max() + 1.0),
        )
        ground_radius = max(50.0, min(physical_width, physical_height) / 2.0)
        required_bounds = (
            min(render_bounds[0], float(local_center[0] - ground_radius)),
            min(render_bounds[1], float(local_center[1] - ground_radius)),
            max(render_bounds[2], float(local_center[0] + ground_radius)),
            max(render_bounds[3], float(local_center[1] + ground_radius)),
        )
        cache = self._local_cache
        cache_reloaded = cache is None or not cache.contains(required_bounds)
        if cache_reloaded:
            cache = self._load_local_points(required_bounds)
        assert cache is not None
        ground_z, ground_sample_count = self._estimate_ground(
            cache.points, local_center, ground_radius
        )
        rgb, stats = self._render_points(
            cache.points,
            local_center,
            local_right,
            local_up,
            ground_z + altitude,
            physical_width,
            physical_height,
            render_bounds,
        )
        resize_started = time.perf_counter()
        if rgb.shape[:2] != (self.rgb_height, self.rgb_width):
            interpolation = (
                cv2.INTER_AREA
                if self.rgb_width <= _INTERNAL_RENDER_WIDTH
                and self.rgb_height <= _INTERNAL_RENDER_HEIGHT
                else cv2.INTER_LINEAR
            )
            rgb = cv2.resize(
                rgb,
                (self.rgb_width, self.rgb_height),
                interpolation=interpolation,
            )
        resize_seconds = time.perf_counter() - resize_started
        stats.update(
            {
                "scene_id": self._scene_id,
                "pointcloud_scene_dir": str(self._pcd_scene_dir),
                "internal_render_width": _INTERNAL_RENDER_WIDTH,
                "internal_render_height": _INTERNAL_RENDER_HEIGHT,
                "output_width": self.rgb_width,
                "output_height": self.rgb_height,
                "resize_seconds": resize_seconds,
                "cache_reloaded": bool(cache_reloaded),
                "cache_bounds": list(cache.bounds),
                "estimated_ground_z_metres": ground_z,
                "ground_sample_count": ground_sample_count,
            }
        )
        self._last_render_key = render_key
        self._last_rgb = rgb
        self._last_render_stats = stats
        return {"rgb": rgb}

    @property
    def last_render_stats(self) -> Mapping[str, Any]:
        """Return diagnostics for the latest non-cached point-cloud render."""
        return dict(self._last_render_stats)

    def close(self) -> None:
        self._local_cache = None
        self._blocks = []
        self._last_rgb = None
        self._last_render_key = None
        super().close()


__all__ = ["PCDSimWrapper"]
