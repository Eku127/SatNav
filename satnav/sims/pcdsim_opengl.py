#!/usr/bin/env python3
"""GPU point-cloud renderer with the public SatSim simulator interface.

PCDSim OpenGL keeps SatSim authoritative for geographic state, movement,
navigability, metrics, and TIF-backed top-down maps.  RGB observations are
rendered by a persistent ModernGL pipeline.  PLY blocks are uploaded lazily,
kept resident while they remain useful, and evicted after a configurable
number of consecutive local views without an intersection.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import moderngl
import numpy as np
from omegaconf import DictConfig

from satnav.sims.pcdsim import (
    _PointBlock,
    _config_value,
    _fill_one_pixel_holes,
    _parse_ply,
    PCDSimWrapper,
)
from satnav.sims.satsim.geoutils import GeoUtils


_VERTEX_SHADER = """#version 330
in vec3 in_position;
in vec3 in_color;

uniform vec3 camera_position;
uniform vec3 image_right;
uniform vec3 image_up;
uniform vec3 camera_forward;
uniform float projection_scale_x;
uniform float projection_scale_y;
uniform float near_clip;
uniform float far_clip;
uniform float point_size;

out vec3 vertex_color;

void main() {
    vec3 relative = in_position - camera_position;
    float camera_x = dot(relative, image_right);
    float camera_y = dot(relative, image_up);
    float depth = dot(relative, camera_forward);

    float depth_a = (far_clip + near_clip) / (far_clip - near_clip);
    float depth_b = -(2.0 * far_clip * near_clip) / (far_clip - near_clip);
    gl_Position = vec4(
        camera_x * projection_scale_x,
        camera_y * projection_scale_y,
        depth_a * depth + depth_b,
        depth
    );
    gl_PointSize = point_size;
    vertex_color = in_color;
}
"""


_FRAGMENT_SHADER = """#version 330
in vec3 vertex_color;
uniform bool circular_points;
out vec4 fragment_color;

void main() {
    if (circular_points) {
        vec2 point_offset = gl_PointCoord - vec2(0.5);
        if (dot(point_offset, point_offset) > 0.25) {
            discard;
        }
    }
    fragment_color = vec4(vertex_color, 1.0);
}
"""


@dataclass(frozen=True)
class _GroundHeightGrid:
    path: Path
    grid_size_metres: float
    anchor_x: float
    anchor_y: float
    x_index_min: int
    y_index_min: int
    heights: np.ndarray
    point_counts: np.ndarray
    default_height_metres: float
    default_height_source: str

    def lookup(self, center_xy: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        grid_x = (float(center_xy[0]) - self.anchor_x) / self.grid_size_metres
        grid_y = (float(center_xy[1]) - self.anchor_y) / self.grid_size_metres
        rows, columns = self.heights.shape
        if rows < 2 or columns < 2:
            raise ValueError("Ground-height grid must contain at least 2x2 samples")

        column = math.floor(grid_x + 1e-9) - self.x_index_min
        row = math.floor(grid_y + 1e-9) - self.y_index_min
        if not (0 <= row < rows and 0 <= column < columns):
            raise ValueError("Camera center is outside the point-cloud height grid")

        # Heights are treated as samples at grid origins. Clamp the base at
        # the outer edge so the final row/column still has four source samples.
        column0 = min(column, columns - 2)
        row0 = min(row, rows - 2)
        x_index0 = self.x_index_min + column0
        y_index0 = self.y_index_min + row0
        fraction_x = float(np.clip(grid_x - x_index0, 0.0, 1.0))
        fraction_y = float(np.clip(grid_y - y_index0, 0.0, 1.0))
        source_coordinates = (
            (row0, column0, x_index0, y_index0),
            (row0, column0 + 1, x_index0 + 1, y_index0),
            (row0 + 1, column0, x_index0, y_index0 + 1),
            (row0 + 1, column0 + 1, x_index0 + 1, y_index0 + 1),
        )
        raw_weights = np.asarray(
            [
                (1.0 - fraction_x) * (1.0 - fraction_y),
                fraction_x * (1.0 - fraction_y),
                (1.0 - fraction_x) * fraction_y,
                fraction_x * fraction_y,
            ],
            dtype=np.float64,
        )
        source_heights = np.asarray(
            [
                self.heights[source_row, source_column]
                for source_row, source_column, _, _ in source_coordinates
            ],
            dtype=np.float64,
        )
        valid_sources = np.isfinite(source_heights)
        weights = np.where(valid_sources, raw_weights, 0.0)
        weight_sum = float(weights.sum())
        fallback_used = not bool(valid_sources.any())
        if fallback_used:
            weights = np.zeros(4, dtype=np.float64)
            height = self.default_height_metres
        else:
            if weight_sum <= 0.0:
                # This only occurs at an exact null-valued origin while one or
                # more zero-weight neighbouring origins are valid. Prefer the
                # closest valid origins instead of using the scene fallback.
                source_offsets = np.asarray(
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    dtype=np.float64,
                )
                distances = np.linalg.norm(
                    source_offsets - np.asarray([fraction_x, fraction_y]),
                    axis=1,
                )
                weights = np.where(
                    valid_sources,
                    1.0 / np.maximum(distances, 1e-12),
                    0.0,
                )
                weight_sum = float(weights.sum())
            weights /= weight_sum
            height = float(
                np.sum(source_heights[valid_sources] * weights[valid_sources])
            )
        source_samples = []
        weighted_point_count = 0.0
        for index, (source_row, source_column, x_index, y_index) in enumerate(
            source_coordinates
        ):
            point_count = int(self.point_counts[source_row, source_column])
            weighted_point_count += point_count * float(weights[index])
            source_samples.append(
                {
                    "x_index": x_index,
                    "y_index": y_index,
                    "origin_xy_metres": [
                        self.anchor_x + x_index * self.grid_size_metres,
                        self.anchor_y + y_index * self.grid_size_metres,
                    ],
                    "height_metres": (
                        float(source_heights[index])
                        if valid_sources[index]
                        else None
                    ),
                    "point_count": point_count,
                    "weight": float(weights[index]),
                }
            )
        return height, {
            "method": (
                "default_height_no_valid_origins"
                if fallback_used
                else "bilinear_four_grid_origins"
            ),
            "source_path": str(self.path),
            "height_metres": height,
            "point_count": int(round(weighted_point_count)),
            "fallback_used": fallback_used,
            "default_height_metres": self.default_height_metres,
            "default_height_source": self.default_height_source,
            "grid_fraction_xy": [fraction_x, fraction_y],
            "source_samples": source_samples,
        }


@dataclass
class _GpuChunk:
    vertex_buffer: Any
    vertex_array: Any
    point_count: int
    byte_count: int


@dataclass
class _GpuCloud:
    block: _PointBlock
    chunks: List[_GpuChunk]
    point_count: int
    byte_count: int
    unused_view_count: int = 0


def _local_view_intersects_block(
    local_view_corners: np.ndarray,
    block: _PointBlock,
) -> bool:
    """Test a rotated local-view quadrilateral against a block XY rectangle.

    This uses the separating-axis theorem and only the block bounds from the
    spatial index.  No PLY bytes or point samples are touched during selection.
    """
    min_x, min_y, _, max_x, max_y, _ = block.bounds
    rectangle_center = np.asarray(
        [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0],
        dtype=np.float64,
    )
    rectangle_half_extent = np.asarray(
        [(max_x - min_x) / 2.0, (max_y - min_y) / 2.0],
        dtype=np.float64,
    )
    edges = np.roll(local_view_corners, -1, axis=0) - local_view_corners
    axes = [
        np.asarray([1.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    ]
    axes.extend(np.column_stack((-edges[:, 1], edges[:, 0])))
    for axis in axes:
        if not np.any(axis):
            continue
        polygon_projection = local_view_corners @ axis
        rectangle_center_projection = float(rectangle_center @ axis)
        rectangle_radius = float(np.abs(axis) @ rectangle_half_extent)
        rectangle_min = rectangle_center_projection - rectangle_radius
        rectangle_max = rectangle_center_projection + rectangle_radius
        if (
            float(polygon_projection.max()) < rectangle_min
            or float(polygon_projection.min()) > rectangle_max
        ):
            return False
    return True


class PCDSimOpenGLWrapper(PCDSimWrapper):
    """SatSim-compatible point-cloud simulator backed by persistent OpenGL."""

    def __init__(
        self,
        config: Union[DictConfig, dict],
        scenes_dir: Optional[str] = None,
    ) -> None:
        super().__init__(config, scenes_dir)
        sim_config = _config_value(config, "SIMULATOR", config)
        pcd_config = _config_value(sim_config, "PCDSIM", {})
        opengl_config = _config_value(sim_config, "PCDSIM_OPENGL", {})
        if _config_value(pcd_config, "POINT_SIZE", None) is None:
            # OpenGL rendering uses a denser framebuffer than V1, so its
            # standalone default is a three-pixel OpenGL point.
            self._point_size = 3
        self._height_grid_filename = str(
            _config_value(
                opengl_config,
                "HEIGHT_GRID_FILENAME",
                "ground_height_grid_20m.json",
            )
        )
        self._max_gpu_buffer_bytes = int(
            _config_value(opengl_config, "MAX_GPU_BUFFER_BYTES", 1_900_000_000)
        )
        self._unused_view_limit = int(
            _config_value(opengl_config, "UNUSED_VIEW_LIMIT", 10)
        )
        self._render_width = int(
            _config_value(opengl_config, "RENDER_WIDTH", 2048)
        )
        self._render_height = int(
            _config_value(opengl_config, "RENDER_HEIGHT", 2048)
        )
        self._point_shape = str(
            _config_value(opengl_config, "POINT_SHAPE", "circle")
        ).strip().lower()
        self._msaa_samples = int(
            _config_value(opengl_config, "MSAA_SAMPLES", 4)
        )
        configured_default_height = _config_value(
            opengl_config, "DEFAULT_GROUND_HEIGHT_METRES", None
        )
        self._configured_default_ground_height = (
            None
            if configured_default_height is None
            else float(configured_default_height)
        )
        require_gl_version = int(
            _config_value(opengl_config, "REQUIRE_GL_VERSION", 330)
        )
        if self._max_gpu_buffer_bytes <= 0:
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL.MAX_GPU_BUFFER_BYTES must be positive"
            )
        if self._unused_view_limit <= 0:
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL.UNUSED_VIEW_LIMIT must be positive"
            )
        if self._render_width <= 0 or self._render_height <= 0:
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL render dimensions must be positive"
            )
        if self._point_shape not in {"square", "circle"}:
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL.POINT_SHAPE must be 'square' or 'circle'"
            )
        if self._msaa_samples < 0:
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL.MSAA_SAMPLES cannot be negative"
            )
        if (
            self._configured_default_ground_height is not None
            and not np.isfinite(self._configured_default_ground_height)
        ):
            raise ValueError(
                "SIMULATOR.PCDSIM_OPENGL.DEFAULT_GROUND_HEIGHT_METRES must be finite"
            )

        self._ground_grid: Optional[_GroundHeightGrid] = None
        self._gpu_clouds: Dict[Path, _GpuCloud] = {}
        self._gl_context: Any = None
        self._gl_program: Any = None
        self._color_texture: Any = None
        self._multisample_color_buffer: Any = None
        self._depth_buffer: Any = None
        self._framebuffer: Any = None
        self._resolve_framebuffer: Any = None
        self._gpu_info: Dict[str, Any] = {}
        try:
            self._gl_context = moderngl.create_standalone_context(
                require=require_gl_version
            )
            self._gl_program = self._gl_context.program(
                vertex_shader=_VERTEX_SHADER,
                fragment_shader=_FRAGMENT_SHADER,
            )
            self._color_texture = self._gl_context.texture(
                (self._render_width, self._render_height),
                components=4,
                dtype="f1",
            )
            max_samples = int(self._gl_context.info.get("GL_MAX_SAMPLES", 0))
            if self._msaa_samples > max_samples:
                raise ValueError(
                    f"Requested {self._msaa_samples}x MSAA, but the OpenGL "
                    f"context supports at most {max_samples} samples"
                )
            if self._msaa_samples > 0:
                self._multisample_color_buffer = self._gl_context.renderbuffer(
                    (self._render_width, self._render_height),
                    components=4,
                    samples=self._msaa_samples,
                    dtype="f1",
                )
                self._depth_buffer = self._gl_context.depth_renderbuffer(
                    (self._render_width, self._render_height),
                    samples=self._msaa_samples,
                )
                self._framebuffer = self._gl_context.framebuffer(
                    color_attachments=[self._multisample_color_buffer],
                    depth_attachment=self._depth_buffer,
                )
                self._resolve_framebuffer = self._gl_context.framebuffer(
                    color_attachments=[self._color_texture]
                )
            else:
                self._depth_buffer = self._gl_context.depth_renderbuffer(
                    (self._render_width, self._render_height)
                )
                self._framebuffer = self._gl_context.framebuffer(
                    color_attachments=[self._color_texture],
                    depth_attachment=self._depth_buffer,
                )
            self._framebuffer.use()
            self._gl_context.viewport = (
                0,
                0,
                self._render_width,
                self._render_height,
            )
            self._gl_context.enable_only(
                moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
            )
            self._gpu_info = {
                "vendor": self._gl_context.info.get("GL_VENDOR"),
                "renderer": self._gl_context.info.get("GL_RENDERER"),
                "version": self._gl_context.info.get("GL_VERSION"),
                "moderngl_version": moderngl.__version__,
            }
        except Exception:
            self._release_gl_resources()
            raise

    def _load_ground_grid(self, path: Path) -> _GroundHeightGrid:
        if not path.is_file():
            raise FileNotFoundError(
                f"PCDSim OpenGL ground-height grid does not exist: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = int(payload["rows"])
        columns = int(payload["columns"])
        heights = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in payload["heights_metres"]
            ],
            dtype=np.float32,
        )
        point_counts = np.asarray(payload["point_counts"], dtype=np.int64)
        if heights.shape != (rows, columns):
            raise ValueError(
                f"Height grid shape {heights.shape} does not match {(rows, columns)}"
            )
        if point_counts.shape != heights.shape:
            raise ValueError("Height-grid point_counts shape does not match heights")
        anchor_x, anchor_y = (
            float(value) for value in payload["grid_anchor_xy_metres"]
        )
        grid_size = float(payload["grid_size_metres"])
        if grid_size <= 0:
            raise ValueError("Height-grid cell size must be positive")
        if self._configured_default_ground_height is None:
            finite_heights = heights[np.isfinite(heights)]
            default_height = (
                float(np.median(finite_heights)) if finite_heights.size else 0.0
            )
            default_height_source = (
                "grid_finite_height_median"
                if finite_heights.size
                else "zero_no_finite_grid_heights"
            )
        else:
            default_height = self._configured_default_ground_height
            default_height_source = "configured"
        return _GroundHeightGrid(
            path=path,
            grid_size_metres=grid_size,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            x_index_min=int(payload["x_index_min"]),
            y_index_min=int(payload["y_index_min"]),
            heights=heights,
            point_counts=point_counts,
            default_height_metres=default_height,
            default_height_source=default_height_source,
        )

    def reset(self, scene_id: str) -> Dict[str, Any]:
        previous_scene_dir = self._pcd_scene_dir
        observations = super().reset(scene_id)
        if self._pcd_scene_dir != previous_scene_dir:
            self._release_all_gpu_clouds()
            assert self._pcd_scene_dir is not None
            self._ground_grid = self._load_ground_grid(
                self._pcd_scene_dir / self._height_grid_filename
            )
        elif self._ground_grid is None:
            assert self._pcd_scene_dir is not None
            self._ground_grid = self._load_ground_grid(
                self._pcd_scene_dir / self._height_grid_filename
            )
        return observations

    @staticmethod
    def _gpu_layout(layout: Any) -> str:
        expected_offsets = {
            "x": 0,
            "y": 4,
            "z": 8,
            "red": 12,
            "green": 13,
            "blue": 14,
        }
        for field, expected_offset in expected_offsets.items():
            source_name = layout.fields[field]
            actual_offset = int(layout.dtype.fields[source_name][1])
            if actual_offset != expected_offset:
                raise ValueError(
                    f"{layout.path.name}: GPU field {field!r} is at byte "
                    f"offset {actual_offset}, expected {expected_offset}"
                )
        if layout.dtype.itemsize == 15:
            return "3f 3f1"
        if layout.dtype.itemsize == 16:
            return "3f 3f1 1x"
        raise ValueError(
            f"{layout.path.name}: direct GPU upload requires 15-byte or "
            f"16-byte XYZRGB records, got {layout.dtype.itemsize}"
        )

    def _upload_block(self, block: _PointBlock) -> Tuple[_GpuCloud, float]:
        if self._gl_context is None or self._gl_program is None:
            raise RuntimeError("PCDSim OpenGL context is closed")
        started = time.perf_counter()
        layout = _parse_ply(block.path)
        vertex_format = self._gpu_layout(layout)
        points = np.memmap(
            block.path,
            dtype=layout.dtype,
            mode="r",
            offset=layout.data_offset,
            shape=(layout.point_count,),
        )
        max_chunk_points = max(
            1, self._max_gpu_buffer_bytes // layout.dtype.itemsize
        )
        chunks: List[_GpuChunk] = []
        try:
            for start in range(0, layout.point_count, max_chunk_points):
                stop = min(start + max_chunk_points, layout.point_count)
                point_chunk = points[start:stop]
                vertex_buffer = None
                vertex_array = None
                try:
                    vertex_buffer = self._gl_context.buffer(point_chunk)
                    vertex_array = self._gl_context.vertex_array(
                        self._gl_program,
                        [
                            (
                                vertex_buffer,
                                vertex_format,
                                "in_position",
                                "in_color",
                            )
                        ],
                    )
                    chunks.append(
                        _GpuChunk(
                            vertex_buffer=vertex_buffer,
                            vertex_array=vertex_array,
                            point_count=stop - start,
                            byte_count=(stop - start) * layout.dtype.itemsize,
                        )
                    )
                except Exception:
                    if vertex_array is not None:
                        vertex_array.release()
                    if vertex_buffer is not None:
                        vertex_buffer.release()
                    raise
            self._gl_context.finish()
        except Exception:
            for chunk in chunks:
                chunk.vertex_array.release()
                chunk.vertex_buffer.release()
            raise
        finally:
            del points
        return (
            _GpuCloud(
                block=block,
                chunks=chunks,
                point_count=layout.point_count,
                byte_count=layout.point_count * layout.dtype.itemsize,
            ),
            time.perf_counter() - started,
        )

    @staticmethod
    def _release_cloud(cloud: _GpuCloud) -> None:
        for chunk in cloud.chunks:
            chunk.vertex_array.release()
            chunk.vertex_buffer.release()

    def _release_all_gpu_clouds(self) -> None:
        for cloud in self._gpu_clouds.values():
            self._release_cloud(cloud)
        self._gpu_clouds.clear()
        if self._gl_context is not None:
            self._gl_context.finish()

    def _sync_gpu_clouds(
        self,
        selected_blocks: Sequence[_PointBlock],
    ) -> Dict[str, Any]:
        selected_by_path = {block.path: block for block in selected_blocks}
        uploaded_files: List[str] = []
        upload_seconds = 0.0
        for path in sorted(selected_by_path, key=lambda item: item.name):
            if path in self._gpu_clouds:
                continue
            cloud, elapsed = self._upload_block(selected_by_path[path])
            self._gpu_clouds[path] = cloud
            uploaded_files.append(path.name)
            upload_seconds += elapsed

        evicted_files: List[str] = []
        for path, cloud in list(self._gpu_clouds.items()):
            if path in selected_by_path:
                cloud.unused_view_count = 0
                continue
            cloud.unused_view_count += 1
            if cloud.unused_view_count < self._unused_view_limit:
                continue
            self._release_cloud(cloud)
            del self._gpu_clouds[path]
            evicted_files.append(path.name)
        if evicted_files and self._gl_context is not None:
            self._gl_context.finish()

        return {
            "uploaded_ply_files": uploaded_files,
            "evicted_ply_files": sorted(evicted_files),
            "gpu_upload_seconds": upload_seconds,
        }

    def _release_gl_resources(self) -> None:
        try:
            self._release_all_gpu_clouds()
        except Exception:
            self._gpu_clouds.clear()
        for attribute in (
            "_resolve_framebuffer",
            "_framebuffer",
            "_depth_buffer",
            "_multisample_color_buffer",
            "_color_texture",
            "_gl_program",
        ):
            resource = getattr(self, attribute, None)
            if resource is not None:
                try:
                    resource.release()
                finally:
                    setattr(self, attribute, None)
        if self._gl_context is not None:
            try:
                self._gl_context.release()
            finally:
                self._gl_context = None

    def get_observations(self) -> Dict[str, Any]:
        if (
            self._registration is None
            or self._pcd_scene_dir is None
            or self._ground_grid is None
        ):
            raise RuntimeError("Point-cloud scene is not loaded; call reset() first")
        if self._framebuffer is None or self._gl_context is None:
            raise RuntimeError("PCDSim OpenGL resources are closed")

        state = self.get_agent_state()
        longitude, latitude, altitude = state.position
        if altitude <= 0:
            raise ValueError("PCDSim OpenGL camera altitude must be positive")
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

        observation_started = time.perf_counter()
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
        local_right_xy = top_right - top_left
        local_right_xy /= np.linalg.norm(local_right_xy)
        local_up_xy = (top_left + top_right) / 2.0 - local_center
        local_up_xy /= np.linalg.norm(local_up_xy)
        physical_width = float(np.linalg.norm(top_right - top_left))
        physical_height = float(np.linalg.norm(top_left - bottom_left))
        view_bounds = (
            float(local_corners[:, 0].min()),
            float(local_corners[:, 1].min()),
            float(local_corners[:, 0].max()),
            float(local_corners[:, 1].max()),
        )
        selected_blocks = sorted(
            [
                block
                for block in self._blocks
                if _local_view_intersects_block(local_corners, block)
            ],
            key=lambda block: block.path.name,
        )
        if not selected_blocks:
            raise RuntimeError(
                f"No PLY block intersects local view bounds {view_bounds}"
            )

        ground_z, ground_metadata = self._ground_grid.lookup(local_center)
        camera_position = np.asarray(
            [local_center[0], local_center[1], ground_z + altitude],
            dtype=np.float32,
        )
        far_clip = max(
            100.0,
            float(camera_position[2])
            - min(block.bounds[2] for block in selected_blocks)
            + 10.0,
        )
        cache_stats = self._sync_gpu_clouds(selected_blocks)

        self._framebuffer.use()
        self._framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        program = self._gl_program
        program["camera_position"].value = tuple(camera_position)
        program["image_right"].value = (
            float(local_right_xy[0]),
            float(local_right_xy[1]),
            0.0,
        )
        program["image_up"].value = (
            float(local_up_xy[0]),
            float(local_up_xy[1]),
            0.0,
        )
        program["camera_forward"].value = (0.0, 0.0, -1.0)
        program["projection_scale_x"].value = 2.0 * altitude / physical_width
        program["projection_scale_y"].value = 2.0 * altitude / physical_height
        program["near_clip"].value = self._near_clip_metres
        program["far_clip"].value = far_clip
        program["point_size"].value = float(self._point_size)
        program["circular_points"].value = self._point_shape == "circle"

        draw_started = time.perf_counter()
        selected_paths = {block.path for block in selected_blocks}
        rendered_point_count = 0
        for path in sorted(selected_paths, key=lambda item: item.name):
            cloud = self._gpu_clouds[path]
            rendered_point_count += cloud.point_count
            for chunk in cloud.chunks:
                chunk.vertex_array.render(
                    mode=moderngl.POINTS,
                    vertices=chunk.point_count,
                )
        self._gl_context.finish()
        gpu_draw_seconds = time.perf_counter() - draw_started

        resolve_started = time.perf_counter()
        read_framebuffer = self._framebuffer
        if self._resolve_framebuffer is not None:
            self._gl_context.copy_framebuffer(
                self._resolve_framebuffer,
                self._framebuffer,
            )
            read_framebuffer = self._resolve_framebuffer
        msaa_resolve_seconds = time.perf_counter() - resolve_started

        readback_started = time.perf_counter()
        raw_rgba = read_framebuffer.read(components=4, alignment=1)
        rgba = np.frombuffer(raw_rgba, dtype=np.uint8).reshape(
            self._render_height,
            self._render_width,
            4,
        )[::-1].copy()
        readback_seconds = time.perf_counter() - readback_started

        valid_mask = rgba[:, :, 3] > 0
        valid_pixels = int(valid_mask.sum())
        hole_fill_started = time.perf_counter()
        filled_image, hole_filled_pixels = _fill_one_pixel_holes(
            rgba[:, :, :3], valid_mask
        )
        hole_fill_seconds = time.perf_counter() - hole_fill_started

        resize_started = time.perf_counter()
        if filled_image.shape[:2] != (self.rgb_height, self.rgb_width):
            interpolation = (
                cv2.INTER_AREA
                if self.rgb_width <= self._render_width
                and self.rgb_height <= self._render_height
                else cv2.INTER_LINEAR
            )
            rgb = cv2.resize(
                filled_image,
                (self.rgb_width, self.rgb_height),
                interpolation=interpolation,
            )
        else:
            rgb = filled_image
        resize_seconds = time.perf_counter() - resize_started

        resident_clouds = sorted(
            self._gpu_clouds.values(), key=lambda cloud: cloud.block.path.name
        )
        stats: Dict[str, Any] = {
            "pipeline": "moderngl_persistent_opengl",
            "scene_id": self._scene_id,
            "pointcloud_scene_dir": str(self._pcd_scene_dir),
            "selected_ply_files": [
                block.path.name for block in selected_blocks
            ],
            "selected_ply_count": len(selected_blocks),
            "rendered_point_count": rendered_point_count,
            "resident_ply_files": [
                cloud.block.path.name for cloud in resident_clouds
            ],
            "resident_ply_count": len(resident_clouds),
            "resident_gpu_bytes": int(
                sum(cloud.byte_count for cloud in resident_clouds)
            ),
            "ply_unused_view_counts": {
                cloud.block.path.name: cloud.unused_view_count
                for cloud in resident_clouds
            },
            "unused_view_limit": self._unused_view_limit,
            "uploaded_ply_files": cache_stats["uploaded_ply_files"],
            "evicted_ply_files": cache_stats["evicted_ply_files"],
            "gpu_upload_seconds": cache_stats["gpu_upload_seconds"],
            "gpu_draw_seconds": gpu_draw_seconds,
            "msaa_resolve_seconds": msaa_resolve_seconds,
            "gpu_readback_seconds": readback_seconds,
            "hole_fill_seconds": hole_fill_seconds,
            "resize_seconds": resize_seconds,
            "observation_seconds": time.perf_counter() - observation_started,
            "internal_render_width": self._render_width,
            "internal_render_height": self._render_height,
            "point_size": self._point_size,
            "point_shape": self._point_shape,
            "msaa_samples": self._msaa_samples,
            "output_width": self.rgb_width,
            "output_height": self.rgb_height,
            "valid_rendered_pixels": valid_pixels,
            "invalid_rendered_pixels": int(
                self._render_width * self._render_height - valid_pixels
            ),
            "hole_filled_pixels": hole_filled_pixels,
            "remaining_unfilled_pixels": int(
                self._render_width
                * self._render_height
                - valid_pixels
                - hole_filled_pixels
            ),
            "ground": ground_metadata,
            "camera_position_pointcloud_xyz_metres": camera_position.tolist(),
            "local_view_bounds_xy_metres": list(view_bounds),
            "ply_intersection_method": "local_view_polygon_vs_block_aabb_sat",
            "near_clip_metres": self._near_clip_metres,
            "far_clip_metres": far_clip,
            "gpu": dict(self._gpu_info),
        }
        self._last_render_key = render_key
        self._last_rgb = rgb
        self._last_render_stats = stats
        return {"rgb": rgb}

    def close(self) -> None:
        self._ground_grid = None
        self._release_gl_resources()
        super().close()


__all__ = ["PCDSimOpenGLWrapper"]
