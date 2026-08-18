"""Renderer-independent vector top-down visualization for task episodes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def _coordinate_frame_name(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("type", value.get("name", ""))
    return str(value or "wgs84").strip().lower()


def _to_local_xy(
    positions: Sequence[Sequence[float]],
    coordinate_frame: object,
) -> np.ndarray:
    """Convert public episode positions to a local metric drawing plane."""
    values = np.asarray([[float(p[0]), float(p[1])] for p in positions], dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)

    frame_name = _coordinate_frame_name(coordinate_frame)
    if "enu" in frame_name or frame_name in {"local", "cartesian", "metric"}:
        return values - values[0]

    lon0 = float(np.mean(values[:, 0]))
    lat0 = float(np.mean(values[:, 1]))
    latitude_scale = 111_132.0
    longitude_scale = 111_320.0 * max(math.cos(math.radians(lat0)), 1e-6)
    return np.column_stack(
        (
            (values[:, 0] - lon0) * longitude_scale,
            (values[:, 1] - lat0) * latitude_scale,
        )
    )


def render_vector_topdown(
    *,
    reference_path: Sequence[Sequence[float]],
    agent_path: Sequence[Sequence[float]],
    agent_position: Sequence[float],
    agent_heading: float,
    current_waypoint_index: int,
    goal_radius: float,
    coordinate_frame: object = "wgs84",
    image_size: Tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Draw path state without requiring a GeoTIFF or simulator internals.

    The returned image is RGB uint8. Heading follows SatNav's convention:
    zero degrees is north and positive angles rotate clockwise.
    """
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")

    reference = [list(point) for point in reference_path]
    travelled = [list(point) for point in agent_path]
    current = list(agent_position)
    positions = reference + travelled + [current]
    if not positions:
        return np.full((height, width, 3), 245, dtype=np.uint8)

    local = _to_local_xy(positions, coordinate_frame)
    reference_xy = local[: len(reference)]
    travelled_xy = local[len(reference): len(reference) + len(travelled)]
    current_xy = local[-1]

    min_xy = np.min(local, axis=0)
    max_xy = np.max(local, axis=0)
    extent = np.maximum(max_xy - min_xy, 1.0)
    padding_m = max(float(np.max(extent)) * 0.12, float(goal_radius) * 1.25, 10.0)
    min_xy -= padding_m
    max_xy += padding_m

    margin_px = 34
    drawable_width = max(width - 2 * margin_px, 1)
    drawable_height = max(height - 2 * margin_px, 1)
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min(drawable_width / span[0], drawable_height / span[1])
    center = (min_xy + max_xy) / 2.0
    pixel_center = np.asarray([width / 2.0, height / 2.0], dtype=float)

    def pixels(points: Iterable[Sequence[float]]) -> np.ndarray:
        points_array = np.asarray(list(points), dtype=float).reshape((-1, 2))
        result = np.empty_like(points_array)
        result[:, 0] = pixel_center[0] + (points_array[:, 0] - center[0]) * scale
        result[:, 1] = pixel_center[1] - (points_array[:, 1] - center[1]) * scale
        return np.rint(result).astype(np.int32)

    image = np.full((height, width, 3), 245, dtype=np.uint8)

    grid_spacing_m = max(10.0, 10.0 ** math.floor(math.log10(max(span) / 5.0)))
    for axis in range(2):
        start = math.floor(min_xy[axis] / grid_spacing_m) * grid_spacing_m
        value = start
        while value <= max_xy[axis] + 1e-6:
            if axis == 0:
                line = pixels([[value, min_xy[1]], [value, max_xy[1]]])
            else:
                line = pixels([[min_xy[0], value], [max_xy[0], value]])
            cv2.line(image, tuple(line[0]), tuple(line[1]), (222, 226, 230), 1, cv2.LINE_AA)
            value += grid_spacing_m

    if len(reference_xy) >= 2:
        cv2.polylines(
            image,
            [pixels(reference_xy)],
            False,
            (65, 105, 225),
            3,
            cv2.LINE_AA,
        )

    if len(travelled_xy) >= 2:
        cv2.polylines(
            image,
            [pixels(travelled_xy)],
            False,
            (242, 153, 32),
            3,
            cv2.LINE_AA,
        )

    if len(reference_xy):
        reference_pixels = pixels(reference_xy)
        cv2.circle(image, tuple(reference_pixels[0]), 7, (30, 170, 80), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(reference_pixels[-1]), 7, (220, 55, 55), -1, cv2.LINE_AA)
        waypoint_index = min(max(int(current_waypoint_index) + 1, 1), len(reference_pixels) - 1)
        if len(reference_pixels) > 1:
            radius_px = max(5, int(round(float(goal_radius) * scale)))
            radius_px = min(radius_px, max(width, height))
            cv2.circle(
                image,
                tuple(reference_pixels[waypoint_index]),
                radius_px,
                (230, 170, 20),
                2,
                cv2.LINE_AA,
            )

    agent_pixel = pixels([current_xy])[0]
    heading = math.radians(float(agent_heading))
    forward = np.asarray([math.sin(heading), -math.cos(heading)])
    right = np.asarray([math.cos(heading), math.sin(heading)])
    triangle = np.asarray(
        [
            agent_pixel + forward * 13.0,
            agent_pixel - forward * 8.0 + right * 7.0,
            agent_pixel - forward * 8.0 - right * 7.0,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, triangle, (25, 25, 25), cv2.LINE_AA)

    cv2.putText(
        image,
        "Vector path view (no satellite imagery)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"grid {grid_spacing_m:g} m",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return image
