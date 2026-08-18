"""H100 renderer for raw HUGE-Bench 3D Gaussian assets.

This module intentionally imports torch, plyfile, scipy, and gsplat lazily so
the base SatNav environment remains free of heavy rendering dependencies.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from applications.huge3dgs_renderer.rendering import RendererBackend
from applications.huge3dgs_renderer.scene_manifest import (
    SceneManifest,
    SceneManifestError,
)


def camera_matrices(
    east: float,
    north: float,
    up: float,
    yaw_degrees: float,
    hfov_degrees: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera and OpenCV intrinsics for a nadir ENU camera."""
    if width <= 0 or height <= 0 or not 0.0 < hfov_degrees < 180.0:
        raise ValueError("camera width/height must be positive and HFOV in (0, 180)")
    yaw = math.radians(float(yaw_degrees))
    heading = np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float32)
    image_down = -heading
    optical_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    camera_to_world = np.stack((right, image_down, optical_axis), axis=1)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world.T
    center = np.array([east, north, up], dtype=np.float32)
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ center
    focal = float(width) / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    intrinsics = np.array(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return world_to_camera, intrinsics


def _properties(vertex: np.ndarray) -> Tuple[str, ...]:
    return tuple(vertex.dtype.names or ())


def _columns(vertex: np.ndarray, names: Tuple[str, ...]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def _numbered(properties: Tuple[str, ...], prefix: str) -> Tuple[str, ...]:
    values = [name for name in properties if name.startswith(prefix)]
    return tuple(sorted(values, key=lambda name: int(name[len(prefix) :])))


def _rgb(vertex: np.ndarray, properties: Tuple[str, ...]) -> np.ndarray:
    alternatives = (("red", "green", "blue"), ("r", "g", "b"))
    for names in alternatives:
        if all(name in properties for name in names):
            colors = _columns(vertex, names)
            if float(np.nanmax(colors)) > 1.0:
                colors /= 255.0
            return np.clip(colors, 0.0, 1.0)
    raise SceneManifestError("fixed-Gaussian PLY requires RGB properties")


def decode_gaussian_ply(
    vertex: np.ndarray,
    representation: str,
    asset_to_enu: np.ndarray,
) -> Dict[str, Any]:
    """Decode common INRIA 3DGS or prepared RGB fixed-Gaussian PLY fields."""
    properties = _properties(vertex)
    if not all(name in properties for name in ("x", "y", "z")):
        raise SceneManifestError("Gaussian PLY lacks x/y/z")
    means = _columns(vertex, ("x", "y", "z"))
    transform = np.asarray(asset_to_enu, dtype=np.float32)
    if transform.shape != (4, 4):
        raise SceneManifestError("asset_to_enu must be 4x4")
    linear = transform[:3, :3]
    if not np.allclose(linear, np.eye(3), atol=1e-6):
        raise SceneManifestError(
            "renderer requires Gaussian assets pre-oriented to ENU; "
            "asset_to_enu may contain translation only"
        )
    means = means @ linear.T + transform[:3, 3]
    raw_3dgs = str(representation).lower() == "raw_3dgs"

    scale_names = _numbered(properties, "scale_")
    if len(scale_names) >= 3:
        scales = _columns(vertex, scale_names[:3])
        if raw_3dgs:
            scales = np.exp(scales)
    elif "radius" in properties:
        radius = np.asarray(vertex["radius"], dtype=np.float32).reshape(-1, 1)
        scales = np.repeat(radius, 3, axis=1)
    else:
        raise SceneManifestError("Gaussian PLY requires scale_0..2 or radius")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise SceneManifestError("Gaussian scales must be finite and positive")

    rotation_names = _numbered(properties, "rot_")
    if len(rotation_names) >= 4:
        quats = _columns(vertex, rotation_names[:4])
    else:
        quats = np.zeros((len(vertex), 4), dtype=np.float32)
        quats[:, 0] = 1.0

    if "opacity" in properties:
        opacities = np.asarray(vertex["opacity"], dtype=np.float32)
        if raw_3dgs:
            opacities = 1.0 / (1.0 + np.exp(-opacities))
    else:
        opacities = np.full(len(vertex), 0.95, dtype=np.float32)
    opacities = np.clip(opacities, 0.0, 1.0)

    dc_names = _numbered(properties, "f_dc_")
    rest_names = _numbered(properties, "f_rest_")
    if len(dc_names) >= 3:
        dc = _columns(vertex, dc_names[:3])[:, None, :]
        if rest_names:
            if len(rest_names) % 3:
                raise SceneManifestError("3DGS f_rest property count must be divisible by 3")
            rest = _columns(vertex, rest_names)
            coefficient_count = len(rest_names) // 3
            rest = rest.reshape(len(vertex), 3, coefficient_count).transpose(0, 2, 1)
            colors = np.concatenate((dc, rest), axis=1)
        else:
            colors = dc
        degree = int(round(math.sqrt(colors.shape[1]) - 1))
        if (degree + 1) ** 2 > colors.shape[1]:
            raise SceneManifestError("3DGS SH coefficient count is invalid")
        sh_degree: Optional[int] = degree
    else:
        colors = _rgb(vertex, properties)
        sh_degree = None

    return {
        "means": np.ascontiguousarray(means, dtype=np.float32),
        "quats": np.ascontiguousarray(quats, dtype=np.float32),
        "scales": np.ascontiguousarray(scales, dtype=np.float32),
        "opacities": np.ascontiguousarray(opacities, dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.float32),
        "sh_degree": sh_degree,
    }


class MeshSurface:
    """Small aligned OBJ index used only for geometry-clearance queries.

    HUGE meshes are simplified (tens of thousands of vertices), so keeping a
    CPU index is substantially cheaper than duplicating a KD-tree for tens of
    millions of Gaussian centers. The mesh never controls camera altitude.
    """

    def __init__(self, path: Path, asset_to_enu: np.ndarray, kd_tree_class: Any):
        vertices = []
        faces = []
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if line.startswith("v "):
                    fields = line.split()
                    if len(fields) < 4:
                        raise SceneManifestError(f"malformed OBJ vertex: {path}")
                    vertices.append(tuple(float(value) for value in fields[1:4]))
                elif line.startswith("f "):
                    indices = []
                    for value in line.split()[1:]:
                        raw = value.split("/", 1)[0]
                        index = int(raw)
                        if index == 0:
                            raise SceneManifestError(f"OBJ indices are one-based: {path}")
                        indices.append(index - 1 if index > 0 else len(vertices) + index)
                    for offset in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[offset], indices[offset + 1]))
        if not vertices or not faces:
            raise SceneManifestError(f"aligned OBJ has no vertices/faces: {path}")
        points = np.asarray(vertices, dtype=np.float32)
        transform = np.asarray(asset_to_enu, dtype=np.float32)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        triangles = np.asarray(faces, dtype=np.int64)
        if np.any(triangles < 0) or np.any(triangles >= len(points)):
            raise SceneManifestError(f"OBJ face index is out of range: {path}")
        self.vertices = points
        self.faces = triangles
        self.face_xyz_index = kd_tree_class(points[triangles].mean(axis=1))

    def clearance(self, position: np.ndarray) -> float:
        """Approximate global search with exact point-to-candidate triangles."""
        count = min(256, len(self.faces))
        _, raw_indices = self.face_xyz_index.query(position, k=count)
        indices = np.atleast_1d(raw_indices).astype(np.int64, copy=False)
        triangle = self.vertices[self.faces[indices]].astype(np.float64, copy=False)
        point = np.asarray(position, dtype=np.float64)
        a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_squared = np.einsum("ij,ij->i", normal, normal)
        valid_plane = normal_squared > 1e-18
        signed_scale = np.zeros(len(triangle), dtype=np.float64)
        signed_scale[valid_plane] = np.einsum(
            "ij,ij->i", a[valid_plane] - point, normal[valid_plane]
        ) / normal_squared[valid_plane]
        projection = point + signed_scale[:, None] * normal

        ap = projection - a
        d00 = np.einsum("ij,ij->i", ab, ab)
        d01 = np.einsum("ij,ij->i", ab, ac)
        d11 = np.einsum("ij,ij->i", ac, ac)
        d20 = np.einsum("ij,ij->i", ap, ab)
        d21 = np.einsum("ij,ij->i", ap, ac)
        denominator = d00 * d11 - d01 * d01
        valid_barycentric = np.abs(denominator) > 1e-18
        weight_b = np.zeros(len(triangle), dtype=np.float64)
        weight_c = np.zeros(len(triangle), dtype=np.float64)
        weight_b[valid_barycentric] = (
            d11[valid_barycentric] * d20[valid_barycentric]
            - d01[valid_barycentric] * d21[valid_barycentric]
        ) / denominator[valid_barycentric]
        weight_c[valid_barycentric] = (
            d00[valid_barycentric] * d21[valid_barycentric]
            - d01[valid_barycentric] * d20[valid_barycentric]
        ) / denominator[valid_barycentric]
        inside = (
            valid_plane
            & valid_barycentric
            & (weight_b >= -1e-8)
            & (weight_c >= -1e-8)
            & (weight_b + weight_c <= 1.0 + 1e-8)
        )
        distances = np.full(len(triangle), np.inf, dtype=np.float64)
        distances[inside] = np.linalg.norm(projection[inside] - point, axis=1)

        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            edge_squared = np.einsum("ij,ij->i", edge, edge)
            factor = np.zeros(len(triangle), dtype=np.float64)
            nonzero = edge_squared > 1e-18
            factor[nonzero] = np.einsum(
                "ij,ij->i", point - start[nonzero], edge[nonzero]
            ) / edge_squared[nonzero]
            factor = np.clip(factor, 0.0, 1.0)
            closest = start + factor[:, None] * edge
            distances = np.minimum(distances, np.linalg.norm(closest - point, axis=1))
        return float(np.min(distances))


class GsplatRendererBackend(RendererBackend):
    """Persistent gsplat backend for normalized SatNav scene manifests."""

    def __init__(self, manifest: SceneManifest, config: Optional[Any] = None):
        try:
            import torch
            from gsplat.rendering import rasterization
            from plyfile import PlyData
            from scipy.spatial import cKDTree
        except ImportError as error:
            raise RuntimeError(
                "GsplatRendererBackend requires torch, gsplat, plyfile, and scipy"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("GsplatRendererBackend requires a CUDA GPU")
        self.manifest = manifest
        self.config = config or {}
        self.torch = torch
        self.rasterization = rasterization
        self.PlyData = PlyData
        self.KDTree = cKDTree
        self.device = str(self._setting("DEVICE", "cuda"))
        self.near_plane = float(self._setting("NEAR_PLANE", 0.1))
        self.far_plane = float(self._setting("FAR_PLANE", 10000.0))
        self.radius_clip = float(self._setting("RADIUS_CLIP", 0.0))
        self.rasterize_mode = str(self._setting("RASTERIZE_MODE", "antialiased"))
        self.valid_alpha_threshold = float(
            self._setting("VALID_ALPHA_THRESHOLD", 0.5)
        )
        self.max_invalid_fraction = float(
            self._setting("MAX_INVALID_FRACTION", 0.02)
        )
        raw_fixed_height = self._setting("FIXED_HEIGHT_M", None)
        if raw_fixed_height is None or str(raw_fixed_height).strip() == "":
            raise ValueError("renderer FIXED_HEIGHT_M is required")
        self.fixed_height_m = float(raw_fixed_height)
        if not 0.0 <= self.valid_alpha_threshold <= 1.0:
            raise ValueError("VALID_ALPHA_THRESHOLD must be in [0, 1]")
        if not 0.0 <= self.max_invalid_fraction <= 1.0:
            raise ValueError("MAX_INVALID_FRACTION must be in [0, 1]")
        if (
            not math.isfinite(self.fixed_height_m) or self.fixed_height_m <= 0.0
        ):
            raise ValueError("FIXED_HEIGHT_M must be finite and positive")
        self.scene_id: Optional[str] = None
        self.record: Optional[Dict[str, Any]] = None
        self.tensors: Dict[str, Any] = {}
        self.sh_degree: Optional[int] = None
        self.clearance_index = None
        self.mesh_surface: Optional[MeshSurface] = None

    def _setting(self, name: str, default: Any) -> Any:
        if isinstance(self.config, Mapping):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    def _asset(self, record: Mapping[str, Any]) -> Tuple[Path, str]:
        prepared = record.get("prepared_asset") or {}
        raw_path = prepared.get("path") or (record.get("asset") or {}).get("path")
        if not raw_path:
            raise SceneManifestError(f"scene {record.get('scene_id')} has no renderer asset")
        path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
        if not path.is_absolute():
            path = self.manifest.base_dir / path
        path = path.resolve()
        if path.is_dir():
            preferred = tuple(path.rglob("point_cloud_utm50.ply"))
            candidates = preferred or tuple(path.rglob("*.ply"))
            if len(candidates) != 1:
                raise SceneManifestError(
                    f"scene asset directory must contain exactly one renderer PLY: {path}"
                )
            path = candidates[0]
        if not path.is_file():
            raise FileNotFoundError(path)
        representation = str(
            prepared.get("representation")
            or ("raw_3dgs" if (record.get("asset") or {}).get("type") == "3dgs" else "fixed_gaussian")
        )
        return path, representation

    def _mesh_asset(self, record: Mapping[str, Any]) -> Optional[Path]:
        prepared = record.get("prepared_asset") or {}
        mesh = prepared.get("mesh") or {}
        raw_path = mesh.get("path") or (record.get("artifacts") or {}).get("mesh")
        if not raw_path:
            return None
        path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
        if not path.is_absolute():
            path = self.manifest.base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_scene(self, scene_id: str) -> None:
        if scene_id == self.scene_id:
            return
        self.close()
        record = self.manifest.get(scene_id)
        path, representation = self._asset(record)
        ply = self.PlyData.read(str(path))
        if "vertex" not in ply:
            raise SceneManifestError(f"Gaussian PLY has no vertex element: {path}")
        frame = record.get("coordinate_frame") or {}
        transform = frame.get("asset_to_enu") or np.eye(4, dtype=np.float32)
        parameters = decode_gaussian_ply(ply["vertex"].data, representation, transform)
        mesh_path = self._mesh_asset(record)
        if mesh_path is not None:
            self.mesh_surface = MeshSurface(mesh_path, np.asarray(transform), self.KDTree)
            self.clearance_index = None
        else:
            # Prepared RGB surfels may not have a mesh.  Retain the historical
            # center-based fallback for those assets; formal HUGE assets are
            # required to provide their aligned mesh.
            self.mesh_surface = None
            self.clearance_index = self.KDTree(parameters["means"])
        self.tensors = {
            name: self.torch.from_numpy(parameters[name]).to(self.device)
            for name in ("means", "quats", "scales", "opacities", "colors")
        }
        self.sh_degree = parameters["sh_degree"]
        self.record = record
        self.scene_id = str(scene_id)

    def _ensure_request_assets(self, request: Mapping[str, Any]) -> None:
        self.load_scene(str(request["scene_id"]))

    def _fixed_camera_up(self, request: Mapping[str, Any]) -> float:
        """Return the scene-normalized fixed flight height.

        The scene manifest translates its frozen ground datum to ENU up=0.
        Episode altitude is therefore an absolute flight-plane height, not a
        per-position offset above the highest mesh surface.
        """
        if self.record is None:
            raise RuntimeError("renderer scene is not loaded")
        camera = self.record.get("camera") or {}
        altitude_mode = str(camera.get("altitude_mode", "")).lower()
        if altitude_mode != "fixed_per_episode":
            raise SceneManifestError(
                f"scene {self.scene_id} camera.altitude_mode must be "
                "'fixed_per_episode'"
            )
        requested_height = float(request["agl"])
        if not math.isfinite(requested_height) or requested_height <= 0.0:
            raise ValueError("requested fixed flight height must be positive")
        if not math.isclose(
            requested_height,
            self.fixed_height_m,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"renderer requires configured fixed height {self.fixed_height_m} m, "
                f"got {requested_height} m"
            )
        return requested_height

    def _request_tensors(self, request: Mapping[str, Any]):
        width = int(request["width"])
        height = int(request["height"])
        east = float(request["east"])
        north = float(request["north"])
        projection = str(request.get("projection", "pinhole")).lower()
        if projection != "pinhole":
            raise ValueError("HUGE 3DGS episode rendering supports pinhole cameras only")
        view, intrinsics = camera_matrices(
            east=east,
            north=north,
            up=self._fixed_camera_up(request),
            yaw_degrees=float(request["yaw"]),
            hfov_degrees=float(request["hfov"]),
            width=width,
            height=height,
        )
        return (
            self.torch.from_numpy(view).to(self.device)[None],
            self.torch.from_numpy(intrinsics).to(self.device)[None],
            width,
            height,
            projection,
        )

    def render_products(self, request: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        self._ensure_request_assets(request)
        viewmats, intrinsics, width, height, camera_model = self._request_tensors(request)
        with self.torch.inference_mode():
            rendered, alpha, _ = self.rasterization(
                self.tensors["means"],
                self.tensors["quats"],
                self.tensors["scales"],
                self.tensors["opacities"],
                self.tensors["colors"],
                viewmats,
                intrinsics,
                width,
                height,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                radius_clip=self.radius_clip,
                sh_degree=self.sh_degree,
                packed=True,
                render_mode="RGB+ED",
                rasterize_mode=self.rasterize_mode,
                camera_model=camera_model,
            )
        rendered = rendered[0]
        alpha = alpha[0, ..., 0]
        rgb = (
            rendered[..., :3].clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
        )
        depth = rendered[..., 3].float().cpu().numpy()
        valid = alpha.float().cpu().numpy()
        return {
            "rgb": rgb,
            "depth": depth,
            "alpha": valid,
            "gaussian_count": int(self.tensors["means"].shape[0]),
            "scene_gaussian_count": int(self.tensors["means"].shape[0]),
        }

    def render(self, request: Mapping[str, Any]) -> np.ndarray:
        return self.render_products(request)["rgb"]

    def render_frame(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        products = self.render_products(request)
        alpha = np.asarray(products["alpha"], dtype=np.float32)
        invalid_fraction = float((alpha < self.valid_alpha_threshold).mean())
        return {
            "rgb": products["rgb"],
            "invalid_fraction": invalid_fraction,
            "render_complete": invalid_fraction <= self.max_invalid_fraction,
            "valid_alpha_threshold": self.valid_alpha_threshold,
            "camera_height_mode": "fixed_per_episode",
            "camera_up_m": self._fixed_camera_up(request),
            "configured_flight_height_m": self.fixed_height_m,
        }

    def is_navigable(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure_request_assets(request)
        east = float(request["east"])
        north = float(request["north"])
        position = np.array(
            [east, north, self._fixed_camera_up(request)],
            dtype=np.float32,
        )
        if self.mesh_surface is not None:
            clearance = self.mesh_surface.clearance(position)
        else:
            assert self.clearance_index is not None
            clearance, _ = self.clearance_index.query(position, k=1)
        required = float(request.get("min_geometry_clearance_m", 10.0))
        return {
            "navigable": bool(float(clearance) >= required),
            "min_geometry_clearance_m": float(clearance),
        }

    def close(self) -> None:
        self.tensors = {}
        self.clearance_index = None
        self.mesh_surface = None
        self.record = None
        self.scene_id = None
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


__all__ = [
    "GsplatRendererBackend",
    "MeshSurface",
    "camera_matrices",
    "decode_gaussian_ply",
]
