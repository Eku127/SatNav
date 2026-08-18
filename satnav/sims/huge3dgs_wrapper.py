#!/usr/bin/env python3
"""SatNav simulator wrapper for HUGE-Bench 3D Gaussian scenes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyproj import Transformer

from satnav.core.simulator import AgentState, Observations, Simulator
from satnav.core.utils import geodesic_distance_with_altitude
from satnav.sims.huge3dgs_ipc import Huge3DGSIPCClient
from satnav.task.actions import Action


_WEB_MERCATOR_RADIUS_M = 6378137.0


def _section(config: Union[DictConfig, Mapping[str, Any]], name: str) -> Any:
    if isinstance(config, DictConfig):
        return getattr(config, name, {})
    return config.get(name, {})


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, DictConfig):
        return getattr(config, name, default)
    if isinstance(config, Mapping):
        return config.get(name, default)
    return default


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"Scene adapter manifest must be a mapping: {path}")
    return value


@dataclass(frozen=True)
class Huge3DGSSceneAdapter:
    """Map one public synthetic-WGS84 scene onto one renderer ENU scene."""

    scene_id: str
    renderer_scene_id: str
    synthetic_origin_3857: Tuple[float, float]
    bounds_enu: Tuple[float, float, float, float]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Huge3DGSSceneAdapter":
        scene_id = str(record.get("scene_id", "")).strip()
        renderer_scene_id = str(record.get("renderer_scene_id", "")).strip()
        coordinate_frame = record.get("coordinate_frame", {})
        if isinstance(coordinate_frame, Mapping):
            frame_type = str(
                coordinate_frame.get("type", "synthetic_wgs84_web_mercator")
            ).lower()
            origin = record.get("synthetic_origin_3857") or coordinate_frame.get(
                "synthetic_origin_3857"
            )
        else:
            frame_type = str(coordinate_frame).lower()
            origin = record.get("synthetic_origin_3857")
        if frame_type != "synthetic_wgs84_web_mercator":
            raise ValueError(
                f"Scene {scene_id!r} uses unsupported coordinate frame {frame_type!r}"
            )
        bounds = record.get("bounds_enu") or record.get("bounds")
        if not scene_id or not renderer_scene_id:
            raise ValueError("Every scene adapter requires scene_id and renderer_scene_id")
        if not isinstance(origin, Sequence) or len(origin) != 2:
            raise ValueError(
                f"Scene {scene_id!r} requires synthetic_origin_3857=[x,y]"
            )
        if not isinstance(bounds, Sequence) or len(bounds) != 4:
            raise ValueError(f"Scene {scene_id!r} requires bounds_enu=[minE,minN,maxE,maxN]")
        origin_values = tuple(float(value) for value in origin)
        bounds_values = tuple(float(value) for value in bounds)
        values = origin_values + bounds_values
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Scene {scene_id!r} contains non-finite coordinates")
        if bounds_values[0] >= bounds_values[2] or bounds_values[1] >= bounds_values[3]:
            raise ValueError(f"Scene {scene_id!r} has invalid ENU bounds")
        return cls(
            scene_id=scene_id,
            renderer_scene_id=renderer_scene_id,
            synthetic_origin_3857=(origin_values[0], origin_values[1]),
            bounds_enu=(
                bounds_values[0],
                bounds_values[1],
                bounds_values[2],
                bounds_values[3],
            ),
        )


def load_scene_adapters(path: Union[str, os.PathLike]) -> Dict[str, Huge3DGSSceneAdapter]:
    """Load and strictly validate logical-scene coordinate adapters."""
    manifest_path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HUGE 3DGS scene adapter manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    records = manifest.get("scenes")
    if isinstance(records, Mapping):
        normalized = []
        for scene_id, raw_record in records.items():
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"Scene adapter {scene_id!r} must be a mapping")
            record = dict(raw_record)
            record.setdefault("scene_id", scene_id)
            normalized.append(record)
        records = normalized
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Scene adapter manifest must contain a 'scenes' list or mapping")
    adapters: Dict[str, Huge3DGSSceneAdapter] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Every scene adapter entry must be a mapping")
        adapter = Huge3DGSSceneAdapter.from_record(raw_record)
        if adapter.scene_id in adapters:
            raise ValueError(f"Duplicate HUGE 3DGS scene_id: {adapter.scene_id}")
        adapters[adapter.scene_id] = adapter
    if not adapters:
        raise ValueError("Scene adapter manifest contains no scenes")
    return adapters


class Huge3DGSSimWrapper(Simulator):
    """Expose HUGE 3DGS scenes through SatNav's legacy WGS84 contract.

    Episode coordinates remain synthetic longitude/latitude values. Internally,
    the wrapper maps Web-Mercator coordinates to the ENU frame used to author
    the corresponding HUGE scene and sends only lightweight render requests to
    a persistent same-host GPU process.
    """

    _wgs84_to_mercator = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    _mercator_to_wgs84 = Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )

    def __init__(
        self,
        config: Union[DictConfig, Mapping[str, Any]],
        scenes_dir: Optional[str] = None,
    ) -> None:
        del scenes_dir  # HUGE scenes are resolved only by the adapter manifest.
        self.config = config
        simulator_config = _section(config, "SIMULATOR")
        manifest_path = _value(simulator_config, "SCENE_ADAPTER_MANIFEST")
        if not manifest_path:
            raise ValueError("SIMULATOR.SCENE_ADAPTER_MANIFEST is required")
        self._scene_adapters = load_scene_adapters(str(manifest_path))

        self.forward_step_size = float(
            _value(simulator_config, "FORWARD_STEP_SIZE", 10.0)
        )
        self.turn_angle = float(_value(simulator_config, "TURN_ANGLE", 15.0))
        if self.forward_step_size <= 0.0 or self.turn_angle <= 0.0:
            raise ValueError("FORWARD_STEP_SIZE and TURN_ANGLE must be positive")

        sensor_config = _value(simulator_config, "RGB_SENSOR", {})
        self.rgb_width = int(_value(sensor_config, "WIDTH", 448))
        self.rgb_height = int(_value(sensor_config, "HEIGHT", 448))
        self.rgb_hfov = float(_value(sensor_config, "HFOV", 90.0))
        if self.rgb_width <= 0 or self.rgb_height <= 0:
            raise ValueError("RGB sensor width and height must be positive")
        if not 0.0 < self.rgb_hfov < 180.0:
            raise ValueError("RGB sensor HFOV must be in (0, 180)")

        renderer_config = _value(simulator_config, "RENDERER", {})
        endpoint = _value(renderer_config, "ENDPOINT")
        if not endpoint:
            raise ValueError("SIMULATOR.RENDERER.ENDPOINT is required")
        self._require_geometry_clearance = bool(
            _value(renderer_config, "REQUIRE_GEOMETRY_CLEARANCE", True)
        )
        self._require_render_complete = bool(
            _value(renderer_config, "REQUIRE_RENDER_COMPLETE", True)
        )
        self._min_geometry_clearance_m = float(
            _value(renderer_config, "MIN_GEOMETRY_CLEARANCE_M", 10.0)
        )
        self._max_invalid_fraction = float(
            _value(renderer_config, "MAX_INVALID_FRACTION", 0.02)
        )
        raw_fixed_height = _value(renderer_config, "FIXED_HEIGHT_M", None)
        if raw_fixed_height is None or str(raw_fixed_height).strip() == "":
            raise ValueError("SIMULATOR.RENDERER.FIXED_HEIGHT_M is required")
        self._configured_flight_height_m = float(raw_fixed_height)
        if self._min_geometry_clearance_m < 0.0:
            raise ValueError("MIN_GEOMETRY_CLEARANCE_M must be non-negative")
        if not 0.0 <= self._max_invalid_fraction <= 1.0:
            raise ValueError("MAX_INVALID_FRACTION must be in [0, 1]")
        if (
            not math.isfinite(self._configured_flight_height_m)
            or self._configured_flight_height_m <= 0.0
        ):
            raise ValueError("FIXED_HEIGHT_M must be finite and positive")
        self._renderer = Huge3DGSIPCClient(
            endpoint=str(endpoint),
            authkey=str(_value(renderer_config, "AUTHKEY", "satnav-huge3dgs")),
            reconnect_attempts=int(
                _value(renderer_config, "RECONNECT_ATTEMPTS", 3)
            ),
            reconnect_delay_s=float(
                _value(renderer_config, "RECONNECT_DELAY_S", 0.1)
            ),
        )
        if bool(_value(renderer_config, "PING_ON_INIT", False)):
            self._renderer.ping()

        self._scene_id: Optional[str] = None
        self._scene: Optional[Huge3DGSSceneAdapter] = None
        self._position_enu: Optional[Tuple[float, float, float]] = None
        self._rotation: Optional[float] = None
        self._flight_height_m: Optional[float] = self._configured_flight_height_m
        self._cached_render_key: Optional[Tuple[Any, ...]] = None
        self._cached_rgb: Optional[np.ndarray] = None
        self._last_render_audit: Dict[str, Any] = {}

    @staticmethod
    def _action_name(action: Union[int, str, Mapping[str, Any]]) -> str:
        if isinstance(action, int):
            return Action.get_action_from_index(action)
        if isinstance(action, Mapping):
            if "action" not in action:
                raise ValueError("Action mapping must contain an 'action' key")
            action = action["action"]
        action_name = str(action)
        if not Action.is_valid_action(action_name):
            raise ValueError(
                f"Invalid action {action_name!r}; expected one of {Action.ALL_ACTIONS}"
            )
        return action_name

    def reset(self, scene_id: str) -> Observations:
        logical_scene_id = str(scene_id)
        try:
            scene = self._scene_adapters[logical_scene_id]
        except KeyError as error:
            available = ", ".join(sorted(self._scene_adapters))
            raise KeyError(
                f"Unknown HUGE 3DGS scene {logical_scene_id!r}; available: {available}"
            ) from error
        self._scene_id = logical_scene_id
        self._scene = scene
        self._position_enu = None
        self._rotation = None
        self._flight_height_m = self._configured_flight_height_m
        self._invalidate_render_cache()
        return {}

    def _require_scene(self) -> Huge3DGSSceneAdapter:
        if self._scene is None:
            raise RuntimeError("reset(scene_id) must be called first")
        return self._scene

    def _wgs84_position_to_enu(
        self, position: Sequence[float]
    ) -> Tuple[float, float, float]:
        scene = self._require_scene()
        if len(position) != 3:
            raise ValueError("Position must contain [longitude, latitude, altitude]")
        values = tuple(float(value) for value in position)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Position values must be finite")
        x, y = self._wgs84_to_mercator.transform(values[0], values[1])
        return (
            float(x) - scene.synthetic_origin_3857[0],
            float(y) - scene.synthetic_origin_3857[1],
            values[2],
        )

    def _enu_position_to_wgs84(
        self, position: Sequence[float]
    ) -> Tuple[float, float, float]:
        scene = self._require_scene()
        x = float(position[0]) + scene.synthetic_origin_3857[0]
        y = float(position[1]) + scene.synthetic_origin_3857[1]
        longitude, latitude = self._mercator_to_wgs84.transform(x, y)
        return float(longitude), float(latitude), float(position[2])

    def _center_inside_bounds(self, position_enu: Sequence[float]) -> bool:
        scene = self._require_scene()
        east, north, agl = (float(value) for value in position_enu)
        min_east, min_north, max_east, max_north = scene.bounds_enu
        return (
            agl > 0.0
            and (
                self._flight_height_m is None
                or math.isclose(agl, self._flight_height_m, abs_tol=1e-6)
            )
            and min_east <= east <= max_east
            and min_north <= north <= max_north
        )

    def _enu_is_navigable(self, position_enu: Sequence[float]) -> bool:
        if not self._center_inside_bounds(position_enu):
            return False
        if not self._require_geometry_clearance:
            return True
        scene = self._require_scene()
        east, north, agl = (float(value) for value in position_enu)
        response = self._renderer.is_navigable(
            {
                "scene_id": scene.renderer_scene_id,
                "east": east,
                "north": north,
                "agl": agl,
                "min_geometry_clearance_m": self._min_geometry_clearance_m,
            }
        )
        clearance = response.get("min_geometry_clearance_m")
        if clearance is None or not math.isfinite(float(clearance)):
            return False
        return bool(response.get("navigable", False)) and (
            float(clearance) >= self._min_geometry_clearance_m
        )

    def set_agent_state(self, position: List[float], rotation: float) -> None:
        position_enu = self._wgs84_position_to_enu(position)
        requested_height = float(position_enu[2])
        if requested_height <= 0.0:
            raise ValueError("Episode flight height must be positive")
        if self._flight_height_m is None:
            self._flight_height_m = requested_height
        elif not math.isclose(
            requested_height,
            self._flight_height_m,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Episode altitude does not match the configured fixed flight "
                f"height: expected {self._flight_height_m} m, got "
                f"{requested_height} m"
            )
        if not self._enu_is_navigable(position_enu):
            raise ValueError(
                "Initial position is outside HUGE scene bounds or violates "
                f"geometry clearance: {list(position)}"
            )
        self._position_enu = position_enu
        self._rotation = float(rotation) % 360.0
        self._invalidate_render_cache()

    def get_agent_state(self) -> AgentState:
        if self._position_enu is None or self._rotation is None:
            raise RuntimeError("agent state has not been initialized")
        return AgentState(
            position=self._enu_position_to_wgs84(self._position_enu),
            rotation=self._rotation,
        )

    def is_navigable(self, position: Sequence[float]) -> bool:
        if self._scene is None:
            return False
        try:
            position_enu = self._wgs84_position_to_enu(position)
        except (TypeError, ValueError):
            return False
        return self._enu_is_navigable(position_enu)

    def _invalidate_render_cache(self) -> None:
        self._cached_render_key = None
        self._cached_rgb = None
        self._last_render_audit = {}

    def _render_request(self) -> Dict[str, Any]:
        if self._position_enu is None or self._rotation is None:
            raise RuntimeError("agent state has not been initialized")
        scene = self._require_scene()
        east, north, agl = self._position_enu
        return {
            "scene_id": scene.renderer_scene_id,
            "east": east,
            "north": north,
            "agl": agl,
            "yaw": self._rotation,
            "hfov": self.rgb_hfov,
            "width": self.rgb_width,
            "height": self.rgb_height,
        }

    @staticmethod
    def _request_key(request: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple(
            request[name]
            for name in (
                "scene_id",
                "east",
                "north",
                "agl",
                "yaw",
                "hfov",
                "width",
                "height",
            )
        )

    def get_observations(self) -> Observations:
        request = self._render_request()
        key = self._request_key(request)
        if key == self._cached_render_key and self._cached_rgb is not None:
            return {"rgb": self._cached_rgb.copy()}

        frame = dict(self._renderer.render_frame(request))
        rgb = np.asarray(frame.pop("rgb"))
        expected = (self.rgb_height, self.rgb_width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid HUGE renderer RGB product: shape={rgb.shape}, dtype={rgb.dtype}"
            )
        invalid_fraction = float(frame.get("invalid_fraction", 1.0))
        render_complete = bool(frame.get("render_complete", False))
        if not math.isfinite(invalid_fraction) or not 0.0 <= invalid_fraction <= 1.0:
            raise RuntimeError(
                f"Renderer returned invalid invalid_fraction={invalid_fraction}"
            )
        if self._require_render_complete and (
            not render_complete or invalid_fraction > self._max_invalid_fraction
        ):
            raise RuntimeError(
                "HUGE 3DGS frame failed completeness validation: "
                f"complete={render_complete}, invalid_fraction={invalid_fraction:.6f}, "
                f"maximum={self._max_invalid_fraction:.6f}, request={request}"
            )

        self._cached_render_key = key
        self._cached_rgb = np.ascontiguousarray(rgb).copy()
        self._last_render_audit = {
            **frame,
            "invalid_fraction": invalid_fraction,
            "render_complete": render_complete,
            "logical_scene_id": self._scene_id,
            "renderer_scene_id": request["scene_id"],
        }
        return {"rgb": self._cached_rgb.copy()}

    def step(self, action: Union[int, str, Dict[str, Any]]) -> Observations:
        if self._position_enu is None or self._rotation is None:
            raise RuntimeError("set_agent_state() must be called before step()")
        action_name = self._action_name(action)
        east, north, agl = self._position_enu
        rotation = self._rotation
        changed = False

        if action_name == Action.TURN_LEFT:
            rotation = (rotation - self.turn_angle) % 360.0
            changed = True
        elif action_name == Action.TURN_RIGHT:
            rotation = (rotation + self.turn_angle) % 360.0
            changed = True
        elif action_name == Action.MOVE_FORWARD:
            scene = self._require_scene()
            projected_north = north + scene.synthetic_origin_3857[1]
            mercator_scale = math.cosh(projected_north / _WEB_MERCATOR_RADIUS_M)
            projected_distance = self.forward_step_size * mercator_scale
            heading = math.radians(rotation)
            candidate = (
                east + math.sin(heading) * projected_distance,
                north + math.cos(heading) * projected_distance,
                agl,
            )
            if self._enu_is_navigable(candidate):
                east, north, agl = candidate
                changed = True

        if changed:
            self._position_enu = (east, north, agl)
            self._rotation = rotation
            self._invalidate_render_cache()
        return self.get_observations()

    def geodesic_distance(
        self, position_a: Sequence[float], position_b: Sequence[float]
    ) -> float:
        return geodesic_distance_with_altitude(position_a, position_b)

    @property
    def sensor_suite(self):
        return {
            "rgb": {
                "width": self.rgb_width,
                "height": self.rgb_height,
                "hfov": self.rgb_hfov,
            }
        }

    @property
    def action_space(self):
        return list(Action.ALL_ACTIONS)

    @property
    def scene_id(self) -> Optional[str]:
        return self._scene_id

    @property
    def renderer_scene_id(self) -> Optional[str]:
        return self._scene.renderer_scene_id if self._scene is not None else None

    @property
    def coordinate_frame(self) -> str:
        return "synthetic_wgs84_web_mercator"

    @property
    def last_render_audit(self) -> Mapping[str, Any]:
        return dict(self._last_render_audit)

    @property
    def flight_height_m(self) -> Optional[float]:
        """Absolute normalized-ENU height locked for the current episode."""
        return self._flight_height_m

    def close(self) -> None:
        self._renderer.close()
        self._scene_id = None
        self._scene = None
        self._position_enu = None
        self._rotation = None
        self._flight_height_m = self._configured_flight_height_m
        self._invalidate_render_cache()


__all__ = [
    "Huge3DGSSimWrapper",
    "Huge3DGSSceneAdapter",
    "load_scene_adapters",
]
