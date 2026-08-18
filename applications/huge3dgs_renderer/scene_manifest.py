"""Versioned scene manifests for non-GeoTIFF SatNav backends.

The manifest is deliberately framework independent.  It records logical scene
identity and reproducibility metadata while resolving local asset paths only at
runtime.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


class SceneManifestError(ValueError):
    """Raised when a scene manifest violates the public contract."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneManifestError(f"{label} must be an object")
    return value


class SceneManifest:
    """Validated collection of logical scene records."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        path: Optional[os.PathLike] = None,
    ) -> None:
        self.path = Path(path).resolve() if path is not None else None
        self.base_dir = self.path.parent if self.path is not None else Path.cwd()
        self.payload = dict(payload)
        self.schema_version = str(payload.get("schema_version", "1.0"))
        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise SceneManifestError("scene manifest must contain a non-empty 'scenes' list")

        self._scenes: Dict[str, Dict[str, Any]] = {}
        for index, raw_record in enumerate(raw_scenes):
            record = dict(_require_mapping(raw_record, f"scenes[{index}]"))
            scene_id = str(record.get("scene_id", "")).strip()
            if not scene_id:
                raise SceneManifestError(f"scenes[{index}].scene_id must be non-empty")
            if os.path.isabs(scene_id):
                raise SceneManifestError(
                    f"scene_id must be logical rather than an absolute path: {scene_id}"
                )
            if scene_id in self._scenes:
                raise SceneManifestError(f"duplicate scene_id: {scene_id}")

            source = _require_mapping(record.get("source"), f"scene {scene_id}.source")
            for field in ("name", "url", "license"):
                if not str(source.get(field, "")).strip():
                    raise SceneManifestError(
                        f"scene {scene_id}.source.{field} must be non-empty"
                    )

            asset = _require_mapping(record.get("asset"), f"scene {scene_id}.asset")
            asset_type = str(asset.get("type", "")).lower()
            if asset_type not in {"pointcloud", "3dgs", "orthomosaic"}:
                raise SceneManifestError(
                    f"scene {scene_id}.asset.type must be pointcloud, 3dgs, or orthomosaic"
                )
            if not str(asset.get("path", "")).strip():
                raise SceneManifestError(f"scene {scene_id}.asset.path must be non-empty")

            frame = _require_mapping(
                record.get("coordinate_frame"),
                f"scene {scene_id}.coordinate_frame",
            )
            frame_type = str(frame.get("type", "")).lower()
            if frame_type not in {"wgs84", "local_enu"}:
                raise SceneManifestError(
                    f"scene {scene_id}.coordinate_frame.type must be wgs84 or local_enu"
                )
            if frame_type == "local_enu":
                if str(frame.get("units", "")).lower() not in {"m", "meter", "meters"}:
                    raise SceneManifestError(
                        f"scene {scene_id} local_enu coordinates must use meters"
                    )
                if str(frame.get("axes", "")).lower() != "east_north_up":
                    raise SceneManifestError(
                        f"scene {scene_id} local_enu axes must be east_north_up"
                    )
                transform = frame.get("asset_to_enu")
                if transform is None or (
                    not isinstance(transform, list)
                    or len(transform) != 4
                    or any(not isinstance(row, list) or len(row) != 4 for row in transform)
                ):
                    raise SceneManifestError(
                        f"scene {scene_id}.coordinate_frame.asset_to_enu must be 4x4"
                    )
                vertical_datum = _require_mapping(
                    frame.get("vertical_datum"),
                    f"scene {scene_id}.coordinate_frame.vertical_datum",
                )
                try:
                    asset_ground_up = float(vertical_datum["asset_ground_up_m"])
                    translated_up = float(transform[2][3])
                except (KeyError, TypeError, ValueError) as error:
                    raise SceneManifestError(
                        f"scene {scene_id} requires numeric asset_ground_up_m"
                    ) from error
                if not math.isfinite(asset_ground_up) or not math.isclose(
                    translated_up, -asset_ground_up, abs_tol=1e-6
                ):
                    raise SceneManifestError(
                        f"scene {scene_id} must translate its frozen ground datum "
                        "to ENU up=0"
                    )

            bounds = record.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 4:
                raise SceneManifestError(
                    f"scene {scene_id}.bounds must be [min_east,min_north,max_east,max_north]"
                )
            try:
                min_e, min_n, max_e, max_n = (float(value) for value in bounds)
            except (TypeError, ValueError) as error:
                raise SceneManifestError(f"scene {scene_id}.bounds must be numeric") from error
            if not min_e < max_e or not min_n < max_n:
                raise SceneManifestError(f"scene {scene_id}.bounds are not ordered")

            camera = _require_mapping(record.get("camera"), f"scene {scene_id}.camera")
            if str(camera.get("altitude_mode", "")).lower() != "fixed_per_episode":
                raise SceneManifestError(
                    f"scene {scene_id}.camera.altitude_mode must be fixed_per_episode"
                )
            if str(camera.get("height_reference", "")).lower() != "normalized_enu_up":
                raise SceneManifestError(
                    f"scene {scene_id}.camera.height_reference must be "
                    "normalized_enu_up"
                )
            _require_mapping(record.get("renderer"), f"scene {scene_id}.renderer")
            self._scenes[scene_id] = record

    @classmethod
    def load(cls, path: os.PathLike) -> "SceneManifest":
        manifest_path = Path(os.path.expandvars(os.path.expanduser(str(path))))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Scene manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(_require_mapping(payload, "scene manifest"), manifest_path)

    @property
    def scene_ids(self) -> Iterable[str]:
        return tuple(self._scenes)

    def get(self, scene_id: str) -> Dict[str, Any]:
        try:
            return dict(self._scenes[str(scene_id)])
        except KeyError as error:
            raise SceneManifestError(
                f"scene_id '{scene_id}' is not present in {self.path or 'manifest'}"
            ) from error

    def resolve_path(self, scene_id: str, field: str = "path") -> str:
        record = self.get(scene_id)
        raw = str(record["asset"].get(field, "")).strip()
        if not raw:
            raise SceneManifestError(f"scene {scene_id}.asset.{field} is empty")
        expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
        if not expanded.is_absolute():
            expanded = self.base_dir / expanded
        return str(expanded.resolve())

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.payload)


__all__ = ["SceneManifest", "SceneManifestError"]
