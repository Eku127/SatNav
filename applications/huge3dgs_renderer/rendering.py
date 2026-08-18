"""Framework-independent local IPC service for heavyweight render backends."""

from __future__ import annotations

import importlib
import threading
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Mapping, Optional, Type

import numpy as np

from applications.huge3dgs_renderer.scene_manifest import SceneManifest


def load_class(class_path: str, expected_base: Optional[Type] = None) -> Type:
    module_name, separator, class_name = str(class_path).rpartition(".")
    if not separator:
        raise ValueError(f"Expected fully-qualified class name, got: {class_path}")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(cls, type) or (
        expected_base is not None and not issubclass(cls, expected_base)
    ):
        raise TypeError(f"{class_path} does not implement {expected_base.__name__}")
    return cls


class RendererBackend:
    """Backend contract owned by the dedicated renderer process."""

    def load_scene(self, scene_id: str) -> None:
        raise NotImplementedError

    def render(self, request: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def render_frame(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "rgb": self.render(request),
            "invalid_fraction": 0.0,
            "render_complete": True,
        }

    def is_navigable(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def _rgb_bytes_response(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(frame)
    rgb = np.ascontiguousarray(payload.pop("rgb"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Renderer returned invalid RGB shape: {rgb.shape}")
    return {
        "ok": True,
        "rgb_bytes": rgb.tobytes(order="C"),
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        **payload,
    }


def serve_renderer(
    endpoint: str,
    authkey: str,
    backend_class: str,
    manifest_path: str,
    config: Any = None,
) -> None:
    """Serve one persistent renderer over an authenticated Unix socket."""
    endpoint_path = Path(endpoint)
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if endpoint_path.exists():
        endpoint_path.unlink()

    manifest = SceneManifest.load(manifest_path)
    backend_type = load_class(backend_class, RendererBackend)
    backend = backend_type(manifest, config)
    listener = Listener(str(endpoint_path), authkey=str(authkey).encode("utf-8"))
    backend_lock = threading.Lock()

    def handle_connection(connection) -> None:
        try:
            while True:
                try:
                    message = connection.recv()
                except EOFError:
                    break
                operation = message.get("operation")
                if operation == "ping":
                    connection.send({"ok": True})
                    continue
                if operation == "is_navigable":
                    try:
                        with backend_lock:
                            result = dict(backend.is_navigable(message["request"]))
                        connection.send({"ok": True, **result})
                    except Exception as error:
                        connection.send(
                            {"ok": False, "error": f"{type(error).__name__}: {error}"}
                        )
                    continue
                if operation != "render":
                    connection.send({"ok": False, "error": "unknown operation"})
                    continue
                try:
                    with backend_lock:
                        frame = backend.render_frame(message["request"])
                    connection.send(_rgb_bytes_response(frame))
                except Exception as error:
                    connection.send(
                        {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    )
        finally:
            connection.close()

    try:
        while True:
            connection = listener.accept()
            threading.Thread(
                target=handle_connection,
                args=(connection,),
                daemon=True,
            ).start()
    finally:
        backend.close()
        listener.close()
        if endpoint_path.exists():
            endpoint_path.unlink()


__all__ = ["RendererBackend", "load_class", "serve_renderer"]
