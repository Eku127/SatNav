#!/usr/bin/env python3
"""Lightweight local IPC client for the persistent HUGE 3DGS renderer."""

from __future__ import annotations

import os
import threading
import time
from multiprocessing.connection import Client
from typing import Any, Dict, Mapping, Optional

import numpy as np


class Huge3DGSIPCClient:
    """Exchange render requests with one same-host GPU renderer.

    The transport deliberately lives outside the renderer implementation so a
    normal SatNav environment never imports Torch, gsplat, or PLY readers.
    Connections are created lazily and re-established after a worker restart.
    """

    def __init__(
        self,
        endpoint: str,
        authkey: str = "satnav-huge3dgs",
        reconnect_attempts: int = 3,
        reconnect_delay_s: float = 0.1,
    ) -> None:
        expanded = os.path.expandvars(os.path.expanduser(str(endpoint)))
        if not expanded:
            raise ValueError("HUGE 3DGS renderer endpoint must not be empty")
        self.endpoint = expanded
        self.authkey = str(authkey).encode("utf-8")
        self.reconnect_attempts = max(1, int(reconnect_attempts))
        self.reconnect_delay_s = max(0.0, float(reconnect_delay_s))
        self._connection = None
        self._lock = threading.Lock()

    def _connect(self):
        if self._connection is None:
            self._connection = Client(self.endpoint, authkey=self.authkey)
        return self._connection

    def _close_unlocked(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _exchange(self, operation: str, request: Optional[Mapping[str, Any]] = None):
        message: Dict[str, Any] = {"operation": str(operation)}
        if request is not None:
            message["request"] = dict(request)

        last_error: Optional[BaseException] = None
        with self._lock:
            for attempt in range(self.reconnect_attempts):
                try:
                    connection = self._connect()
                    connection.send(message)
                    response = connection.recv()
                    if not isinstance(response, Mapping):
                        raise RuntimeError(
                            "HUGE 3DGS renderer returned a non-mapping response"
                        )
                    if not bool(response.get("ok", False)):
                        raise RuntimeError(
                            str(response.get("error", "HUGE 3DGS renderer failed"))
                        )
                    return dict(response)
                except RuntimeError:
                    # A valid renderer error is deterministic and should be
                    # surfaced without replaying an expensive render request.
                    raise
                except (BrokenPipeError, ConnectionError, EOFError, OSError) as error:
                    last_error = error
                    self._close_unlocked()
                    if attempt + 1 < self.reconnect_attempts:
                        time.sleep(self.reconnect_delay_s)

        raise RuntimeError(
            "Could not communicate with HUGE 3DGS renderer at "
            f"{self.endpoint!r} after {self.reconnect_attempts} attempt(s)"
        ) from last_error

    def ping(self) -> None:
        """Fail clearly when the configured local renderer is unavailable."""
        self._exchange("ping")

    def render_frame(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Return strict HWC uint8 RGB plus renderer audit metadata."""
        response = self._exchange("render", request)
        expected = (int(request["height"]), int(request["width"]), 3)
        if "rgb_bytes" in response:
            shape = tuple(int(value) for value in response.pop("rgb_shape", ()))
            dtype = str(response.pop("rgb_dtype", ""))
            if shape != expected or dtype != "uint8":
                raise RuntimeError(
                    "HUGE 3DGS renderer returned an invalid byte payload: "
                    f"shape={shape}, dtype={dtype}, expected={expected}/uint8"
                )
            rgb = np.frombuffer(response.pop("rgb_bytes"), dtype=np.uint8).reshape(
                expected
            )
        elif "rgb" in response:
            # Backward-compatible path for same-environment test renderers.
            rgb = np.asarray(response.pop("rgb"))
        else:
            raise RuntimeError(
                "HUGE 3DGS renderer response has neither 'rgb_bytes' nor 'rgb'"
            )
        if rgb.shape != expected:
            raise RuntimeError(
                f"HUGE 3DGS renderer returned RGB shape {rgb.shape}, expected {expected}"
            )
        if rgb.dtype != np.uint8:
            raise RuntimeError(
                f"HUGE 3DGS renderer returned RGB dtype {rgb.dtype}, expected uint8"
            )
        response.pop("ok", None)
        response["rgb"] = np.ascontiguousarray(rgb)
        return response

    def is_navigable(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Query geometry clearance without importing renderer dependencies."""
        response = self._exchange("is_navigable", request)
        response.pop("ok", None)
        return response

    def close(self) -> None:
        """Close the local connection; the shared renderer remains alive."""
        with self._lock:
            self._close_unlocked()


__all__ = ["Huge3DGSIPCClient"]
