"""Command-line entry point for the dedicated HUGE 3DGS renderer."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from applications.huge3dgs_renderer.rendering import serve_renderer


def _load_prebuilt_gsplat(extension_path: Path) -> None:
    """Install a validated gsplat extension without an implicit JIT build."""
    import gsplat

    spec = importlib.util.spec_from_file_location("gsplat_cuda", str(extension_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load gsplat extension from {extension_path}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["gsplat_cuda"] = extension
    gsplat.csrc = extension
    sys.modules["gsplat.csrc"] = extension


def _path(value: Any, config_path: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = config_path.parent / expanded
    return str(expanded.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m applications.huge3dgs_renderer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_server = subparsers.add_parser("render-server")
    render_server.add_argument("--config", required=True, type=Path)
    render_server.add_argument(
        "--gsplat-extension",
        type=Path,
        default=os.environ.get("SATNAV_GSPLAT_EXTENSION"),
        help="Optional prebuilt gsplat_cuda shared library.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = OmegaConf.load(config_path)
    simulator = config.get("SIMULATOR", {})
    renderer = simulator.get("RENDERER", {})
    dataset = config.get("DATASET", {})
    endpoint = renderer.get("ENDPOINT")
    manifest_path = dataset.get("SCENE_MANIFEST") or simulator.get(
        "SCENE_MANIFEST"
    )
    backend_class = renderer.get(
        "BACKEND_CLASS",
        "applications.huge3dgs_renderer.gsplat_backend.GsplatRendererBackend",
    )
    if not endpoint or not manifest_path:
        raise ValueError(
            "Renderer config requires SIMULATOR.RENDERER.ENDPOINT and "
            "DATASET.SCENE_MANIFEST"
        )

    if args.gsplat_extension is not None:
        extension_path = args.gsplat_extension.expanduser().resolve()
        if not extension_path.is_file():
            raise FileNotFoundError(extension_path)
        _load_prebuilt_gsplat(extension_path)

    serve_renderer(
        endpoint=_path(endpoint, config_path),
        authkey=str(renderer.get("AUTHKEY", "satnav-huge3dgs")),
        backend_class=str(backend_class),
        manifest_path=_path(manifest_path, config_path),
        config=renderer,
    )


if __name__ == "__main__":
    main()
