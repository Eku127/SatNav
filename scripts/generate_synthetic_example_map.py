#!/usr/bin/env python3
"""Generate the redistributable synthetic GeoTIFF used by public examples.

The renderer is deliberately procedural: it does not open, sample, transform,
or otherwise consume provider imagery.  The fixed georeferencing matches the
two bundled example episodes so existing public configs remain runnable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


WIDTH = 3441
HEIGHT = 3169
CRS = "EPSG:3857"
TRANSFORM = Affine(
    0.5971642834776122,
    0.0,
    12696898.933590123,
    0.0,
    -0.5971642834780654,
    2577851.0207853806,
)
BLOCK_SIZE = 256
GENERATOR_VERSION = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("applications/resources/map.tif"),
        help="GeoTIFF destination (default: applications/resources/map.tif)",
    )
    return parser


def _render_window(row_offset: int, column_offset: int, height: int, width: int):
    rows, columns = np.indices((height, width), dtype=np.int64)
    y = rows + row_offset
    x = columns + column_offset

    # Low-frequency deterministic texture makes motion visible without
    # resembling or deriving from any real aerial image.
    texture = ((x // 31) * 17 + (y // 29) * 23 + (x // 7) + (y // 11)) % 24
    red = 88 + texture
    green = 126 + texture // 2
    blue = 78 + texture // 3

    # Abstract blocks and roofs.
    cell_x = x % 160
    cell_y = y % 140
    buildings = (cell_x >= 28) & (cell_x <= 132) & (cell_y >= 24) & (cell_y <= 112)
    roof_tone = ((x // 160) * 37 + (y // 140) * 19) % 34
    red = np.where(buildings, 142 + roof_tone, red)
    green = np.where(buildings, 132 + roof_tone // 2, green)
    blue = np.where(buildings, 112 + roof_tone // 3, blue)

    # Synthetic park and pond shapes are simple analytic ellipses.
    park = ((x - WIDTH * 0.25) / (WIDTH * 0.15)) ** 2 + (
        (y - HEIGHT * 0.72) / (HEIGHT * 0.12)
    ) ** 2 <= 1.0
    red = np.where(park, 72 + texture // 3, red)
    green = np.where(park, 145 + texture // 2, green)
    blue = np.where(park, 76 + texture // 4, blue)

    water = ((x - WIDTH * 0.82) / (WIDTH * 0.18)) ** 2 + (
        (y - HEIGHT * 0.20) / (HEIGHT * 0.13)
    ) ** 2 <= 1.0
    wave = ((x + 2 * y) // 19) % 12
    red = np.where(water, 54 + wave // 3, red)
    green = np.where(water, 119 + wave, green)
    blue = np.where(water, 174 + wave, blue)

    # A regular street grid, central cross, and diagonal connector provide
    # stable visual structure for movement and viewer smoke tests.
    street = ((x % 320) < 15) | ((y % 280) < 15)
    arterial_x = np.abs(x - WIDTH // 2) <= 24
    arterial_y = np.abs(y - HEIGHT // 2) <= 24
    diagonal = np.abs(((x + 2 * y) % 1100) - 550) <= 10
    roads = street | arterial_x | arterial_y | diagonal
    road_texture = ((x + y) // 17) % 7
    red = np.where(roads, 112 + road_texture, red)
    green = np.where(roads, 116 + road_texture, green)
    blue = np.where(roads, 120 + road_texture, blue)

    center_lines = (
        (arterial_x & (np.abs(x - WIDTH // 2) <= 1))
        | (arterial_y & (np.abs(y - HEIGHT // 2) <= 1))
    ) & (((x + y) // 28) % 2 == 0)
    red = np.where(center_lines, 244, red)
    green = np.where(center_lines, 203, green)
    blue = np.where(center_lines, 76, blue)

    return np.stack((red, green, blue)).astype(np.uint8, copy=False)


def generate(output: Path) -> str:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        profile = {
            "driver": "GTiff",
            "width": WIDTH,
            "height": HEIGHT,
            "count": 3,
            "dtype": "uint8",
            "crs": CRS,
            "transform": TRANSFORM,
            "tiled": True,
            "blockxsize": BLOCK_SIZE,
            "blockysize": BLOCK_SIZE,
            "compress": "deflate",
            "predictor": 2,
            "zlevel": 9,
            "interleave": "pixel",
        }
        with rasterio.open(temporary_path, "w", **profile) as destination:
            for row_offset in range(0, HEIGHT, BLOCK_SIZE):
                height = min(BLOCK_SIZE, HEIGHT - row_offset)
                for column_offset in range(0, WIDTH, BLOCK_SIZE):
                    width = min(BLOCK_SIZE, WIDTH - column_offset)
                    destination.write(
                        _render_window(row_offset, column_offset, height, width),
                        window=Window(column_offset, row_offset, width, height),
                    )
            destination.update_tags(
                AREA_OR_POINT="Area",
                SATNAV_ASSET_KIND="procedural-synthetic-example",
                SATNAV_GENERATOR="scripts/generate_synthetic_example_map.py",
                SATNAV_GENERATOR_VERSION=GENERATOR_VERSION,
                SATNAV_LICENSE="CC0-1.0",
                SATNAV_PROVENANCE=(
                    "Generated entirely from deterministic coordinate formulas; "
                    "contains no third-party map or satellite imagery."
                ),
            )
        # ``mkstemp`` deliberately creates a private 0600 file.  Distribution
        # archives preserve that mode, which would make the bundled example
        # unreadable after a system-wide installation by another user.
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return digest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    digest = generate(args.output)
    print(f"wrote {args.output}: sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
