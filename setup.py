#!/usr/bin/env python3
"""Editable-install configuration for the SatNav source repository."""

from setuptools import find_packages, setup

CORE_REQUIREMENTS = [
    "numpy>=1.19.0",
    "omegaconf>=2.1.0",
    "pyyaml>=5.4.0",
    "attrs>=21.0.0",
    "rasterio>=1.3.0",
    "pyproj>=3.4.0",
    "opencv-python>=4.5.0",
    "scipy>=1.7.0",
    # The bundled public task enables TopDownMap, whose agent sprite is loaded
    # through imageio during Env.reset().  Keep the documented core quickstart
    # runnable without requiring the broader applications extra.
    "imageio>=2.9.0",
]

CLASSIC_REQUIREMENTS = [
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "pillow>=8.0.0",
    "tqdm>=4.60.0",
    "imageio>=2.9.0",
    "imageio-ffmpeg>=0.4.0",
    "swanlab[dashboard]>=0.7.13",
]

# External VLMs intentionally own their model-specific environments.  This
# extra contains only lightweight dependencies shared by SatNav adapters.
VLM_REQUIREMENTS = [
    "pillow>=8.0.0",
    "tqdm>=4.60.0",
]

APPLICATION_REQUIREMENTS = [
    "pillow>=8.0.0",
    "requests>=2.25.0",
    "tqdm>=4.60.0",
    "imageio>=2.9.0",
    "imageio-ffmpeg>=0.4.0",
]


def _unique(*requirement_groups):
    return list(dict.fromkeys(item for group in requirement_groups for item in group))


setup(
    name="satnav",
    version="0.1.0",
    description="A testing platform for Vision-and-Language Navigation in continuous space",
    packages=find_packages(exclude=("tests", "tests.*")),
    py_modules=["run"],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        # Core dependencies are installed by default; keep the explicit alias
        # for ``pip install -e '.[core]'`` from the source checkout.
        "core": [],
        "classic": CLASSIC_REQUIREMENTS,
        "vlm": VLM_REQUIREMENTS,
        "applications": APPLICATION_REQUIREMENTS,
        "all": _unique(
            CLASSIC_REQUIREMENTS,
            VLM_REQUIREMENTS,
            APPLICATION_REQUIREMENTS,
        ),
    },
)
