#!/usr/bin/env python3
"""Setuptools configuration for SatNav."""

import os
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.egg_info import manifest_maker as _manifest_maker
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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


_RELEASE_ARTIFACT_DIRECTORIES = frozenset(
    {
        ".local",
        "checkpoints",
        "logs",
        "output",
        "rebuttal",
        "results",
        "runs",
        "runtime",
        "wandb",
    }
)


def _is_release_artifact(path):
    return bool(_RELEASE_ARTIFACT_DIRECTORIES.intersection(Path(path).parts))


def _normalize_tree_permissions(root):
    root = Path(root)
    if not root.is_symlink():
        root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


class SatNavBuildPy(_build_py):
    """Keep ignored machine-local overlays out of direct wheel builds."""

    def run(self):
        super().run()
        _normalize_tree_permissions(self.build_lib)

    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        return [path for path in files if not _is_release_artifact(path)]


class SatNavManifestMaker(_manifest_maker):
    """Apply ignored-directory release boundaries at arbitrary depth."""

    def prune_file_list(self):
        super().prune_file_list()
        self.filelist.files = [
            path for path in self.filelist.files if not _is_release_artifact(path)
        ]


class SatNavEggInfo(_egg_info):
    """Generate a sanitized SOURCES.txt used by both wheels and sdists."""

    def find_sources(self):
        manifest_filename = os.path.join(self.egg_info, "SOURCES.txt")
        manifest = SatNavManifestMaker(self.distribution)
        manifest.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
        manifest.manifest = manifest_filename
        manifest.run()
        self.filelist = manifest.filelist


class SatNavSdist(_sdist):
    """Normalize staged source permissions independently of the build umask."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        _normalize_tree_permissions(base_dir)


class SatNavBdistWheel(_bdist_wheel):
    """Normalize wheel staging after dist-info metadata has been written."""

    def write_wheelfile(self, wheelfile_base, generator="bdist_wheel"):
        super().write_wheelfile(wheelfile_base, generator)
        _normalize_tree_permissions(wheelfile_base)


setup(
    name="satnav",
    version="0.1.0",
    author="SatNav Team",
    description="A testing platform for Vision-and-Language Navigation in continuous space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eku127/SatNav",
    license="MIT",
    license_files=["LICENSE", "NOTICE", "DATA_LICENSE.md"],
    project_urls={
        "Source": "https://github.com/Eku127/SatNav",
        "Issues": "https://github.com/Eku127/SatNav/issues",
    },
    packages=[
        package
        for package in find_packages(exclude=("tests", "tests.*"))
        if not _is_release_artifact(Path(*package.split(".")))
    ],
    py_modules=["run"],
    include_package_data=True,
    package_data={
        "satnav.utils.assets.maps_topdown_agent_sprite": ["*.png"],
        "applications.resources": [
            "satnav_example_episodes.json",
            "*.yaml",
            "map.tif",
            "*.md",
        ],
        "applications.map_downloader": ["*.yaml"],
        "applications.map_downloader.google_downloader": [
            "*.yaml",
            "assets/*.png",
        ],
        "applications.map_downloader.mapbox_downloader": [
            "*.yaml",
            "assets/*.png",
        ],
        "applications.satsim_viewer": ["*.yaml"],
        "baselines.vlm.navila": ["configs/zero2.json"],
        "baselines.vlm.openfly": ["configs/zero2.json"],
        "baselines.vlm.streamvln": ["configs/zero2.json"],
        "baselines.vlm.uninavid": ["configs/zero1.json"],
    },
    # Defense in depth for wheels built directly from a developer checkout.
    # MANIFEST.in is the primary sdist boundary; this prevents package-data
    # discovery from reintroducing component-local state into a wheel.
    exclude_package_data={
        "": [
            ".local/*",
            "results/*",
            "tests/*",
            "local_*.yaml",
            "*.local.yaml",
        ]
    },
    cmdclass={
        "bdist_wheel": SatNavBdistWheel,
        "build_py": SatNavBuildPy,
        "egg_info": SatNavEggInfo,
        "sdist": SatNavSdist,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        # Core dependencies are installed by default; keep the explicit alias
        # so ``pip install satnav[core]`` is also a documented stable entrypoint.
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
