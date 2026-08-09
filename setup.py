#!/usr/bin/env python3
"""Setup script for SatNav project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satnav",
    version="0.1.0",
    author="SatNav Team",
    description="A testing platform for Vision-and-Language Navigation in continuous space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eku127/SatNav",
    license="MIT",
    project_urls={
        "Source": "https://github.com/Eku127/SatNav",
        "Issues": "https://github.com/Eku127/SatNav/issues",
    },
    packages=find_packages(),
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
    install_requires=requirements,
)
