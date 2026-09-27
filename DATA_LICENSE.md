# SatNav License Summary

This repository separates the license for source code, documentation, benchmark
metadata, and third-party map content.

## Source Code

Source code, scripts, configuration files, and package metadata in this
repository are released under the MIT License. See `LICENSE`.

## Documentation

Documentation files, including `README.md`, files under `docs/`, and application
or example README files, are released under the Creative Commons Attribution
4.0 International License (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

## Episode Metadata

SatNav episode JSON files and related benchmark metadata are released under the
Open Database License 1.0 (ODbL-1.0), because they may contain information
derived from OpenStreetMap-based geospatial structure:

https://opendatacommons.org/licenses/odbl/1-0/

This includes example episode JSON files bundled for quick-start validation,
as well as separate SatNav-Episodes dataset releases.

## Satellite and Map Imagery

The bundled `applications/resources/map.tif` is generated from deterministic
coordinate formulas by `scripts/generate_synthetic_example_map.py` and is
released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Its GeoTIFF tags and `applications/resources/README.md` record the provenance.

The 59 benchmark GeoTIFF scenes are available through
[SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1); its dataset card describes access and use terms.
Users can also generate scenes with their own imagery-provider credentials
using the map downloader. Provider agreements govern imagery storage,
attribution, redistribution, and research use.

## Third-Party Notices

See `NOTICE` for OpenStreetMap attribution and third-party provider notices.
