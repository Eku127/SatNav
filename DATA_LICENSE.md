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

The bundled `applications/resources/map.tif` is an exception to the following
third-party-imagery warning because it is not imagery from a map provider. It
is generated entirely from deterministic coordinate formulas by
`scripts/generate_synthetic_example_map.py` and is dedicated to the public
domain under CC0 1.0. Its GeoTIFF tags and
`applications/resources/README.md` record this provenance.

https://creativecommons.org/publicdomain/zero/1.0/

Google Maps, Mapbox, and other third-party satellite or map imagery are not
included in the SatNav-Episodes dataset release and are not sublicensed by the
authors. The map downloader utilities are provided only to help users prepare
local scene assets with their own provider credentials. Users are responsible
for complying with the terms of the imagery provider they choose, including any
restrictions on caching, redistribution, and ML/AI use.

Do not assume imagery downloaded through SatNav utilities is covered by the MIT
license, CC BY 4.0, or ODbL-1.0.

## Third-Party Notices

See `NOTICE` for OpenStreetMap attribution and third-party provider notices.
