# Bundled synthetic example map

`map.tif` is a deterministic, procedurally generated RGB GeoTIFF used only by
the two public example episodes. It contains no Google, Mapbox, OpenStreetMap,
or other third-party map or satellite pixels. Its abstract blocks, streets,
park, and pond are generated from coordinate formulas in
[`scripts/generate_synthetic_example_map.py`](../../scripts/generate_synthetic_example_map.py).

The raster is dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Its fixed size,
CRS, transform, and bounds keep the repository examples stable; they do not
claim that the synthetic visual features represent the georeferenced place.

Regenerate it from the repository root:

```bash
python scripts/generate_synthetic_example_map.py
```

The generator writes atomically and records the generator path, version,
license, and synthetic provenance in the GeoTIFF tags.

When SatNav is installed as a wheel, load the example config through the
resource helper so episode and scene paths resolve inside `site-packages`:

```python
from applications.resources import load_example_task_config

config = load_example_task_config()
```
