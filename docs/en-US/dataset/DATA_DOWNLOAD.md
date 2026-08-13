# Download SatNav Episodes

This guide covers downloading, configuring, and validating SatNav episode
metadata. Satellite scenes are prepared separately; see
[Satellite Scene Download](../applications/MAP_DOWNLOAD.md).

## 1. Dataset

| Name | Data | Source | Contents |
| --- | --- | --- | --- |
| SatNav-Episodes-v0.1 | Episode metadata | [Kaggle](https://www.kaggle.com/datasets/07af1ab653c3d8d0518027b41d05dfa677d6a414131b27c4b024b887d74c6a68) | Instructions, starts, goals, waypoints, reference paths, and splits |

The dataset contains episode JSON files, train/evaluation splits,
`scenes_list.yaml`, and documentation. The scene list records geographic
bounds only; it contains no imagery.

The release does not include raw OpenStreetMap data, satellite imagery, map
tiles, GeoTIFF scenes, or offline training trajectories.

## 2. Download and configure

Download the archive from Kaggle, extract it, and name the root
`SatNav-v0.1`. The important layout is:

```text
data/satnav_datasets/SatNav-v0.1/
├── episodes/
│   ├── train/all_episodes.json
│   └── eval/
│       ├── val_seen/all_episodes.json
│       └── val_unseen/all_episodes.json
├── scenes_list.yaml
└── SHA256SUMS
```

From the SatNav root, set the dataset path. Replace it with an absolute path if
the data lives on another volume:

```bash
export SATNAV_DATA_ROOT="$PWD/data/satnav_datasets/SatNav-v0.1"
```

## 3. Validate integrity

Run the published checksum file from the dataset root:

```bash
cd "$SATNAV_DATA_ROOT"
sha256sum -c SHA256SUMS
cd -
```

Expected episode counts:

| Split | Episodes |
| --- | ---: |
| train | 105,164 |
| val_seen | 4,574 |
| val_unseen | 8,756 |
| **Total** | **118,494** |

## 4. Data license

Episode JSON and benchmark parameter files are released under
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Dataset
documentation is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See
[DATA_LICENSE.md](../../../DATA_LICENSE.md) for the complete terms.

Satellite imagery is not part of SatNav-Episodes-v0.1 and is not sublicensed
by SatNav.

## 5. Next step

Follow [Satellite Scene Download](../applications/MAP_DOWNLOAD.md) to prepare
the GeoTIFF scenes required by SatSim.
