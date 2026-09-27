# Download SatNav Episodes

This guide covers downloading, configuring, and validating SatNav episode
metadata. Satellite scenes are prepared separately; see
[Satellite Scene Download](../applications/MAP_DOWNLOAD.md).

## 1. Dataset

| Name | Data | Source | Contents |
| --- | --- | --- | --- |
| SatNav-Episodes-v0.1 | Episode metadata | [Hugging Face](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1) | Instructions, starts, goals, waypoints, reference paths, and splits |

The Episodes dataset contains episode JSON files, train/evaluation splits, scene bounds in `scenes_list.yaml`, and documentation. Obtain the companion GeoTIFFs by requesting [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1), or generate them using your own imagery API credentials. See [Satellite Scene Download](../applications/MAP_DOWNLOAD.md) for both workflows.

## 2. Download and configure

Download the release from Hugging Face into your local dataset directory:

```bash
pip install -U huggingface_hub
hf download Eku127/SatNav-Episodes-v0.1 --repo-type dataset \
  --local-dir data/satnav_datasets/SatNav-v0.1
```

For private access, first run `hf auth login` with an authorized account.
SatNav reads the JSON files under `episodes/`; Parquet files under `data/`
support the Hugging Face Dataset Viewer. The important layout is:

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

Companion scenes use the access terms on [SatNav-Scenes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1). API-generated scenes follow the corresponding imagery provider terms.

## 5. Next step

Follow [Satellite Scene Download](../applications/MAP_DOWNLOAD.md) to prepare
the GeoTIFF scenes required by SatSim.
