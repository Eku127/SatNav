# SatNav Data Format

This guide describes SatNav-v0.1 episodes, scenes, and offline trajectories.
The episode release contains navigation metadata only; users prepare satellite
scenes and offline trajectories separately.

See [tasks and metrics](../concepts/TASKS_AND_METRICS.md) for task families and path relationships, and [expert trajectories](../concepts/EXPERT_TRAJECTORIES.md) for image/action alignment.

## 1. Data components

| Data | Main files | Description |
| --- | --- | --- |
| Episodes | `episodes/**/all_episodes.json` | Instructions, starts, goals, waypoints, and reference paths |
| Scene list | `scenes_list.yaml` | Logical names and geographic bounds for 59 scenes; no imagery |
| GeoTIFF scenes | `<scene_id>.tif` | User-provided imagery used by SatSim |
| Offline trajectories | `annotations.json` and `images/` | Training data generated from train episodes and GeoTIFFs |

See [Episode Download](DATA_DOWNLOAD.md),
[Satellite Scene Download](../applications/MAP_DOWNLOAD.md), and
[Trajectory Generation](../applications/TRAJECTORY_GENERATION.md).

## 2. Splits

| Split | Episodes | Scenes | Purpose |
| --- | ---: | ---: | --- |
| train | 105,164 | 56 | Model training |
| val_seen | 4,574 | 56 | Scenes also present in train |
| val_unseen | 8,756 | 3 | Scenes not present in train |
| **Total** | **118,494** | **59** |  |

All split files use the same layout:

```text
episodes/
├── train/all_episodes.json
└── eval/
    ├── val_seen/all_episodes.json
    └── val_unseen/all_episodes.json
```

The top level contains an `episodes` list:

```json
{
  "episodes": [
    {
      "episode_id": 0,
      "trajectory_id": 0,
      "trajectory_type": "Road",
      "trajectory_subtype": "road",
      "scene_id": "Amsterdam-1",
      "start_position": [4.8784032, 52.3762329, 50],
      "start_rotation": 90.0,
      "goals": [{"position": [4.8810, 52.3770, 50]}],
      "instruction": {
        "instruction_text": "Continue along the road and stop at the junction.",
        "instruction_type": "natural"
      },
      "waypoints": [
        [4.8784032, 52.3762329, 50],
        [4.8810, 52.3770, 50]
      ],
      "reference_path": [
        [4.8784032, 52.3762329, 50],
        [4.8797, 52.3766, 50],
        [4.8810, 52.3770, 50]
      ],
      "aux_info": {}
    }
  ]
}
```

This example illustrates the Episode schema.

## 3. Episode fields

| Field | Type | Description |
| --- | --- | --- |
| `episode_id` | integer/string | Episode identifier within the split and scene |
| `trajectory_id` | integer/string | Route identifier; one route may have multiple instruction styles |
| `trajectory_type` | string | `Boundary`, `LandmarkSet`, or `Road` |
| `trajectory_subtype` | string | Finer-grained task type |
| `scene_id` | string | Stable logical scene name, such as `Amsterdam-1` |
| `start_position` | list | `[longitude, latitude, altitude]` |
| `start_rotation` | number | Initial heading in degrees |
| `goals` | list | Goal objects, usually containing one `position` |
| `instruction` | object | `instruction_text` and `instruction_type` |
| `waypoints` | list | Original sparse navigation points |
| `reference_path` | list | Dense path from start to goal |
| `aux_info` | object | Task-specific extension metadata |

Neither `episode_id` nor `trajectory_id` is globally unique. Persist and join
results with:

```text
<split>::<scene_id>::<episode_id>
```

The loader converts IDs to strings and losslessly preserves unknown fields in
episodes, instructions, and goals.

## 4. Coordinates and scenes

Positions use WGS84 coordinates:

```text
[longitude, latitude, altitude]
```

- longitude and latitude use decimal degrees;
- altitude is in meters and controls SatSim's observation footprint;
- heading uses north as `0°` and increases clockwise; `90°` points east.

`scene_id` is a logical scene name. SatNav uses `DATASET.SCENES_DIR` to resolve
it to a local file path:

```text
<SCENES_DIR>/Amsterdam-1.tif
```

The machine-local path is stored only in runtime `scene_path` and is excluded
from normal episode serialization and evaluation results.

## 5. Task types

| Task | `trajectory_type` | Common `trajectory_subtype` values |
| --- | --- | --- |
| Boundary | `Boundary` | `loop`, `arc`, `extended` |
| Landmark | `LandmarkSet` | `one_turn`, `two_turn` |
| Road | `Road` | `road`, `waterway`, `hybrid` |

`waypoints` are the high-level nodes used during task generation;
`reference_path` is the dense path used for navigation and evaluation. Expert
trajectory generation follows `reference_path`.

## 6. Offline trajectories

Offline trajectories are not included in SatNav-Episodes-v0.1. Generate them
with [Trajectory Generation](../applications/TRAJECTORY_GENERATION.md):

```text
trajectory_data/
├── annotations.json
├── summary.json
└── images/
    └── <scene_id>_satnav_<episode-index>/
        ├── .done
        ├── .annotation.json
        └── rgb/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

`annotations.json` is a training list. One item has this shape:

```json
{
  "id": 0,
  "trajectory_id": "0",
  "steps": 3,
  "video": "images/Amsterdam-1_satnav_000000",
  "instructions": ["Continue along the road and stop at the junction."],
  "actions": [-1, 1, 1, 0]
}
```

| Field | Description |
| --- | --- |
| `id` | Index in the source JSON `episodes` list; not `episode_id` |
| `trajectory_id` | Route identifier copied from the source episode |
| `steps` | Executable actions, equal to `len(actions) - 1` |
| `video` | Relative path to the RGB-frame directory |
| `instructions` | Instructions associated with the trajectory |
| `actions` | Discrete actions aligned with observations |

Action encoding:

| ID | Action |
| ---: | --- |
| -1 | `INIT`, the observation before the first action |
| 0 | `STOP` |
| 1 | `MOVE_FORWARD` |
| 2 | `TURN_LEFT` |
| 3 | `TURN_RIGHT` |

Production output uses 448 × 448 RGB JPEGs. Every trajectory satisfies:

```text
JPEG count = len(actions) = steps + 1
```

`summary.json`, `.done`, and `.annotation.json` support integrity checks and
resume. Do not edit them manually. Training reads public `annotations.json`
and the corresponding `images/` tree.

## 7. Load with `SatNavDataset`

```python
from omegaconf import OmegaConf
from satnav.dataset import SatNavDataset

config = OmegaConf.create({
    "DATA_PATH": "data/satnav_datasets/SatNav-v0.1/episodes/train/all_episodes.json",
    "SPLIT": "train",
    "SCENES_DIR": "data/satnav_datasets/scenes",
})

dataset = SatNavDataset(config)
episode = dataset.episodes[0]

print(len(dataset.episodes))
print(episode.episode_key)
print(episode.scene_id, episode.scene_path)
```

`SatNavDataset` supports `.json`, `.json.gz`, and paths containing a `{split}`
placeholder. Validate configured data before loading:

```bash
bash scripts/validation/data_validation.sh
```
