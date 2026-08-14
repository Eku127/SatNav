# SatNav Core API

This guide explains SatNav's runtime interface from a user perspective: how an
episode is loaded, rendered, executed, and evaluated, and which layer an
application or model should use.

If the environment or data is not ready, begin with
[Installation](../getting-started/INSTALLATION.md),
[Data Format](../dataset/DATASET_FORMAT.md), and
[Examples](../getting-started/EXAMPLES.md). To connect a new navigation model,
read [Model Integration](../development/MODEL_INTEGRATION.md).

## 1. Core objects

The runtime stack contains Dataset, Environment, Task, and Simulator:

```text
SatNavDataset
    └── VLNEpisode
           │
           v
          Env
           │
           └── VLNTask
                  ├── Sensors
                  ├── Measures
                  └── Simulator
                         └── SatSimWrapper -> SatSim -> GeoTIFF
```

| Object | Responsibility |
| --- | --- |
| `SatNavDataset` | Build `VLNEpisode` objects from JSON and resolve local scenes |
| `VLNEpisode` | Store instruction, start, goal, reference path, and logical scene ID |
| `Env` | Manage episode order, step counts, termination, and public interaction |
| `VLNTask` | Assemble sensors, actions, measures, and the simulator |
| `Simulator` | Define scene loading, state, actions, and observations |
| `SatSimWrapper` | Adapt built-in SatSim to the public `Simulator` interface |

Most users interact only with `Env`. Applications and baselines must not use
private members such as `env._dataset`, `env._task`, or `env._sim`.

## 2. Minimal example

The repository includes two synthetic episodes and a CC0 GeoTIFF:

```python
from applications.resources import load_example_task_config
from satnav.core.env import Env

config = load_example_task_config()
env = Env(config)

try:
    observation = env.reset()

    print(env.current_episode.episode_key)
    print(observation["rgb"].shape)
    print(observation["instruction"]["text"])
    print(observation["agent_pose"])

    observation, done, info = env.step("STOP")
    print(done, info["termination_reason"])
    print(env.get_metrics())
finally:
    env.close()
```

This performs a complete lifecycle: construct the environment, load an
episode, read its initial observation, act, inspect metrics, and release
resources.

> `Env` is not currently a context manager. Always call `close()` in
> `try/finally`. Repeated calls to `close()` are safe.

## 3. Configuration

`Env` accepts a complete `DictConfig` or Python `dict`. Load YAML with:

```python
from satnav.core.config import load_config

config = load_config("configs/satnav_eval_task.yaml")
```

The four main sections are:

```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500

SIMULATOR:
  TYPE: satsim
  FORWARD_STEP_SIZE: 10
  TURN_ANGLE: 15
  RGB_SENSOR:
    WIDTH: 448
    HEIGHT: 448
    HFOV: 90

TASK:
  TYPE: VLN
  SUCCESS_DISTANCE:
    DEFAULT: 10.0
    Boundary: 10.0
    LandmarkSet: 30.0
    Road: 10.0
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, ORACLE_SUCCESS, SPL, PATH_LENGTH]

DATASET:
  TYPE: SatNav
  SPLIT: val_seen
  DATA_PATH: path/to/all_episodes.json
  SCENES_DIR: path/to/scenes
```

| Setting | Meaning |
| --- | --- |
| `ENVIRONMENT.MAX_EPISODE_STEPS` | Maximum actions per episode |
| `SIMULATOR.TYPE` | Use `satsim` for the built-in simulator |
| `SIMULATOR.FORWARD_STEP_SIZE` | `MOVE_FORWARD` distance in meters |
| `SIMULATOR.TURN_ANGLE` | Left/right turn angle in degrees |
| `SIMULATOR.RGB_SENSOR` | RGB resolution and horizontal field of view |
| `TASK.SUCCESS_DISTANCE` | Default or trajectory-type-specific success radius |
| `TASK.POSSIBLE_ACTIONS` | Available discrete actions |
| `TASK.MEASUREMENTS` | Enabled metrics |
| `DATASET.SPLIT` | Split name used in the stable episode key |
| `DATASET.DATA_PATH` | Episode JSON/JSON.GZ; supports `{split}` |
| `DATASET.SCENES_DIR` | Directory containing `<scene_id>.tif` |

Use `configs/satnav_eval_task.yaml` for online evaluation. Its success radii
are 10 m for Boundary, 30 m for LandmarkSet, and 10 m for Road. The 3 m
LandmarkSet value in trajectory generation is an expert-waypoint tolerance,
not an online evaluation threshold.

Keep real paths out of public configs. Copy to ignored `configs/local_*.yaml`
or inject paths through local environment variables.

## 4. Construct `Env`

```python
Env(config, dataset=None, cycle=False)
```

| Argument | Meaning |
| --- | --- |
| `config` | Complete `ENVIRONMENT`, `SIMULATOR`, `TASK`, and `DATASET` config |
| `dataset` | Optional `SatNavDataset`; otherwise built from `config.DATASET` |
| `cycle` | Restart after exhaustion; defaults to `False` |

Keep `cycle=False` for evaluation so episodes are not repeated. Continuous
interaction may use:

```python
env = Env(config, cycle=True)
```

Or construct the dataset explicitly:

```python
from satnav.dataset import SatNavDataset
from satnav.core.env import Env

dataset = SatNavDataset(config.DATASET)
env = Env(config, dataset=dataset)
```

Even with an explicit dataset, simulator and task parameters come from the
complete config, and `DATASET.SCENES_DIR` resolves scenes.

## 5. Episode lifecycle

### 5.1 `reset()`

```python
observation = env.reset()
```

`reset()` gets the next episode and:

1. clears step count, termination state, and previous `info`;
2. resolves logical `scene_id` to a local GeoTIFF;
3. loads or reuses the scene;
4. sets `start_position` and `start_rotation`;
5. resets sensors and measures;
6. returns the initial observation.

With `cycle=False`, calling `reset()` after exhaustion raises `RuntimeError`.
An empty dataset is rejected during environment construction.

### 5.2 `reset_to_episode()`

```python
episode = env.episodes[0]
observation = env.reset_to_episode(episode)
```

This bypasses the dataset iterator and loads one `VLNEpisode`, which is useful
for debugging, fixed-episode visualization, and deterministic collection. It
sets `current_episode`, clears termination and `last_step_info`, and resets the
elapsed step count.

### 5.3 `step()`

```python
observation, done, info = env.step("MOVE_FORWARD")
```

Each step executes an action, updates measures, creates the next observation,
and increments elapsed steps. Episodes end only when:

- the agent executes `STOP`; or
- `MAX_EPISODE_STEPS` is reached.

Entering the success radius does not end an episode. The agent must choose
`STOP` at an appropriate location for `SUCCESS` to become 1. Calling `step()`
before `reset()` or after `done=True` raises `RuntimeError`.

### 5.4 `close()`

```python
env.close()
```

`close()` releases task and simulator resources. Built-in SatSim closes cached
Rasterio scenes and clears its scene cache. Repeated calls are safe.

## 6. Observations

Default `VLNTask` returns these after reset and every step:

| Key | Type | Shape/structure | Meaning |
| --- | --- | --- | --- |
| `rgb` | `numpy.ndarray` | `(H, W, 3)`, `uint8` | RGB rendered from the GeoTIFF at the agent pose |
| `instruction` | `dict` | `{"text": str}` | Current natural-language instruction |
| `agent_pose` | `numpy.ndarray` | `(4,)`, `float32` | Ego-frame pose relative to episode start |

RGB resolution comes from `SIMULATOR.RGB_SENSOR.WIDTH` and `HEIGHT`. Never
hard-code 224 or 448 in an application or adapter.

### 6.1 Agent pose

```text
[delta_forward_m, delta_right_m, sin(delta_heading), cos(delta_heading)]
```

- `delta_forward_m`: displacement along the initial heading;
- `delta_right_m`: displacement to the right of the initial heading;
- `delta_heading`: current heading relative to the initial heading.

Immediately after reset:

```text
[0.0, 0.0, 0.0, 1.0]
```

Sine/cosine avoids the discontinuity at -180°/180°. This pose is relative and
does not depend on absolute scene orientation.

### 6.2 `observation_space`

```python
print(env.observation_space)
```

This is a lightweight descriptive dictionary, not `gym.Space`. It emphasizes
RGB shape and instruction structure. Actual observations also contain
`agent_pose`; generic adapters should inspect keys returned by `reset()` or
`step()`.

## 7. Actions

| ID | Action | Effect |
| ---: | --- | --- |
| 0 | `STOP` | Do not move and terminate the episode |
| 1 | `MOVE_FORWARD` | Move `FORWARD_STEP_SIZE` meters along heading |
| 2 | `TURN_LEFT` | Decrease heading by `TURN_ANGLE` degrees |
| 3 | `TURN_RIGHT` | Increase heading by `TURN_ANGLE` degrees |

Equivalent inputs:

```python
env.step("MOVE_FORWARD")
env.step(1)
env.step({"action": "MOVE_FORWARD"})
```

Action strings are case-sensitive. Invalid strings, out-of-range integers,
mappings without `action`, and unsupported types raise errors.

```python
print(env.action_space["actions"])
# ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
```

SatSim keeps the full camera footprint inside the scene. A forward move that
would leave the GeoTIFF is blocked but still consumes one step. Turning does
not change position.

> Offline action `-1` aligns the initial observation; it is not accepted by
> `Env.step()`.

## 8. `step()` return values

```python
observation, done, info = env.step(action)
```

| Value | Meaning |
| --- | --- |
| `observation` | Sensor observations after the action |
| `done` | Terminated by `STOP` or the step cap |
| `info` | Metrics, episode identity, and termination state |

Stable `info` fields:

| Field | Meaning |
| --- | --- |
| `metrics` | Results from enabled measures |
| `episode_id` | Original episode ID |
| `episode_key` | Stable `<split>::<scene_id>::<episode_id>` identity |
| `scene_id` | Logical scene name; never a local path |
| `elapsed_steps` | Actions executed so far |
| `episode_over` | Same value as `done` |
| `stop_called` | Whether this step terminated with STOP |
| `max_steps_reached` | Whether the step cap was reached |
| `termination_reason` | `"stop"`, `"max_steps"`, or `None` |

`env.last_step_info` references the latest `info`; it is `None` after reset.

## 9. Complete interaction loop

```python
from applications.resources import load_example_task_config
from satnav.core.env import Env


def choose_action(observation):
    # Invoke your policy here.
    return "STOP"


env = Env(load_example_task_config())
try:
    observation = env.reset()
    done = False

    while not done:
        action = choose_action(observation)
        observation, done, info = env.step(action)

    metrics = env.get_metrics()
    print(env.current_episode.episode_key)
    print(info["termination_reason"])
    print(metrics)
finally:
    env.close()
```

Evaluation must not rely on JSON order or use `episode_id` alone. Use
`episode_key` for stable identity across scenes and splits.

## 10. Metrics

`TASK.MEASUREMENTS` controls `get_metrics()` and `info["metrics"]`:

| Config name | Result key | Meaning |
| --- | --- | --- |
| `DISTANCE_TO_GOAL` | `distance_to_goal` | Distance to the first goal in meters |
| `SUCCESS` | `success` | 1 when STOP occurs inside the success radius |
| `ORACLE_SUCCESS` | `oracle_success` | 1 if the agent ever entered the success radius |
| `PATH_LENGTH` | `path_length` | Sum of distances between consecutive agent positions |
| `SPL` | `spl` | Combined success and path-efficiency metric |
| `TOP_DOWN_MAP` | `top_down_map` | Visualization data for debugging and video |

### 10.1 Success

For a normal episode:

```text
STOP was executed and distance_to_goal < success_distance
```

For a Boundary episode whose start equals its goal, SatNav uses
leave-and-return semantics: the agent must first move farther than
`2 × success_distance`, return inside the radius, and STOP. This prevents an
immediate STOP from succeeding.

### 10.2 Oracle Success

Oracle Success records whether the agent ever reached the goal region; STOP is
not required. Once 1, it remains 1 for the rest of the episode. Boundary
episodes still require leaving before returning.

### 10.3 SPL

SatNav computes geographic reference-path distance:

```text
SPL = Success × reference_path_length
                / max(reference_path_length, actual_path_length)
```

If no valid reference path exists, SatNav attempts the start-to-goal straight
line. If no valid path length can be obtained, SPL is 0.

### 10.4 Top-down map

With `TOP_DOWN_MAP` enabled:

| Field | Meaning |
| --- | --- |
| `map` | `(H, W, 3)` top-down image |
| `agent_map_coord` | Agent `(row, col)` on the map |
| `agent_angle` | Current heading |
| `bounds` | Geographic visualization bounds |
| `step_count` | Number of recorded visualization steps |

Top-down maps add rendering and memory overhead. Remove this measurement for
training or large-scale evaluation without videos.

## 11. Public `Env` properties

| Property | Type | Meaning |
| --- | --- | --- |
| `episodes` | `list[VLNEpisode]` | Dataset episodes; empty without a dataset |
| `current_episode` | `VLNEpisode | None` | Active episode |
| `simulator` | `Simulator` | Public simulator interface |
| `agent_state` | `AgentState` | Current WGS84 position and heading |
| `last_step_info` | `dict | None` | Most recent step info |
| `episode_over` | `bool` | Whether the episode has ended |
| `observation_space` | `dict` | Lightweight observation description |
| `action_space` | `dict` | Lightweight action description |
| `max_episode_steps` | `int` | Per-episode step cap |

```python
state = env.agent_state
print(state.position)  # [longitude, latitude, altitude]
print(state.rotation)  # 0° = North, clockwise
```

`AgentState.position` is list-like and supports `.tolist()`.

## 12. Episode API

`env.current_episode` returns `VLNEpisode`; see
[Data Format](../dataset/DATASET_FORMAT.md) for all fields. The key Core API
boundary is identity and serialization:

```python
episode = env.current_episode

print(episode.episode_id)
print(episode.scene_id)
print(episode.episode_key)

public_data = episode.to_dict()
debug_data = episode.to_dict(include_runtime=True)
```

`to_dict()` returns public benchmark fields and excludes local paths. Only
`include_runtime=True` adds `split`, `scene_path`, and `episode_key`.
`scene_id` must remain a stable logical ID; never serialize local GeoTIFF paths
into benchmark results.

## 13. Simulator API

Normal code interacts through `Env`. Navigation helpers and low-level
debugging may use `env.simulator`:

| Method/property | Meaning |
| --- | --- |
| `reset(scene_id)` | Load and reset a scene |
| `set_agent_state(position, rotation)` | Set WGS84 position and heading |
| `get_agent_state()` | Return `AgentState` |
| `step(action)` | Execute a simulator action directly |
| `get_observations()` | Get raw simulator observations |
| `geodesic_distance(a, b)` | Distance between geographic positions |
| `is_navigable(position)` | Whether the full camera footprint is in bounds |
| `sensor_suite` | Simulator sensor description |
| `action_space` | Supported simulator actions |
| `close()` | Release simulator resources |

Direct `simulator.step()` does not update `Env` step counts, termination, or
measures. Normal rollouts must call `env.step()`.

### 13.1 Coordinates and rendering

Public positions are WGS84 `[longitude, latitude, altitude]`; heading is
clockwise from north. SatSim transforms WGS84 to EPSG:3857 Web Mercator,
performs movement and camera-footprint calculations in planar meters, then
crops, rotates, and resizes the GeoTIFF into RGB.

Ground coverage depends on altitude, HFOV, width, and height. Higher altitude
covers more ground. If the complete footprint does not fit, an initial state
is rejected or a forward action remains at the current position.

### 13.2 Scene resolution and caching

Episodes store logical `scene_id` only. `SatNavDataset` resolves a runtime
`scene_path` under `SCENES_DIR`, and SatSim adds `.tif` when needed. Consecutive
episodes in one scene reuse an open Rasterio dataset. `close()` releases and
clears the cache.

## 14. External simulators

SatNav can load an external backend by fully qualified class name:

```yaml
SIMULATOR:
  TYPE: custom
  CLASS: my_package.my_simulator.MySimulator
```

The class must:

1. inherit `satnav.core.simulator.Simulator`;
2. accept `(config, scenes_dir)` in its constructor;
3. implement the abstract scene, action, state, observation, distance, and
   navigability interfaces;
4. return task-compatible `rgb` observations.

```python
from satnav.core.simulator import AgentState, Simulator


class MySimulator(Simulator):
    def __init__(self, config, scenes_dir):
        ...

    def reset(self, scene_id):
        ...

    def step(self, action):
        ...

    def get_agent_state(self) -> AgentState:
        ...

    def set_agent_state(self, position, rotation):
        ...

    def get_observations(self):
        ...

    def geodesic_distance(self, position_a, position_b):
        ...

    def is_navigable(self, position):
        ...

    @property
    def sensor_suite(self):
        ...

    @property
    def action_space(self):
        ...
```

Unknown `TYPE` values do not fall back to SatSim. Missing `SIMULATOR.CLASS`,
failed imports, or classes not derived from `Simulator` fail immediately.

## 15. Troubleshooting

### Why is `done` still `False` after reaching the goal?

Entering the success radius does not terminate the episode. The policy must
execute `STOP`; otherwise the episode continues until the step cap.

### Why is `success` 0 after `STOP`?

At STOP, `distance_to_goal` must be below the trajectory type's
`SUCCESS_DISTANCE`. Boundary episodes must also leave the start region before
returning.

### Why did `MOVE_FORWARD` not change position?

The proposed camera footprint may leave the safe GeoTIFF bounds. SatSim keeps
the current position, but the action still counts as a step.

### Why does `reset()` report that all episodes are exhausted?

The default is `cycle=False`. End evaluation after exhaustion, or construct
`Env(..., cycle=True)` for repeated data.

### Why is RGB not 224 × 224?

Resolution is simulator- and config-defined. Read `observation["rgb"].shape`
or `env.observation_space["rgb"]["shape"]`; never hard-code it.

### Why can SatNav not find a scene?

`DATASET.SCENES_DIR` must contain the logical scene name:

```text
<SCENES_DIR>/Amsterdam-1.tif
```

Validate paths with:

```bash
bash scripts/validation/data_validation.sh
```

### Should I use `Env` or SatSim directly?

Models, applications, and evaluation should use `Env`, which synchronizes
observations, metrics, termination, and episode identity. Access
`env.simulator` directly only for backend implementation, rendering debugging,
or navigation helpers.
