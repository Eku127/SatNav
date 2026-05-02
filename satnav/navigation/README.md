# SatNav Navigation Module

Navigation utilities for path following and discrete action planning in continuous geographic space.

## Overview

This module is adapted from [VLN-CE](https://github.com/jacobkrantz/VLN-CE) navigation components, but customized for SatNav's **geographic coordinate system (WGS84)**:

| Feature | Habitat-lab / VLN-CE | SatNav |
|---------|---------------------|--------|
| Coordinate System | 3D Cartesian (x, y, z) | Geographic (lon, lat, alt) |
| Rotation | Quaternion | Degrees (0° = North) |
| Distance | Euclidean | Geodesic (Haversine) |
| Direction | Vector math | Geographic bearing |

## Components

### 1. DiscretePathPlanner

**Purpose**: Converts a relative waypoint (distance + bearing) into a complete discrete action sequence in a single call.

**Use Case**: Waypoint model inference - converting high-level waypoint predictions to low-level actions.

```python
from satnav.navigation import DiscretePathPlanner

# Initialize (parameters should match simulator config)
planner = DiscretePathPlanner(
    forward_distance=0.25,  # meters per MOVE_FORWARD
    turn_angle=15.0,        # degrees per TURN_LEFT/RIGHT
    goal_radius=0.5,        # arrival threshold (meters)
)

# Plan action sequence
# Target: 10m ahead, 45° to the right
actions = planner.plan(distance_meters=10.0, bearing_degrees=45.0)
# Returns: ['TURN_RIGHT', 'TURN_RIGHT', 'TURN_RIGHT', 'MOVE_FORWARD', ...]

# VLN-CE compatible interface (polar coordinates, radians)
import math
actions = planner.plan_polar(r=10.0, theta=math.radians(-45))
```

**Bearing Convention**:
- `0°`: directly ahead (forward)
- `90°`: to the right
- `-90°` or `270°`: to the left
- `180°`: behind

---

### 2. SatNavPathFollower

**Purpose**: Given a goal position in geographic coordinates, iteratively returns the next action (one action per call).

**Use Cases**:
- Teacher forcing during training (oracle action generation)
- Oracle sensor (ShortestPathSensor)
- Online navigation with replanning

```python
from satnav.navigation import SatNavPathFollower

# Initialize
follower = SatNavPathFollower(
    goal_radius=3.0,   # success distance threshold (meters)
    turn_angle=15.0,   # degrees per turn action
)

# Iteratively get actions
goal = [114.07, 22.54, 100.0]  # [longitude, latitude, altitude]
while True:
    action = follower.get_next_action(goal, simulator)
    if action == "STOP":
        break
    simulator.step(action)

# Or get complete sequence at once
actions = follower.get_action_sequence_to_goal(
    goal_position=goal,
    simulator=simulator,
    max_steps=1000,
    execute_actions=True  # whether to execute in simulator
)
```

---

### 3. ReferencePathFollower

**Purpose**: Navigates along a predefined reference path (multiple waypoints) sequentially.

**Use Cases**:
- Converting `reference_path` from dataset to action sequences
- Generating teacher forcing data for offline/recollection training
- Evaluating path following accuracy

```python
from satnav.navigation import ReferencePathFollower

# Initialize
path_follower = ReferencePathFollower(
    goal_radius=3.0,       # success distance for all waypoints including goal
    turn_angle=15.0,
)

# Reference path (loaded from dataset)
reference_path = [
    [114.06, 22.54, 100.0],  # start
    [114.07, 22.54, 100.0],  # waypoint 1
    [114.08, 22.55, 100.0],  # waypoint 2 (goal)
]

# Option 1: Get complete action sequence at once
actions = path_follower.follow_path(
    reference_path=reference_path,
    simulator=simulator,
    execute=True,
    max_steps=2000
)

# Option 2: Iterate for online training data collection
path_follower.reset(reference_path)
training_data = []
while True:
    action = path_follower.get_next_action(simulator)
    if action == "STOP":
        break
    obs = simulator.step(action)
    training_data.append((obs, action))

# Check progress
print(f"Progress: {path_follower.get_progress():.1%}")
print(f"Current waypoint: {path_follower.get_current_waypoint_index()}")
```

---

### 4. ShortestPathSensor

**Purpose**: Sensor-style wrapper compatible with SatNav sensor interface for getting oracle actions.

```python
from satnav.navigation.path_follower import ShortestPathSensor

sensor = ShortestPathSensor(
    goal_radius=3.0,
    turn_angle=15.0,
    return_index=True  # return action index instead of string
)

# Get oracle action using episode goal
oracle_action = sensor.get_observation(
    simulator=simulator,
    episode=current_episode  # uses episode.goals[0].position
)

# Or specify goal position directly
oracle_action = sensor.get_observation(
    simulator=simulator,
    goal_position=[114.07, 22.54, 100.0]
)
```

---

## Typical Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Data Generation                     │
│                                                                 │
│  Dataset (reference_path)  ──▶  ReferencePathFollower           │
│                                      │                          │
│                                      ▼                          │
│                            Action Sequences + Observations      │
│                            (for supervised learning)            │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Model Training                          │
│                                                                 │
│  Observation ──▶ Model ──▶ Predicted Action                     │
│                    ▲                                            │
│                    │                                            │
│  Oracle Action ◀── SatNavPathFollower / ShortestPathSensor      │
│                                                                 │
│  Loss = CrossEntropy(predicted_action, oracle_action)           │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Waypoint Model Inference                     │
│                                                                 │
│  Observation ──▶ Model ──▶ Waypoint (r, θ) ──▶ DiscretePathPlanner
│                                                    │            │
│                                                    ▼            │
│                                          Action Sequence ──▶ Execute
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Comparison

| Feature | DiscretePathPlanner | SatNavPathFollower | ReferencePathFollower |
|---------|---------------------|-------------------|----------------------|
| **Input** | Relative position (distance, bearing) | Absolute goal coordinates | Complete reference path |
| **Output** | Complete action list | Single action | Complete list or single action |
| **Requires Simulator** | ❌ No | ✅ Yes | ✅ Yes |
| **Primary Use** | Waypoint model inference | Oracle sensor, online navigation | Training data generation |
| **VLN-CE Equivalent** | `discrete_planner.py` | `shortest_path_follower.py` | Dataset trajectory |

---

## Utility Functions

### calculate_bearing

Calculates the bearing (azimuth) between two geographic coordinates.

```python
from satnav.navigation.path_follower import calculate_bearing

# Bearing from Beijing to Shanghai
bearing = calculate_bearing(
    lon1=116.4, lat1=39.9,  # Beijing
    lon2=121.5, lat2=31.2   # Shanghai
)
print(f"Bearing: {bearing:.1f}°")  # ~136° (Southeast)
```

### normalize_angle_diff

Normalizes an angle difference to the [-180°, 180°] range.

```python
from satnav.navigation.path_follower import normalize_angle_diff

normalize_angle_diff(270)   # → -90
normalize_angle_diff(-270)  # → 90
normalize_angle_diff(450)   # → 90
```

---

## Configuration Alignment

Ensure navigation component parameters match your simulator configuration:

```yaml
# configs/satnav_task.yaml
SIMULATOR:
  FORWARD_STEP_SIZE: 0.25   # ← planner.forward_distance
  TURN_ANGLE: 15            # ← planner.turn_angle

TASK:
  SUCCESS_DISTANCE: 3.0     # ← follower.goal_radius
```

```python
# Initialize using config
from satnav.core.config import load_config

config = load_config("configs/satnav_task.yaml")

planner = DiscretePathPlanner(
    forward_distance=config.SIMULATOR.FORWARD_STEP_SIZE,
    turn_angle=config.SIMULATOR.TURN_ANGLE,
    goal_radius=config.TASK.SUCCESS_DISTANCE,
)
```

---

## References

- [VLN-CE: habitat_extensions/discrete_planner.py](https://github.com/jacobkrantz/VLN-CE)
- [Habitat-lab: habitat/tasks/nav/shortest_path_follower.py](https://github.com/facebookresearch/habitat-lab)
