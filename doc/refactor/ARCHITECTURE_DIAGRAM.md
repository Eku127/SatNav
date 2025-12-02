# SatNav 架构图

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         SatNav Platform                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Dataset    │──────│     Env      │──────│   Task   │ │
│  │  (R2R)       │      │              │      │  (VLN)   │ │
│  └──────────────┘      └──────┬───────┘      └────┬─────┘ │
│                                │                    │       │
│                                │                    │       │
│                         ┌──────▼───────┐           │       │
│                         │  Simulator   │           │       │
│                         │ (SatSimWrap) │           │       │
│                         └──────┬───────┘           │       │
│                                │                    │       │
│                         ┌──────▼───────┐           │       │
│                         │    satsim    │           │       │
│                         │  (待实现)     │           │       │
│                         └──────────────┘           │       │
│                                                      │       │
│  ┌──────────────┐      ┌──────────────┐      ┌────▼─────┐ │
│  │   Config     │──────│   Sensors    │──────│ Measures │ │
│  │  (YAML)      │      │              │      │          │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心组件关系图

```
                    ┌─────────────┐
                    │    Config   │
                    │  (YAML)     │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Dataset    │   │   Simulator  │   │     Task     │
│              │   │              │   │              │
│ - episodes   │   │ - reset()    │   │ - sensors    │
│ - load()     │   │ - step()     │   │ - actions    │
│              │   │ - get_obs()  │   │ - measures   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │     Env      │
                   │              │
                   │ - reset()    │
                   │ - step()     │
                   │ - metrics()  │
                   └──────────────┘
```

## 3. 数据流图

### 3.1 初始化流程

```
Config (YAML)
    │
    ├──→ Dataset.__init__()
    │        │
    │        └──→ load episodes from JSON.gz
    │
    ├──→ Simulator.__init__()
    │        │
    │        └──→ initialize satsim (待实现)
    │
    └──→ Task.__init__()
             │
             ├──→ create Sensors
             └──→ create Measures
                     │
                     └──→ Env.__init__(dataset, simulator, task)
```

### 3.2 Reset 流程

```
env.reset()
    │
    ├──→ dataset.get_episode()
    │        │
    │        └──→ episode (VLNEpisode)
    │
    ├──→ task.reset(episode)
    │        │
    │        ├──→ set goal position
    │        └──→ reset all measures
    │
    ├──→ sim.reset(episode.scene_id)
    │        │
    │        ├──→ load scene (待实现)
    │        └──→ set agent state (待实现)
    │
    └──→ get observations
             │
             ├──→ sim.get_observations()
             │        └──→ RGB image
             │
             └──→ task.get_observations()
                      └──→ Instruction (text)
             │
             └──→ return combined obs
```

### 3.3 Step 流程

```
env.step(action)
    │
    ├──→ task.step(action)
    │        │
    │        └──→ validate action
    │
    ├──→ sim.step(action)
    │        │
    │        ├──→ execute action (待实现)
    │        └──→ update agent state (待实现)
    │
    ├──→ sim.get_observations()
    │        └──→ RGB image
    │
    ├──→ update measures
    │        │
    │        ├──→ DistanceToGoal.update()
    │        ├──→ Success.update()
    │        ├──→ PathLength.update()
    │        └──→ SPL.update() (if done)
    │
    ├──→ check done
    │        │
    │        ├──→ Success == True?
    │        └──→ steps >= max_steps?
    │
    └──→ return (obs, reward, done, info)
```

## 4. 任务组件图

```
                    ┌──────────────┐
                    │   VLNTask    │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Sensors    │   │   Actions    │   │   Measures   │
│              │   │              │   │              │
│ - RGB        │   │ - STOP       │   │ - Success    │
│ - Instruction│   │ - MOVE_FWD   │   │ - SPL        │
│              │   │ - TURN_LEFT  │   │ - DistToGoal │
│              │   │ - TURN_RIGHT│   │ - PathLength │
└──────────────┘   └──────────────┘   └──────────────┘
```

## 5. 传感器数据流

```
Simulator
    │
    ├──→ RGBSensor
    │        │
    │        └──→ RGB Image (H, W, 3) uint8
    │
    └──→ InstructionSensor (from Task)
             │
             └──→ Instruction
                      └──→ text: str
```

## 6. 评价指标计算流程

```
Each Step:
    │
    ├──→ DistanceToGoal
    │        │
    │        └──→ sim.geodesic_distance(agent_pos, goal_pos)
    │
    ├──→ Success
    │        │
    │        └──→ DistanceToGoal < success_distance?
    │
    └──→ PathLength
             │
             └──→ accumulate euclidean_distance(prev_pos, curr_pos)

Episode End:
    │
    └──→ SPL
             │
             └──→ Success * (ref_path_len / max(ref_path_len, actual_path_len))
```

## 7. 仿真器接口图（待实现）

```
SatSimWrapper
    │
    ├──→ reset(scene_id)
    │        │
    │        └──→ satsim.load_scene() [待实现]
    │
    ├──→ step(action)
    │        │
    │        └──→ satsim.execute_action() [待实现]
    │
    ├──→ get_agent_state()
    │        │
    │        └──→ satsim.get_state() [待实现]
    │
    ├──→ get_observations()
    │        │
    │        └──→ satsim.get_rgb() [待实现]
    │
    └──→ geodesic_distance(pos_a, pos_b)
             │
             └──→ satsim.compute_geodesic() [待实现]
```

## 8. Episode 数据结构图

```
VLNEpisode (卫星地图)
    │
    ├──→ episode_id: str
    ├──→ scene_id: str
    ├──→ start_position: [longitude, latitude, altitude]
    ├──→ start_rotation: roll角度 (0-360度，0表示正北)
    ├──→ goals: List[NavigationGoal]
    │        │
    │        └──→ position: [longitude, latitude, altitude]
    │
    ├──→ reference_path: List[[lon, lat, alt], ...]
    ├──→ instruction: InstructionData
    │        │
    │        └──→ instruction_text: str
    │
    └──→ trajectory_id: str
```

## 9. 配置文件结构图

```
vln_task.yaml
    │
    ├──→ ENVIRONMENT
    │        └──→ MAX_EPISODE_STEPS: 500
    │
    ├──→ SIMULATOR
    │        ├──→ FORWARD_STEP_SIZE: 0.25
    │        ├──→ TURN_ANGLE: 15
    │        ├──→ RGB_SENSOR
    │        │        ├──→ WIDTH: 224
    │        │        ├──→ HEIGHT: 224
    │        │        └──→ HFOV: 90
    │
    ├──→ TASK
    │        ├──→ TYPE: VLN-v0
    │        ├──→ SUCCESS_DISTANCE: 3.0
    │        ├──→ POSSIBLE_ACTIONS: [...]
    │        └──→ MEASUREMENTS: [...]
    │
    └──→ DATASET
             ├──→ TYPE: SatNav (compatibility only, not used)
             ├──→ SPLIT: train
             ├──→ DATA_PATH: ...
             └──→ SCENES_DIR: ...
```

## 10. 类继承关系图

```
Object
    │
    ├──→ Dataset
    │        │
    │        └──→ R2RDataset
    │
    ├──→ Simulator (ABC)
    │        │
    │        └──→ SatSimWrapper
    │
    ├──→ Task (ABC)
    │        │
    │        └──→ VLNTask
    │
    ├──→ Sensor (ABC)
    │        │
    │        ├──→ RGBSensor
    │        └──→ InstructionSensor
    │
    ├──→ Measure (ABC)
    │        │
    │        ├──→ Success
    │        ├──→ SPL
    │        ├──→ DistanceToGoal
    │        └──→ PathLength
    │
    └──→ Env
```

## 11. 模块依赖关系

```
core/
    ├──→ config.py (独立)
    ├──→ episode.py (独立)
    ├──→ simulator.py (独立)
    └──→ env.py
            │
            ├──→ depends on: config, episode, simulator
            ├──→ depends on: dataset
            └──→ depends on: task

task/
    ├──→ actions.py (独立)
    ├──→ sensors.py
    │        │
    │        └──→ depends on: core.simulator
    ├──→ measures.py
    │        │
    │        └──→ depends on: core.simulator
    └──→ vln_task.py
            │
            ├──→ depends on: sensors, actions, measures
            └──→ depends on: core.episode

dataset/
    └──→ r2r_dataset.py
            │
            └──→ depends on: core.episode

sims/
    └──→ satsim_wrapper.py
            │
            └──→ depends on: core.simulator
```

## 12. 执行时序图

```
User Code          Env            Task          Simulator      Dataset
    │               │               │               │              │
    │──reset()──────>│               │               │              │
    │               │──get_episode()──────────────────────────────>│
    │               │<──────────────episode────────────────────────│
    │               │──reset(ep)───>│               │              │
    │               │               │──reset()──────>│              │
    │               │               │<──ready────────│              │
    │               │<──obs─────────│               │              │
    │<──obs─────────│               │               │              │
    │               │               │               │              │
    │──step(action)->│               │               │              │
    │               │──step(action)->│               │              │
    │               │               │──step(action)->│              │
    │               │               │<──obs──────────│              │
    │               │               │──update()──────│              │
    │               │<──done────────│               │              │
    │<──(obs,done)──│               │               │              │
```

## 总结

以上架构图展示了 SatNav 平台的核心组件、数据流、依赖关系和执行流程。整体设计简洁清晰，各组件职责明确，便于理解和实现。

