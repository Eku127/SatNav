# Integrating a Model with SatNav

This guide explains how to connect a new navigation model to SatNav and run Episodes through the shared online evaluator. The integration layer is framework-independent: PyTorch, Transformers, a remote inference service, or a rule-based policy can all implement the same `PolicyAdapter` interface.

Before starting, read [Core API](../core/CORE_API.md) and [Evaluation](../evaluation/README.md). The first describes observations, actions, and `Env`; the second covers Episode selection, result files, and aggregation.

See [architecture](../concepts/OVERVIEW.md) for environment, policy, and evaluator responsibilities and their interaction loop.

## 1. Prepare the environment

Install SatNav Core in the model's own Python environment:

```bash
python -m pip install -e .
python -c "from satnav.evaluation import Evaluator, PolicyAdapter; print('SatNav import OK')"
```

Models may use different PyTorch, Transformers, or CUDA dependencies. SatNav only requires the model environment to import `satnav`; model dependencies do not need to be installed in the Core environment.

For an initial integration, use the bundled synthetic data and scene:

```text
applications/resources/
├── map.tif
├── satnav_example_episodes.json
└── satnav_example_task.yaml
```

These resources require no additional download. After the integration completes a rollout on them, switch to real Episodes and GeoTIFF scenes. See [Download Episode Data](../dataset/DATA_DOWNLOAD.md) and [Download Satellite Scenes](../applications/MAP_DOWNLOAD.md).

## 2. Integration interface

An evaluation follows this call sequence:

```text
Episode
  → Env.reset_to_episode()
  → observation
  → PolicyAdapter.act()
  → PolicyStep
  → Env.step()
  → metrics and result
```

A model implements three methods:

```python
class PolicyAdapter:
    def reset(self, context): ...
    def act(self, observation): ...
    def close(self): ...
```

| Method | Called | Responsibility |
| --- | --- | --- |
| `reset(context)` | At the start of each Episode | Clear history and read the current Episode context |
| `act(observation)` | Before each environment step | Return one SatNav primitive action |
| `close()` | When the worker finishes | Release the model, file handles, or external connections |

`PolicyAdapter` is a Python Protocol. Any object implementing these three methods can be passed to `Evaluator`.

## 3. Minimal adapter

This adapter moves forward three times and then returns `STOP`. It loads no model, but demonstrates Episode state reset, action output, and resource cleanup.

```python
from typing import Any, Mapping, Optional

from satnav.evaluation import EpisodeContext, PolicyStep
from satnav.task.actions import Action


class ForwardThenStopAdapter:
    def __init__(self, forward_steps: int = 3) -> None:
        self.forward_steps = int(forward_steps)
        if self.forward_steps < 0:
            raise ValueError("forward_steps must be non-negative")
        self._step = 0
        self._episode_key: Optional[str] = None

    def reset(self, context: EpisodeContext) -> None:
        self._step = 0
        self._episode_key = context.episode_key

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        del observation
        if self._episode_key is None:
            raise RuntimeError("reset() must be called before act()")

        if self._step >= self.forward_steps:
            action = Action.STOP
        else:
            action = Action.MOVE_FORWARD

        self._step += 1
        return PolicyStep(
            action=action,
            info={"adapter_step": self._step},
        )

    def close(self) -> None:
        self._step = 0
        self._episode_key = None
```

The adapter object remains alive after an Episode ends and handles later Episodes. Reinitialize every Episode-scoped value in `reset()`.

## 4. EpisodeContext

`reset()` receives an `EpisodeContext` with the current Episode and evaluation process information:

| Field | Description |
| --- | --- |
| `episode` | Current `VLNEpisode` |
| `episode_key` | Stable `<split>::<scene_id>::<episode_id>` identifier |
| `split` | Current dataset split |
| `episode_index` | Position in the globally selected Episode list |
| `rank` / `world_size` | Current worker and total worker count |
| `max_steps` | Maximum actions for the Episode |
| `seed` | Deterministic Episode seed |
| `environment` | Current public `Env` object |
| `simulator` | Convenience access to `environment.simulator` |
| `agent_state` | Convenience access to `environment.agent_state` |

When the policy samples randomly, initialize an Episode-scoped random-number generator from `context.seed` in every `reset()`. The same Episode then receives the same random sequence across different rank counts and resume boundaries.

The adapter may read `context.episode.instruction`, `reference_path`, or `trajectory_type`, and may read the current position through `context.agent_state`. Do not access private `Env` fields such as `_dataset`, `_task`, or `_sim`.

## 5. Observations and actions

The default `VLNTask` returns:

| Key | Type | Description |
| --- | --- | --- |
| `rgb` | `numpy.ndarray` | Satellite observation with shape `(H, W, 3)` and dtype `uint8` |
| `instruction` | `dict` | `{"text": str}` |
| `agent_pose` | `numpy.ndarray` | Relative position and heading with shape `(4,)` |

RGB dimensions come from the task configuration. Preprocessing must read the actual input shape instead of assuming 224 or 448. See [Core API — Observations](../core/CORE_API.md#6-observations) for the full definition.

SatNav actions are:

| ID | Action |
| ---: | --- |
| 0 | `STOP` |
| 1 | `MOVE_FORWARD` |
| 2 | `TURN_LEFT` |
| 3 | `TURN_RIGHT` |

`PolicyStep.action` accepts an action string or integer ID. Map model output to a SatNav action explicitly before returning it:

```python
from satnav.task.actions import Action


MODEL_ACTIONS = {
    "stop": Action.STOP,
    "forward": Action.MOVE_FORWARD,
    "left": Action.TURN_LEFT,
    "right": Action.TURN_RIGHT,
}


def decode_action(model_output: str) -> str:
    key = model_output.strip().lower()
    if key not in MODEL_ACTIONS:
        raise ValueError(f"unsupported model action: {model_output!r}")
    return MODEL_ACTIONS[key]
```

Do not silently map unrecognized output to `STOP`. Doing so records a model-format failure as a normal termination and changes evaluation metrics.

`PolicyStep.info` can hold diagnostic values such as generated text, confidence, or action-queue length. This content is written to JSONL when action traces are enabled, so it must:

- be JSON serializable;
- exclude full RGB observations, model tensors, and other large objects;
- exclude local absolute checkpoint, dataset, and scene paths;
- exclude tokens, credentials, and remote-service request headers.

## 6. Run the minimal evaluation

Connect `ForwardThenStopAdapter` from Section 3 to the bundled example environment:

```python
import json
from pathlib import Path

from applications.resources import load_example_task_config
from satnav.core.env import Env
from satnav.evaluation import EvaluationConfig, Evaluator


config = load_example_task_config()
environment = Env(config, cycle=False)
policy = ForwardThenStopAdapter(forward_steps=3)

summary = Evaluator(
    environment=environment,
    policy=policy,
    config=EvaluationConfig(
        output_dir=Path("output/model_integration/forward_then_stop"),
        split=str(config.DATASET.SPLIT),
        policy_id="forward-then-stop",
        limit=2,
        max_steps=10,
        fail_on_episode_error=True,
    ),
).run()

print(json.dumps(summary, indent=2, sort_keys=True))
```

`Evaluator.run()` closes both the policy and environment on successful or exceptional completion. Results are written to:

```text
output/model_integration/forward_then_stop/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

Check `status` and `error_episode_count` in `summary.json`, then inspect the action trace and metrics for individual Episodes in JSONL. See [Evaluation — Output format](../evaluation/README.md#5-output-format) for all result fields.

## 7. Integrate a real model

A real model is normally loaded once when the adapter is created and reused across Episodes. Only Episode-scoped state—such as recurrent state, image history, prompts, and queued actions—must be cleared in `reset()`.

```python
from typing import Any, List, Mapping, Optional

from satnav.evaluation import EpisodeContext, PolicyStep


class MyModelAdapter:
    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self._context: Optional[EpisodeContext] = None
        self._history: List[Any] = []
        self._action_queue: List[str] = []
        self._closed = False

    @classmethod
    def from_pretrained(cls, model_path: str, device: str) -> "MyModelAdapter":
        # Import the model framework and load the checkpoint here.
        model, processor = load_your_model(model_path, device=device)
        model.eval()
        return cls(model=model, processor=processor, device=device)

    def reset(self, context: EpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("adapter is closed")
        self._context = context
        self._history.clear()
        self._action_queue.clear()
        reset_model_state(self.model)

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")

        if self._action_queue:
            action = self._action_queue.pop(0)
            source = "queued"
        else:
            rgb = observation["rgb"]
            instruction = observation["instruction"]["text"]
            model_input = self.processor(
                image=rgb,
                text=instruction,
                history=self._history,
            )
            prediction = run_model(self.model, model_input, device=self.device)
            action, remaining = parse_model_actions(prediction)
            self._action_queue.extend(remaining)
            source = "generated"

        self._history.append(observation["rgb"])
        return PolicyStep(
            action=action,
            info={
                "source": source,
                "queue_remaining": len(self._action_queue),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._context = None
        self._history.clear()
        self._action_queue.clear()
        self.model = None
        self.processor = None
        self._closed = True
```

The concrete model supplies `load_your_model()`, `reset_model_state()`, `run_model()`, and `parse_model_actions()`. Verify that:

- checkpoint loading is complete and the model enters evaluation mode;
- inference disables gradient computation;
- RGB channel order, size, and normalization match training;
- the instruction comes from the current Episode, not a stale cache;
- recurrent or KV cache, frame history, and action queue are cleared in every `reset()`;
- an action chunk is returned one primitive action at a time rather than executing several environment steps inside one `act()`;
- repeated calls to `close()` are safe.

## 8. Project layout

A model may live in a separate repository and depend only on SatNav's public interfaces:

```text
my_satnav_model/
├── pyproject.toml
├── my_satnav_model/
│   ├── adapter.py
│   └── evaluate.py
├── configs/
│   └── satnav_task.yaml
└── README.md
```

For a baseline maintained inside SatNav, use this layout:

```text
baselines/vlm/<model_name>/
├── adapter.py
├── evaluate.py
├── configs/
│   └── satnav_task.yaml
├── scripts/
│   └── eval.sh
├── requirements.txt
├── local.env.example
└── README.md
```

Public configurations should contain relative paths and model-independent settings only. Supply checkpoint, Episode, GeoTIFF, upstream source, and output paths through CLI arguments, environment variables, or a Git-ignored `.local/env.sh`.

Import model dependencies lazily inside `adapter.py` or the model factory. Running only `import satnav` or `import satnav.evaluation` must not load PyTorch, Transformers, or a model checkpoint.

## 9. Evaluation entry point

A model evaluation command normally:

1. parses model, Episode, scene, and output paths;
2. loads a task configuration and sets the split, `DATASET.DATA_PATH`, and `DATASET.SCENES_DIR`;
3. creates `SatNavDataset` and `Env(cycle=False)`;
4. loads the model and creates its adapter;
5. builds `EvaluationConfig`;
6. calls `Evaluator.run()`;
7. prints the aggregate values from `summary.json`.

Provide these common arguments for each model:

| Argument | Purpose |
| --- | --- |
| `--model-path` | Checkpoint or model directory |
| `--task-config` | SatNav task configuration |
| `--episodes` | Episode JSON for the selected split |
| `--scenes-dir` | GeoTIFF scene directory |
| `--split` | `val_seen`, `val_unseen`, or `test` |
| `--output-dir` | Evaluation output directory |
| `--offset` / `--limit` | Episode selection range |
| `--rank` / `--world-size` | Multi-process Episode sharding |
| `--base-seed` | Evaluation random seed |
| `--max-steps` | Maximum actions per Episode |
| `--resume` | Skip Episodes completed by the current rank |
| `--fail-on-episode-error` | Fail the final command if any Episode fails |
| `--dry-run` | Validate configuration and Episode selection without loading the model |

Existing entry points provide useful examples:

- `baselines/classic/evaluate.py` for a Classic factory and the shared evaluator;
- `baselines/vlm/navila/evaluate.py` for an external model, isolated environment, and GPU selection;
- `baselines/vlm/openfly/evaluate.py` for local checkpoint and processor loading.

`--dry-run` should exit before model loading so developers can first inspect the Episode count, rank shard, and output directory.

## 10. Multi-rank evaluation

Online evaluation shards Episodes. Each rank loads an independent model and runs its own Episodes, so the model does not need DDP wrapping. All ranks must use identical data, split, selection arguments, seed, maximum steps, and output directory.

The following launches two GPUs; `my_satnav_model.evaluate` represents the model's own CLI:

```bash
CUDA_VISIBLE_DEVICES=0 python -m my_satnav_model.evaluate \
  --rank 0 --world-size 2 --device cuda:0 \
  --output-dir output/my_model/val_seen &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python -m my_satnav_model.evaluate \
  --rank 1 --world-size 2 --device cuda:0 \
  --output-dir output/my_model/val_seen &
pid1=$!

wait "$pid0"
wait "$pid1"

python scripts/evaluation/aggregate.py output/my_model/val_seen \
  --fail-on-episode-error
```

`limit` applies to the globally sorted list before stride sharding. With `limit=8` and `world_size=2`, each rank processes four Episodes. See [Evaluation — Episode selection and sharding](../evaluation/README.md#3-episode-selection-and-sharding) for the full rules.

## 11. Resume and error handling

After an interruption, restart with the same arguments and output directory and set `resume=True` or pass `--resume`. The evaluator skips every `episode_key` already present in the current rank's JSONL.

Resume only when the model, data, split, rank count, seed, and evaluation settings remain unchanged. Use a new output directory after changing run conditions.

While debugging a new adapter, enable:

```text
fail_fast=True
fail_on_episode_error=True
capture_action_trace=True
```

`fail_fast` stops the worker after recording its first failed Episode. `fail_on_episode_error` makes the final command return a nonzero status when a failure exists. The action trace shows the primitive actions actually executed after adapter decoding.

Once a large-scale evaluation is stable, set `capture_action_trace=False` to reduce result size.

## 12. Troubleshooting

### Where should a model read the instruction?

Use either `observation["instruction"]["text"]` or `context.episode.instruction.instruction_text` in `reset()`. The observation is convenient for uniform per-step preprocessing; the context is convenient for constructing an Episode-level prompt.

### What if the model generates several actions at once?

Parse the result into an action queue. Return the first primitive action from the current `act()` call and return later actions on subsequent calls. Continue updating observation history and model state after every `Env.step()`.

### Must model source live in the SatNav repository?

No. An external package only needs to install SatNav and implement `PolicyAdapter`. Put a model under `baselines/` only when SatNav itself will maintain and release the integration.

### Can an adapter control the simulator directly?

Normally, no. Return an action and let the evaluator call `Env.step()` so Episode termination, metrics, and action traces remain synchronized. Use public `context.simulator` interfaces only when a navigation helper needs to read lower-level state.

### Why is Success still 0 after the model returns STOP?

`STOP` only means the model chose to end the Episode. `SUCCESS` is 1 only when the agent is within the success radius for the current task type at that moment. See [Evaluation](../evaluation/README.md) for the standard thresholds.
