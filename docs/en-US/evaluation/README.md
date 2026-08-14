# Evaluation

SatNav provides one online evaluation interface for the Classic Random, ReferenceFollower, Seq2Seq, and CMA baselines, as well as StreamVLN, NaVILA, Uni-NaVid, and OpenFly. All of them use the same `satnav.evaluation` rollout and result contract.

## 1. Evaluation workflow

The shared evaluation layer:

- sorts, selects, and shards Episodes by a stable key;
- assigns a deterministic seed to each Episode;
- performs rollouts through `PolicyAdapter` and the public `Env` API;
- appends one JSONL result stream per rank;
- resumes unfinished runs from existing Episode records;
- aggregates scalar metrics into `summary.json`.

Each baseline initializes its model, tokenizer, or processor and connects it through a `PolicyAdapter`. This lets Classic methods and VLMs in separate dependency environments share the same Episode selection, rollout behavior, and result format.

## 2. Quickstart

Train and evaluate the Classic baselines on the repository's tiny example:

```bash
bash scripts/quickstart_models.sh
```

See [Training Models with SatNav](../training/README.md) for dependencies, training stages, and output locations.

To call the shared Classic CLI directly:

```bash
python -m baselines.classic \
  --method random \
  --config configs/baselines/random_agent.yaml \
  --split test \
  --limit 2 \
  --max-steps 5 \
  --output-dir output/baselines/classic/random-example
```

Store canonical SatNav-v0.1 paths in the Git-ignored `baselines/classic/.local/env.sh`. After configuration, run:

```bash
bash scripts/classic/eval.sh random val_seen 8
bash scripts/classic/eval.sh reference_follower val_seen 8
bash scripts/classic/eval_parallel.sh random val_seen 2 0,1 8
```

`scripts/classic/eval.sh` uses `SATNAV_MAX_STEPS=5` by default. Set a larger rollout limit explicitly:

```bash
SATNAV_MAX_STEPS=500 bash scripts/classic/eval.sh random val_seen -1
```

### SatNav-v0.1 evaluation settings

The versioned SatNav-v0.1 settings are under `configs/benchmark/`:

| Split | Purpose | Episodes | Maximum steps | Configuration |
| --- | --- | ---: | ---: | --- |
| `val_seen` | Smoke | 4,574 | 5 | `satnav_v0_1_val_seen_smoke.json` |
| `val_seen` | Official | 4,574 | 500 | `satnav_v0_1_val_seen.json` |
| `val_unseen` | Smoke | 8,756 | 5 | `satnav_v0_1_val_unseen_smoke.json` |
| `val_unseen` | Official | 8,756 | 500 | `satnav_v0_1_val_unseen.json` |

Use the smoke settings to confirm that data, model, and environment rollouts work. Use the official settings when reporting benchmark results. Select the corresponding scale with `--split`, `--limit`, and `--max-steps`.

Seq2Seq and CMA also require local checkpoint and vocabulary paths. See [`baselines/classic/local.env.example`](../../../baselines/classic/local.env.example) for the available variables.

Released StreamVLN, NaVILA, Uni-NaVid, and OpenFly checkpoints are available in
the [SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo).

## 3. Episode selection and sharding

Each Episode has the stable key:

```text
<split>::<scene_id>::<episode_id>
```

Evaluation processes Episodes in this order:

1. sort by stable key;
2. apply `offset`;
3. apply `limit`;
4. apply stride sharding, where rank `r` receives `selected[r::world_size]`.

`limit=-1` selects all Episodes after the offset, while `limit=0` selects none. The same `base_seed` and Episode key always produce the same Episode seed, regardless of rank count or launch order.

For seven selected Episodes and `world_size=3`:

```text
rank 0: episode 0, 3, 6
rank 1: episode 1, 4
rank 2: episode 2, 5
```

## 4. PolicyAdapter

Every baseline implements three methods:

```python
class PolicyAdapter:
    def reset(self, context): ...
    def act(self, observation): ...
    def close(self): ...
```

`act()` returns `PolicyStep(action=..., info=...)`. The adapter owns recurrent state, KV cache, frame history, action chunking, and tokenization; these model-specific details do not belong in the shared evaluator.

For the complete path from a minimal adapter to a real model with multi-rank and resume support, see [Model Integration](../development/MODEL_INTEGRATION.md).

`EpisodeContext` provides the current Episode, stable key, split, rank and world size, maximum steps, seed, and public environment reference. Adapters must not access private `Env` fields.

## 5. Output format

A single-rank run produces:

```text
output-dir/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

A multi-rank run produces:

```text
output-dir/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
├── rank_00001/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

### 5.1 `episodes.jsonl`

Each line represents one completed or failed Episode. A successful record includes:

| Field | Meaning |
| --- | --- |
| `episode_key` | Stable Episode key |
| `episode_index` | Position in the globally selected Episode list |
| `rank` | Rank that wrote the record |
| `status` | `ok` or `error` |
| `seed` | Episode seed |
| `episode` | Logical scene, Episode, and trajectory identifiers |
| `steps_taken` | Number of primitive actions executed |
| `terminated_by` | Environment termination reason or `max_steps` |
| `initial_metrics` | Scalar metrics after reset |
| `metrics` | Scalar metrics at Episode completion |
| `initial_agent_state` | Initial agent state, when the environment provides it |
| `final_agent_state` | Final agent state, when the environment provides it |
| `action_trace` | Optional sequence of actions and policy information |

Non-scalar map and image metrics are omitted from JSONL. The machine-local `scene_path` is not serialized by default.

When an Episode fails, SatNav records a bounded error type and generic message. It does not write a traceback or exception content that may reveal local paths.

### 5.2 `done.json`

After finishing its local shard, each rank atomically writes:

```json
{
  "schema_version": 1,
  "rank": 0,
  "status": "complete",
  "expected_count": 2,
  "record_count": 2,
  "error_count": 0
}
```

The aggregator uses this file to read rank completion status and record counts.

### 5.3 `summary.json`

An aggregated result has the form:

```json
{
  "schema_version": 1,
  "status": "complete",
  "expected_episode_count": 2,
  "record_count": 2,
  "unique_record_count": 2,
  "ok_episode_count": 2,
  "error_episode_count": 0,
  "metrics": {
    "success": 0.5,
    "spl": 0.4
  }
}
```

For each metric, `summary.json` reports the arithmetic mean of finite values from records with `status=ok`. If an Episode key appears more than once, the aggregator keeps the last record and exposes the difference through `record_count` and `unique_record_count`.

## 6. Resume

SatNav does not append to an existing output directory unless `--resume` is explicit:

```bash
python -m baselines.classic ... --output-dir output/run --resume
```

Resume performs these steps:

1. read the current rank's `episodes.jsonl`;
2. truncate an incomplete final JSON line left by an interrupted process;
3. collect existing `episode_key` values;
4. skip existing keys and run only the remaining Episodes;
5. rewrite `done.json` and, for a single-rank run, aggregate `summary.json` again.

Reuse an output directory only when data, model, seed, `world_size`, and every other run condition remain unchanged. Use a new output directory after changing experiment conditions.

A complete but invalid JSONL line remains an error and is not silently removed.

## 7. Multi-rank evaluation and aggregation

All ranks must use the same:

- Episode file and split;
- `offset`, `limit`, and `world_size`;
- `base_seed` and `max_steps`;
- output directory.

After all ranks finish, run:

```bash
python scripts/evaluation/aggregate.py output/run
```

To fail the command when any Episode has an error:

```bash
python scripts/evaluation/aggregate.py \
  output/run \
  --fail-on-episode-error
```

The aggregator reads the `rank_XXXXX` directories that exist under the output directory. Launchers must wait for every worker to finish successfully before aggregation; the aggregator does not infer which rank directory is missing.

## 8. Error handling

By default, an Episode exception is persisted as `status=error`, and evaluation continues with the remaining Episodes.

- `--fail-fast` stops the worker immediately after writing the current error record;
- `--fail-on-episode-error` completes rollout and aggregation, then makes the command fail if any error record exists;
- with neither option, the result status is `completed_with_errors` and a summary is still produced.

## 9. Classic CLI

For the full SatNav-v0.1 workflow covering data preparation, Seq2Seq and CMA training, and single- or multi-GPU evaluation, see [Classic Baselines](../training/CLASSIC.md).

All four methods use the same CLI:

```bash
python -m baselines.classic --help
```

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--method` | `random`, `reference_follower`, `seq2seq`, or `cma` |
| `--config` | Baseline YAML; defaults to the configuration for the selected method |
| `--split` | Dataset split |
| `--output-dir` | Output directory |
| `--offset` / `--limit` | Selection range after global sorting |
| `--rank` / `--world-size` | Stride-sharding parameters |
| `--seed` | Base seed |
| `--max-steps` | Maximum primitive actions per Episode |
| `--resume` | Skip existing Episode keys |
| `--checkpoint` / `--vocab` | Seq2Seq or CMA model inputs |
| `--set KEY=VALUE` | Repeatable OmegaConf override |
| `--print-config` | Resolve and print configuration without creating a dataset or model |

## 10. VLM entry points

The four VLM integrations use separate environments but share the same selection and result arguments:

```bash
python -m baselines.vlm.streamvln.evaluate --help
python -m baselines.vlm.navila.evaluate --help
python -m baselines.vlm.uninavid.evaluate --help
python -m baselines.vlm.openfly.evaluate --help
```

For a five-step smoke rollout, pass `--max-steps 5`:

```bash
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path /path/to/model \
  --episodes /path/to/all_episodes.json \
  --scenes-dir /path/to/scenes \
  --limit 1 \
  --max-steps 5 \
  --output-dir output/baselines/vlm/streamvln/smoke
```

Each VLM guide explains its model path, upstream checkout, processor, isolated Python environment, training, and single- or multi-GPU evaluation:

- [StreamVLN Baseline](../training/vlm/STREAMVLN.md)
- [NaVILA Baseline](../training/vlm/NAVILA.md)
- [Uni-NaVid Baseline](../training/vlm/UNINAVID.md)
- [OpenFly Baseline](../training/vlm/OPENFLY.md)

## 11. Python API

A minimal integration looks like this:

```python
from satnav.evaluation import (
    EvaluationConfig,
    Evaluator,
    PolicyAdapter,
    PolicyStep,
)


class StopPolicy(PolicyAdapter):
    def reset(self, context):
        self.context = context

    def act(self, observation):
        return PolicyStep(action=0, info={"reason": "example"})

    def close(self):
        pass


summary = Evaluator(
    environment=env,
    policy=StopPolicy(),
    config=EvaluationConfig(
        output_dir="output/example",
        split="test",
        policy_id="stop-policy",
        limit=2,
        max_steps=5,
    ),
).run()
```

`satnav.evaluation` does not import PyTorch or any trainer. Only a concrete baseline factory or adapter should load its model framework.

## 12. Troubleshooting

### Why does evaluation reject an existing output directory?

The directory already contains rank records or a completion marker. Add `--resume` only when continuing the same run; otherwise, use a new output directory.

### Why was an Episode not rerun after resume?

The current rank's JSONL already contains the same `episode_key`. To rerun it, use a new output directory instead of manually combining JSONL files from different runs.

### Why is the Episode count in a multi-rank summary incorrect?

Confirm that all workers use the same `world_size` and output directory, and run aggregation only after every worker has completed.

### Why is there no top-down map in the results?

The shared JSONL format retains scalar metrics only. Save image and map payloads through a separate video or visualization hook.

### Which LandmarkSet threshold should I use: 3 m or 30 m?

Online evaluation uses the 30 m threshold in `configs/satnav_eval_task.yaml`. The 3 m threshold in trajectory-generation configurations is only for determining whether an expert waypoint has been reached.
