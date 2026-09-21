# SatNav 模型接入

本文介绍如何将新的导航模型接入 SatNav，并通过统一在线评测接口运行 Episode。接入层不
限制模型框架：PyTorch、Transformers、远程推理服务或规则策略都可以实现相同的
`PolicyAdapter` 接口。

开始前建议先阅读 [Core API](../core/CORE_API.md) 和 [Evaluation](../evaluation/README.md)。前者介绍
observation、action 与 `Env`，后者介绍 Episode 选择、结果格式和聚合方式。

环境、策略和评测器的职责及交互闭环见[系统全景](../concepts/OVERVIEW.md)。

## 1. 准备环境

在模型自己的 Python 环境中安装 SatNav Core：

```bash
python -m pip install -e .
python -c "from satnav.evaluation import Evaluator, PolicyAdapter; print('SatNav import OK')"
```

不同模型可以使用不同的 PyTorch、Transformers 或 CUDA 依赖。SatNav 只要求模型环境能够
导入 `satnav`；模型依赖不需要安装到 Core 环境中。

首次接入建议使用仓库自带的合成数据和场景：

```text
applications/resources/
├── map.tif
├── satnav_example_episodes.json
└── satnav_example_task.yaml
```

这组资源不需要下载额外数据。接入代码能够在示例资源上完成 rollout 后，再切换到真实
Episode 和 GeoTIFF 场景。真实数据准备流程参阅 [Episode 数据下载](../dataset/DATA_DOWNLOAD.md)和
[卫星场景下载](../applications/MAP_DOWNLOAD.md)。

## 2. 接入接口

一次评测的调用关系如下：

```text
Episode
  → Env.reset_to_episode()
  → observation
  → PolicyAdapter.act()
  → PolicyStep
  → Env.step()
  → metrics and result
```

模型只需实现三个方法：

```python
class PolicyAdapter:
    def reset(self, context): ...
    def act(self, observation): ...
    def close(self): ...
```

| 方法 | 调用时机 | 责任 |
| --- | --- | --- |
| `reset(context)` | 每个 Episode 开始时 | 清空历史状态并读取当前 Episode 信息 |
| `act(observation)` | 每个环境 step 前 | 返回一个 SatNav primitive action |
| `close()` | 整个 worker 结束时 | 释放模型、文件句柄或外部服务连接 |

`PolicyAdapter` 是 Python Protocol，不要求继承某个基类。只要对象提供这三个方法，就可以
传给 `Evaluator`。

## 3. 最小 Adapter

下面的 adapter 在每个 Episode 中前进三步，然后执行 `STOP`。它不加载模型，但完整演示
了 Episode 状态重置、动作返回和资源释放。

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

同一个 adapter 会连续处理多个 Episode，因此所有
Episode 级状态都应在 `reset()` 中重新初始化。

## 4. `EpisodeContext`

`reset()` 接收的 `EpisodeContext` 提供当前 Episode 和评测进程信息：

| 字段 | 说明 |
| --- | --- |
| `episode` | 当前 `VLNEpisode` |
| `episode_key` | `<split>::<scene_id>::<episode_id>` 稳定标识 |
| `split` | 当前数据划分 |
| `episode_index` | 当前 Episode 在全局选择结果中的位置 |
| `rank` / `world_size` | 当前 worker 和总 worker 数 |
| `max_steps` | 当前 Episode 最大动作数 |
| `seed` | 当前 Episode 的确定性随机种子 |
| `environment` | 当前公开 `Env` 对象 |
| `simulator` | `environment.simulator` 的便捷属性 |
| `agent_state` | `environment.agent_state` 的便捷属性 |

需要随机采样时，应在每次 `reset()` 中使用 `context.seed` 初始化 Episode 级随机数生成器。
这样同一 Episode 在不同 rank 数量或 resume 边界下仍能得到相同随机序列。

Adapter 可以读取 `context.episode.instruction`、`reference_path` 或 `trajectory_type`，也可以
通过 `context.agent_state` 获取当前位置。不要访问 `Env` 的 `_dataset`、`_task` 或 `_sim`
等私有属性。

## 5. Observation 与 Action

默认 `VLNTask` 返回：

| Key | 类型 | 说明 |
| --- | --- | --- |
| `rgb` | `numpy.ndarray` | `(H, W, 3)`、`uint8` 的卫星图 observation |
| `instruction` | `dict` | `{"text": str}` |
| `agent_pose` | `numpy.ndarray` | `(4,)` 的相对位置与朝向 |

RGB 尺寸由 task config 决定。预处理代码应读取输入的实际 shape，不要固定为 224 或 448。
完整 observation 定义参阅 [Core API - Observation](../core/CORE_API.md#6-observation)。

SatNav action 为：

| ID | Action |
| ---: | --- |
| 0 | `STOP` |
| 1 | `MOVE_FORWARD` |
| 2 | `TURN_LEFT` |
| 3 | `TURN_RIGHT` |

`PolicyStep.action` 可以使用动作字符串或整数 ID。建议 adapter 在返回前完成模型输出到
SatNav action 的显式映射：

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

不要将无法解析的模型输出静默转换为 `STOP`，否则会把模型格式错误记录成正常终止并影响
评测指标。

`PolicyStep.info` 可保存生成文本、置信度、action queue 长度等诊断信息。这里的内容会在
启用 action trace 时写入 JSONL，因此应满足以下要求：

- 可以序列化为 JSON；
- 不包含完整 RGB、模型 tensor 或其他大对象；
- 不包含 checkpoint、数据集或场景的本机绝对路径；
- 不包含 token、凭据或远程服务请求头。

## 6. 运行最小评测

将第 3 节的 `ForwardThenStopAdapter` 与内置示例环境连接：

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

`Evaluator.run()` 会在成功或异常结束时关闭 policy 和 environment。运行结果位于：

```text
output/model_integration/forward_then_stop/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

首先检查 `summary.json` 中的 `status` 和 `error_episode_count`，再查看 JSONL 中单个 Episode
的 action trace 与 metrics。结果字段的完整定义参阅 [Evaluation - 输出格式](../evaluation/README.md#5-输出格式)。

## 7. 接入真实模型

真实模型通常在 adapter 创建时加载一次，在多个 Episode 之间复用。只有 recurrent state、
历史图像、prompt 和待执行 action queue 等 Episode 级状态需要在 `reset()` 中清空。

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
        # 在这里导入模型框架并加载 checkpoint。
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

示例中的 `load_your_model()`、`reset_model_state()`、`run_model()` 和
`parse_model_actions()` 由具体模型实现。接入时应重点确认：

- checkpoint 加载完整，模型进入 evaluation mode；
- 推理过程关闭梯度计算；
- RGB 的颜色顺序、尺寸和归一化与模型训练时一致；
- instruction 使用当前 Episode，而不是上一个 Episode 的缓存；
- recurrent/KV cache、历史帧和 action queue 在每次 `reset()` 中清空；
- action chunk 按 primitive action 逐步返回，不在一次 `act()` 中直接执行多个环境 step；
- `close()` 可以重复调用。

## 8. 项目组织

模型可以保存在独立仓库中，只依赖 SatNav 的公开接口：

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

如果模型作为 SatNav 仓库内维护的 baseline，建议使用：

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

公开配置只保存相对路径和模型无关参数。checkpoint、Episode、GeoTIFF、上游源码路径和
输出目录应通过 CLI、环境变量或 Git ignored 的 `.local/env.sh` 提供。

模型依赖应在 `adapter.py` 或模型 factory 内延迟导入。仅执行
`import satnav` 或 `import satnav.evaluation` 时，不应加载 PyTorch、Transformers 或模型
checkpoint。

## 9. 评测入口

一个模型评测入口通常按以下顺序工作：

1. 解析模型、Episode、场景和输出路径；
2. 加载 task config，并设置 split、`DATASET.DATA_PATH` 和 `DATASET.SCENES_DIR`；
3. 创建 `SatNavDataset` 和 `Env(cycle=False)`；
4. 加载模型并创建 adapter；
5. 构造 `EvaluationConfig`；
6. 调用 `Evaluator.run()`；
7. 输出 `summary.json` 中的汇总结果。

建议为所有模型提供以下公共参数：

| 参数 | 作用 |
| --- | --- |
| `--model-path` | checkpoint 或模型目录 |
| `--task-config` | SatNav task config |
| `--episodes` | 当前 split 的 Episode JSON |
| `--scenes-dir` | GeoTIFF 场景目录 |
| `--split` | `val_seen`、`val_unseen` 或 `test` |
| `--output-dir` | 评测输出目录 |
| `--offset` / `--limit` | Episode 选择范围 |
| `--rank` / `--world-size` | 多进程 Episode 分片 |
| `--base-seed` | 评测随机种子 |
| `--max-steps` | 每个 Episode 最大动作数 |
| `--resume` | 跳过当前 rank 已完成的 Episode |
| `--fail-on-episode-error` | 任一 Episode 失败时令最终命令失败 |
| `--dry-run` | 只检查配置和 Episode 选择，不加载模型 |

可以参考现有入口的组织方式：

- `baselines/classic/evaluate.py`：Classic factory 与统一 evaluator；
- `baselines/vlm/navila/evaluate.py`：外部模型、独立环境和 GPU 选择；
- `baselines/vlm/openfly/evaluate.py`：本地 checkpoint 与 processor 加载。

`--dry-run` 应在模型加载前结束，使开发者可以先检查 Episode 数、rank shard 和输出目录。

## 10. 多 rank 评测

在线评测使用 Episode 分片，每个 rank 独立加载模型并运行自己的 Episode。模型不需要通过
DDP 包装，但所有 rank 必须使用相同的数据、split、选择参数、seed、最大步数和输出目录。

下面是两张 GPU 的启动示例；`my_satnav_model.evaluate` 表示模型自己的 CLI：

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

`limit` 先应用于全局排序结果，再进行 stride sharding。例如 `limit=8`、`world_size=2` 时，
两个 rank 各处理四个 Episode。完整规则参阅 [Evaluation - Episode 选择与分片](../evaluation/README.md#3-episode-选择与分片)。

## 11. Resume 与错误处理

同一运行中断后，可以使用相同参数和输出目录重新启动，并设置 `resume=True` 或 CLI 的
`--resume`。Evaluator 会跳过当前 rank JSONL 中已有的 `episode_key`。

只有模型、数据、split、rank 数量、seed 和评测参数保持一致时才应 resume。更改运行条件
时应使用新的输出目录。

调试新 adapter 时建议同时启用：

```text
fail_fast=True
fail_on_episode_error=True
capture_action_trace=True
```

`fail_fast` 在记录首个失败 Episode 后停止当前 worker；`fail_on_episode_error` 让最终命令
在存在失败记录时返回非零状态；action trace 用于查看模型输出经过 adapter 后实际执行的
primitive action。

大规模正式评测确认接入稳定后，可以设置 `capture_action_trace=False` 减少结果文件体积。


## 12. 常见问题

### 模型应该从哪里读取 instruction？

可以读取 `observation["instruction"]["text"]`，也可以在 `reset()` 中读取
`context.episode.instruction.instruction_text`。前者适合逐 step 统一预处理，后者适合构建
Episode 级 prompt。

### 模型一次生成多个动作怎么办？

将生成结果解析为 action queue。当前 `act()` 返回第一个 primitive action，后续调用依次
返回剩余动作；每次 `Env.step()` 后仍应更新历史 observation 和模型状态。

### 必须把模型代码放进 SatNav 仓库吗？

不需要。外部包只要安装 SatNav 并实现 `PolicyAdapter` 即可。只有希望由 SatNav 仓库直接
维护和发布的 baseline，才需要放入 `baselines/`。

### Adapter 可以直接控制 simulator 吗？

通常不需要。Adapter 应返回 action，由 Evaluator 调用 `Env.step()`，这样 Episode 终止、
metrics 和 action trace 才会同步更新。只在 navigation helper 需要读取底层状态时使用
`context.simulator` 的公开接口。

### 为什么模型已经输出 STOP，但 Success 仍为 0？

`STOP` 只表示模型决定结束 Episode。只有执行 STOP 时 agent 位于当前任务类型的成功半径
内，`SUCCESS` 才为 1。标准阈值参阅 [Evaluation](../evaluation/README.md)。
