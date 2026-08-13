# SatNav OpenFly Baseline

本文介绍如何使用 SatNav-v0.1 trajectory 训练 OpenFly，并在 SatSim 中完成单卡或多卡在线
评测。OpenFly 使用独立 Python 环境；其适配后的模型运行时代码已包含在 SatNav 中，不需要
安装外部 OpenFly package，也不与 SatNav Core、Classic 或其他 VLM baseline 共用环境。

开始前建议先阅读[环境安装](../../getting-started/INSTALLATION.md)、[模型训练](../README.md)和
[统一评测](../../evaluation/README.md)。

## 1. 模型与 SatNav 接口

OpenFly 根据导航指令、当前 RGB observation、前两帧 observation 和历史动作生成下一步
动作。SatNav adapter 将模型输出转换为四个 primitive action：

| 模型输出 | SatNav action | 环境行为 |
| --- | --- | --- |
| `stop` | `STOP` | 结束当前 Episode |
| `forward` | `MOVE_FORWARD` | 前进 10 m |
| `left` | `TURN_LEFT` | 左转 15° |
| `right` | `TURN_RIGHT` | 右转 15° |

每次推理严格使用三帧，顺序为当前帧、前一帧、前二帧；Episode 开始阶段缺少历史帧时复用
当前可用帧。Prompt 最多保留最近 16 个历史动作。

OpenFly 支持两种 checkpoint action format：

- `compact`：模型直接生成 `stop`、`forward`、`left` 或 `right`；
- `original`：模型生成 OpenFly 原有的 8 维 action token，再映射到相同的四个 SatNav
  primitive action。

评测默认使用 `auto`，按照 checkpoint 中记录的设置选择 action format。训练或评测时不能
将一个 checkpoint 强制解释为另一种 format。

SatNav 中的 OpenFly 运行时代码基于
[`SHAILAB-IPEC/OpenFly-Platform@c075075497a7122bad82f5b76b9be926ad5a81b3`](https://github.com/SHAILAB-IPEC/OpenFly-Platform/commit/c075075497a7122bad82f5b76b9be926ad5a81b3)，
并参考固定版本的 SwiftVLN adapter 保留三帧和 prompt 行为。使用模型代码和权重前，请阅读
[UPSTREAM](../../../../baselines/vlm/openfly/UPSTREAM.md)、
[NOTICE](../../../../baselines/vlm/openfly/NOTICE) 和
[Upstream License](../../../../baselines/vlm/openfly/LICENSE.upstream)。

## 2. 准备独立环境

在 SatNav 仓库根目录运行环境脚本：

```bash
bash baselines/vlm/openfly/scripts/bootstrap_env.sh satnav-openfly
conda activate satnav-openfly
export PYTHONNOUSERSITE=1
```

脚本会创建 Python 3.10 环境，安装 PyTorch 2.3.0、CUDA 12.1 对应 wheel、OpenFly 固定
依赖和 SatNav，并在结束前检查依赖版本。不要在已有的 VILA、OpenFly 或其他 VLM 环境中运行
训练与评测。

确认关键依赖和入口：

```bash
python - <<'PY'
import accelerate
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m pip check
python -m baselines.vlm.openfly.dataset --help
python -m baselines.vlm.openfly.trainer --help
python -m baselines.vlm.openfly.evaluate --help
python -m baselines.vlm.openfly.checkpoint --help
command -v python
```

记录 `command -v python` 的输出，后续将其设置为 `OPENFLY_PYTHON`。

## 3. OpenFly 代码来源

训练和评测直接使用 `baselines/vlm/openfly/` 中随 SatNav 发布的固定实现，不需要 clone 或
安装 OpenFly-Platform，也不要将另一个 OpenFly checkout 添加到 `PYTHONPATH`。

如需进行源码对照，可以在 SatNav 仓库之外准备一个只读 checkout：

```bash
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git \
  /path/to/OpenFly-Platform
git -C /path/to/OpenFly-Platform checkout --detach \
  c075075497a7122bad82f5b76b9be926ad5a81b3
```

该 checkout 不是训练或评测的必需输入。正常使用 OpenFly baseline 时可以跳过本节中的
clone 命令。

## 4. 准备 SatNav 数据

### 4.1 Episode、GeoTIFF 与 Trajectory

按照 [Episode 数据下载](../../dataset/DATA_DOWNLOAD.md)、[卫星场景下载](../../applications/MAP_DOWNLOAD.md)和
[轨迹数据生成](../../applications/TRAJECTORY_GENERATION.md)准备 SatNav-v0.1。OpenFly 训练同时读取
train Episode 文件与 trajectory JPEG：

```text
SatNav-v0.1/
└── episodes/
    └── train/
        └── all_episodes.json

trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            └── ...
```

Episode 文件与 trajectory 必须来自同一版本的数据。OpenFly 会根据 scene、trajectory ID
和 instruction 匹配二者，并使用 Episode 中的 `trajectory_type`。

在当前终端设置数据路径：

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/scenes
export SATNAV_TRAJECTORY_ROOT=/path/to/trajectory_data
export SATNAV_OPENFLY_TRAIN_EPISODES="$SATNAV_DATA_ROOT/episodes/train/all_episodes.json"
```

先检查 Episode 和 59 个 GeoTIFF：

```bash
bash scripts/validation/data_validation.sh
```

### 4.2 校验 Trajectory

对完整 train split 检查 Episode 匹配、action、frame 数量和图片文件：

```bash
mkdir -p output/baselines/vlm/openfly

bash baselines/vlm/openfly/scripts/validate_data.sh \
  "$SATNAV_TRAJECTORY_ROOT" \
  output/baselines/vlm/openfly/data_validation.json
```

正式训练开始前，launcher 会再次检查完整 train split。Episode 无法匹配、缺失或多余图片、
越界路径、action 不合法或 frame 数量不一致都会直接终止训练，不会静默跳过样本。

## 5. 准备模型

SatNav 源码仓库不保存 OpenFly 权重。可以使用 SatNav 已发布 checkpoint、OpenFly 上游
模型或已有训练产物。

### 5.1 SatNav 已发布 checkpoint

[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
提供两个 OpenFly checkpoint：

- [scratch](https://huggingface.co/Eku127/openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5)：从 OpenVLA 与 OpenFly processor 资源开始训练；
- [continue](https://huggingface.co/Eku127/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5)：从 OpenFly Agent checkpoint 继续训练。

将所需 checkpoint 下载到本地模型目录：

```bash
export OPENFLY_MODEL_ROOT=/path/to/openfly-models
python -m huggingface_hub.commands.huggingface_cli download \
  Eku127/openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 \
  --local-dir "$OPENFLY_MODEL_ROOT/openfly-satnav-continue"
```

评测时将下载后的目录传给 `--model-path`。发布目录包含推理所需文件，不包含 optimizer 与
scheduler state。

### 5.2 Continue：完整 HF checkpoint

`continue` 接受一个本地 Hugging Face 格式的完整 OpenFly checkpoint：

```text
openfly-hf-checkpoint/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
├── model.safetensors
└── ...
```

权重也可以由多个 `model-*.safetensors` 与 `model.safetensors.index.json` 组成。模型目录必须
同时包含 processor、tokenizer 和全部权重，不能只提供单独的 adapter 或部分 shard。

### 5.3 Scratch：原生 checkpoint 与 processor

`scratch` 接受 OpenFly 原生 run 目录或其中一个 `.pt` checkpoint：

```text
openfly-native-run/
└── checkpoints/
    ├── step-10000-epoch-1-loss=...pt
    └── ...
```

如果传入 run 目录，launcher 会选择 `checkpoints/` 中 step 最大的 `.pt`。此外还需要一个与
该模型匹配的 processor/tokenizer 目录：

```text
openfly-processor/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

训练时会自动将原生模型转换到当前 OpenFly HF runtime。转换过程需要额外磁盘空间；原始
`.pt` 和 processor 目录不会被修改。

## 6. 配置本地路径

此时环境、SatNav 数据和模型均已准备完成。复制 OpenFly 本地配置模板：

```bash
mkdir -p baselines/vlm/openfly/.local
cp baselines/vlm/openfly/local.env.example \
  baselines/vlm/openfly/.local/env.sh
```

在 `baselines/vlm/openfly/.local/env.sh` 中填写前面步骤得到的实际路径：

```bash
export OPENFLY_PYTHON="${OPENFLY_PYTHON:-/path/to/satnav-openfly/bin/python}"

export OPENFLY_CONTINUE_MODEL="${OPENFLY_CONTINUE_MODEL:-/path/to/openfly-hf-checkpoint}"
export OPENFLY_NATIVE_RUN="${OPENFLY_NATIVE_RUN:-/path/to/openfly-native-run}"
export OPENFLY_PROCESSOR_PATH="${OPENFLY_PROCESSOR_PATH:-/path/to/openfly-processor}"
export OPENFLY_NATIVE_HF_CACHE_DIR="${OPENFLY_NATIVE_HF_CACHE_DIR:-/path/to/openfly-native-hf-cache}"

export SATNAV_OPENFLY_TRAIN_DATA="${SATNAV_OPENFLY_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_OPENFLY_TRAIN_EPISODES="${SATNAV_OPENFLY_TRAIN_EPISODES:-/path/to/SatNav-v0.1/episodes/train/all_episodes.json}"
export SATNAV_OPENFLY_EVAL_EPISODES="${SATNAV_OPENFLY_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_OPENFLY_SCENES_DIR="${SATNAV_OPENFLY_SCENES_DIR:-/path/to/scenes}"
export SATNAV_OPENFLY_OUTPUT="${SATNAV_OPENFLY_OUTPUT:-output/baselines/vlm/openfly}"
```

只使用 `continue` 时，可以不设置 `OPENFLY_NATIVE_RUN`、`OPENFLY_PROCESSOR_PATH` 和
`OPENFLY_NATIVE_HF_CACHE_DIR`。只使用 `scratch` 时，可以不设置
`OPENFLY_CONTINUE_MODEL`。

各变量对应的来源为：

| 变量 | 路径来源 |
| --- | --- |
| `OPENFLY_PYTHON` | 第 2 节 `command -v python` 的输出 |
| `OPENFLY_CONTINUE_MODEL` | 第 5.1 或 5.2 节准备的完整 HF checkpoint |
| `OPENFLY_NATIVE_RUN` | 第 5.3 节准备的原生 run 或 `.pt` |
| `OPENFLY_PROCESSOR_PATH` | 第 5.3 节准备的 processor/tokenizer 目录 |
| `OPENFLY_NATIVE_HF_CACHE_DIR` | 原生模型转换使用的本地缓存目录 |
| `SATNAV_OPENFLY_TRAIN_DATA` | 第 4 节校验通过的 `trajectory_data/` |
| `SATNAV_OPENFLY_TRAIN_EPISODES` | 与 trajectory 对应的 train Episode 文件 |
| `SATNAV_OPENFLY_EVAL_EPISODES` | SatNav-v0.1 eval Episode 路径模板 |
| `SATNAV_OPENFLY_SCENES_DIR` | 第 4 节校验通过的 GeoTIFF 目录 |
| `SATNAV_OPENFLY_OUTPUT` | 训练和评测输出根目录 |

启动脚本会自动读取该文件。临时在终端显式设置的同名环境变量优先于 `.local/env.sh`。
本地数据、模型、输出目录、缓存和环境路径均不应提交到 Git。

如需在当前终端后续命令中直接使用这些变量，加载本地配置：

```bash
source baselines/vlm/openfly/.local/env.sh
```

## 7. 训练配置与 Checkpoint 检查

使用 `continue` 前，先确认完整 checkpoint 可以加载：

```bash
python -m baselines.vlm.openfly.checkpoint \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --device cuda:0 \
  --dtype bfloat16
```

该命令会检查 config、processor、tokenizer 和全部模型权重。`scratch` 的原生 `.pt` 会在训练
启动时完成格式检查和转换。

默认训练参数位于 `baselines/vlm/openfly/configs/train.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| RGB frames per sample | 3 |
| Action format | `compact` |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `2e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 4 |

训练使用 `baselines/vlm/openfly/configs/zero2.json` 中的 DeepSpeed ZeRO-2，并更新 vision
backbone、language model 和 projector。训练完成后，launcher 会重新加载输出并确认三个模型
组件都发生了有效更新。

训练命令中的 `--max-steps` 表示 optimizer step 数；评测命令中的同名参数表示每个 Episode
最多执行的 primitive action 数。

## 8. Smoke 训练

先运行一个 optimizer step，检查数据、模型更新和 checkpoint 保存：

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-smoke \
  --max-episodes 2 \
  --max-samples 4 \
  --max-steps 1 \
  --gpus 1
```

`--max-episodes` 和 `--max-samples` 只限制实际送入 trainer 的 bounded subset；启动前的数据
检查仍覆盖完整 train split。训练成功后，输出目录包含可直接评测的完整模型和
`checkpoint-*`，launcher 还会自动完成严格重载与模型更新检查。

## 9. 完整训练与 Resume

从完整 HF checkpoint 继续训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-v0-1 \
  --gpus 8
```

从原生 OpenFly checkpoint 开始训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend scratch \
  --model-path "$OPENFLY_NATIVE_RUN" \
  --processor-path "$OPENFLY_PROCESSOR_PATH" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/scratch-v0-1 \
  --gpus 8
```

显存不足时优先减小 `--batch-size`，并通过 `--gradient-accumulation` 调整有效 batch size。

训练中断且输出目录中已经存在完整的 `checkpoint-*` 时，使用完全相同的参数和输出目录重新
执行，并添加 `--resume`：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/train.sh \
  --backend continue \
  --model-path "$OPENFLY_CONTINUE_MODEL" \
  --trajectory-root "$SATNAV_OPENFLY_TRAIN_DATA" \
  --output-dir output/baselines/vlm/openfly/train/continue-v0-1 \
  --gpus 8 \
  --resume
```

Resume 只适用于继续同一次训练。更换模型、数据、action format、配置、GPU 数量或采样参数
时，应使用新的输出目录。

## 10. 配置在线评测

设置待评测 checkpoint。训练输出目录本身就是完整 HF checkpoint：

```bash
export OPENFLY_CHECKPOINT=/path/to/trained-openfly-checkpoint
```

检查 checkpoint、Episode、scene 和输出路径能否解析：

```bash
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run 会检查 checkpoint 的配置文件以及评测数据选择，但不加载模型权重，也不执行环境
交互。

## 11. 单 GPU Smoke 评测

选择 4 个 Episode，每个 Episode 最多执行 5 步：

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/5steps-1rank \
  --fail-on-episode-error
```

结果结构为：

```text
<output-dir>/
├── rank_00000/
│   ├── episodes.jsonl
│   └── done.json
└── summary.json
```

确认 `summary.json` 的 `status` 为 `complete`、`error_episode_count` 为 0，并检查
`distance_to_goal`、`success`、`oracle_success`、`spl` 和 `path_length`。

## 12. 多 GPU 评测

每个 rank 都读取完整 Episode 文件，统一 evaluator 会完成稳定排序和 stride sharding，不需要
提前拆分数据。两张 GPU 的 smoke 命令为：

```bash
CUDA_DEVICES=0,1 \
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

多 GPU 评测必须显式设置 `--output-dir`。所有 rank 成功退出后，launcher 会自动生成合并后的
`summary.json`。

## 13. 完整评测

完整 `val_seen` 评测使用 500 步上限和全部 Episode：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/openfly/scripts/eval.sh \
  --model-path "$OPENFLY_CHECKPOINT" \
  --episodes "$SATNAV_OPENFLY_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_OPENFLY_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/openfly/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

完成后，将 split 和输出目录改为 `val_unseen` 再运行一次。
`SATNAV_OPENFLY_EVAL_EPISODES` 中的 `{split}` 会自动替换为当前 split。

评测中断后，可以使用完全相同的参数和输出目录重新执行，并在命令末尾添加 `--resume`。
Resume 会跳过当前 rank 已经写入的 Episode。更换 checkpoint、数据、seed、GPU 数量或评测
参数时，应使用新的输出目录。

## 14. 常见问题

### 为什么找不到 OpenFly Python 环境？

确认 `OPENFLY_PYTHON` 指向第 2 节创建环境中的可执行文件，并检查：

```bash
"$OPENFLY_PYTHON" -c "import torch, transformers; print(torch.__version__)"
```

不要将该变量指向 SatNav Core、VILA 或其他 VLM 环境。

### 为什么 continue checkpoint 无法加载？

确认模型目录包含完整的 config、processor、tokenizer 和全部 safetensors 权重。OpenFly 不接受
PyTorch `.bin`、单独 adapter、缺失 shard 或仅包含 optimizer state 的目录。

### 为什么 scratch 找不到 processor？

通过 `--processor-path` 或 `OPENFLY_PROCESSOR_PATH` 指向与原生模型匹配的完整 processor
目录。该目录至少需要包含模型 config、image processor 和 tokenizer 文件。

### 为什么 action format 不匹配？

`compact` 与 `original` 使用不同的输出和训练目标。评测时保持默认 `auto` 即可；如果显式
设置 `--action-format`，其值必须与 checkpoint 中声明的 format 相同。

### 为什么数据校验提示 Episode 无法匹配？

确认 `SATNAV_OPENFLY_TRAIN_EPISODES` 与 trajectory 来自同一 SatNav-v0.1 train release，且
instruction 没有被重新改写。不要混用不同版本的 Episode 和 trajectory export。

### 为什么训练启动后长时间没有加载模型？

训练会先检查完整 SatNav-v0.1 trajectory。数据位于网络存储或低速磁盘时，该阶段可能需要
较长时间；建议将 Episode、annotation 和 JPEG 放在本地高速存储。

### 为什么训练 Resume 被拒绝？

确认输出目录中存在完整的 `checkpoint-*`，并使用与首次训练完全相同的 backend、模型、数据、
action format、配置、GPU 数量和采样参数。需要改变实验设置时，请创建新的输出目录。

### 为什么重新评测时没有执行部分 Episode？

使用 `--resume` 时，evaluator 会跳过当前输出目录中已经完成的 Episode。只有继续完全相同
的评测时才应复用该目录；其他情况请指定新的 `--output-dir`。
