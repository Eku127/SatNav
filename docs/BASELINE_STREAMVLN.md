# SatNav StreamVLN Baseline

本文介绍如何使用 SatNav-v0.1 trajectory 训练 StreamVLN，并在 SatSim 中完成单卡或多卡
在线评测。StreamVLN 使用独立 Python 环境和外部模型代码，不与 SatNav Core、Classic 或其他
VLM baseline 共用环境。

开始前建议先阅读[环境安装](INSTALLATION.md)、[模型训练](TRAINING.md)和
[统一评测](EVALUATION.md)。

## 1. 模型与 SatNav 接口

StreamVLN 根据导航指令、当前 RGB observation 和历史视觉信息生成符号动作。SatNav adapter
将模型输出转换为四个 primitive action：

| 模型输出 | SatNav action | 环境行为 |
| --- | --- | --- |
| `STOP` | `STOP` | 结束当前 Episode |
| `↑` | `MOVE_FORWARD` | 前进 10 m |
| `←` | `TURN_LEFT` | 左转 15° |
| `→` | `TURN_RIGHT` | 右转 15° |

模型一次可以生成一组动作，adapter 将动作放入队列，再按照 SatNav 的逐步交互接口依次执行。
默认每 32 个 observation 建立一个视觉窗口，保留 8 个历史采样，并使用 4 个 future action
进行训练。

SatNav 当前适配的上游版本为
[`Eku127/StreamVLN@60476e81f4c01b29f1a51a7469f1cb4addbc1d62`](https://github.com/Eku127/StreamVLN/commit/60476e81f4c01b29f1a51a7469f1cb4addbc1d62)。
使用模型代码和权重前，请阅读 [UPSTREAM](../baselines/vlm/streamvln/UPSTREAM.md) 和
[NOTICE](../baselines/vlm/streamvln/NOTICE)。

## 2. 准备独立环境

在 SatNav 仓库根目录创建 StreamVLN 环境：

```bash
conda env create -f baselines/vlm/streamvln/environment/conda.yml
conda activate satnav-streamvln

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装与 Python 3.9、PyTorch 2.5、CUDA 和本机 ABI 匹配的 FlashAttention wheel：

```bash
export STREAMVLN_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl
test -f "$STREAMVLN_FLASH_ATTN_WHEEL"
python -m pip install "$STREAMVLN_FLASH_ATTN_WHEEL"
```

继续安装其余依赖和 SatNav：

```bash
python -m pip install -r baselines/vlm/streamvln/requirements.txt
python baselines/vlm/streamvln/scripts/normalize_decord_wheel.py
python -m pip install -e .
python -m pip check
```

上述脚本用于修复 Decord 0.6.0 官方 wheel 的已知安装元数据问题，不会修改视频解码实现。

检查关键依赖和入口：

```bash
python - <<'PY'
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m baselines.vlm.streamvln.dataset --help
python -m baselines.vlm.streamvln.trainer --help
python -m baselines.vlm.streamvln.evaluate --help
command -v python
```

记录 `command -v python` 的输出，后续将其设置为 `STREAMVLN_PYTHON`。

## 3. 准备上游代码

将 StreamVLN clone 到 SatNav 仓库之外，并 checkout 到适配版本：

```bash
git clone https://github.com/Eku127/StreamVLN.git /path/to/StreamVLN
git -C /path/to/StreamVLN checkout \
  60476e81f4c01b29f1a51a7469f1cb4addbc1d62
```

训练和评测启动时会检查当前 revision。不要在同一个目录中切换到其他 StreamVLN 版本后
继续使用已有输出。

## 4. 准备 SatNav 数据

### 4.1 Episode、GeoTIFF 与 Trajectory

按照 [Episode 数据下载](DATA_DOWNLOAD.md)、[卫星场景下载](APPLICATION_MAP_DOWNLOAD.md)和
[轨迹数据生成](APPLICATION_TRAJ_GENERATION.md)准备 SatNav-v0.1。StreamVLN 使用与其他
baseline 相同的 trajectory export：

```text
trajectory_data/
├── annotations.json
└── images/
    └── <episode>/
        └── rgb/
            ├── 001.jpg
            └── ...
```

在当前终端设置数据路径：

```bash
export SATNAV_DATA_ROOT=/path/to/SatNav-v0.1
export SATNAV_SCENES_DIR=/path/to/scenes
export SATNAV_TRAJECTORY_ROOT=/path/to/trajectory_data
```

先检查 Episode 和 59 个 GeoTIFF：

```bash
bash scripts/validation/data_validation.sh
```

### 4.2 校验 Trajectory

对完整 train split 运行数据校验：

```bash
mkdir -p output/baselines/vlm/streamvln

python scripts/validation/validate_trajectory_output.py \
  --annotations "$SATNAV_TRAJECTORY_ROOT/annotations.json" \
  --output-root "$SATNAV_TRAJECTORY_ROOT" \
  --source-episodes "$SATNAV_DATA_ROOT/episodes/train/all_episodes.json" \
  --generation-config applications/episode_processing/configs/trajectory_generation.yaml \
  --scenes-dir "$SATNAV_SCENES_DIR" \
  --expected-count 105164 \
  --decode-images \
  --report output/baselines/vlm/streamvln/data_validation.json
```

然后检查 StreamVLN 的 32-frame 分段、action 和 frame 对齐：

```bash
python -m baselines.vlm.streamvln.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --num-frames 32 \
  --strict-frames
```

校验成功后会输出可用 Episode 和训练 sample 数量。正式训练前应解决缺失图片、越界路径、
action 不合法或 frame 数量不足等错误。

## 5. 下载模型

选择模型保存目录：

```bash
export STREAMVLN_MODEL_ROOT=/path/to/streamvln-models
mkdir -p "$STREAMVLN_MODEL_ROOT"
```

`continue` 训练使用官方 StreamVLN checkpoint：

```bash
bash baselines/vlm/streamvln/scripts/download.sh \
  --model-root "$STREAMVLN_MODEL_ROOT"
```

`scratch` 训练需要 LLaVA-Video base model；该目录也可以为不包含 tokenizer 的 checkpoint
提供 tokenizer。训练和评测都需要本地 SigLIP vision tower：

```bash
bash baselines/vlm/streamvln/scripts/download.sh \
  --repo lmms-lab/LLaVA-Video-7B-Qwen2 \
  --model-root "$STREAMVLN_MODEL_ROOT"

bash baselines/vlm/streamvln/scripts/download.sh \
  --repo google/siglip-so400m-patch14-384 \
  --model-root "$STREAMVLN_MODEL_ROOT"
```

默认下载到 `STREAMVLN_MODEL_ROOT`。需要使用 Hugging Face mirror 时添加
`--endpoint https://hf-mirror.com`。也可以安装 ModelScope 后添加 `--source modelscope`。

下载完成后，模型目录通常为：

```text
streamvln-models/
├── StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/
├── LLaVA-Video-7B-Qwen2/
└── siglip-so400m-patch14-384/
```

其中：

- `LLaVA-Video-7B-Qwen2/` 是 scratch 训练使用的 base model，也可以为不包含 tokenizer
  的 checkpoint 提供 tokenizer；
- `siglip-so400m-patch14-384/` 是训练和评测使用的 vision tower；
- `StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3/` 是 continue 训练的起点。

## 6. 配置本地路径

此时环境、上游代码、SatNav 数据和模型目录均已准备完成。复制 StreamVLN 本地配置模板：

```bash
mkdir -p baselines/vlm/streamvln/.local
cp baselines/vlm/streamvln/local.env.example \
  baselines/vlm/streamvln/.local/env.sh
```

在 `baselines/vlm/streamvln/.local/env.sh` 中填写前面步骤得到的实际路径：

```bash
export STREAMVLN_REPO="${STREAMVLN_REPO:-/path/to/StreamVLN}"
export STREAMVLN_PYTHON="${STREAMVLN_PYTHON:-/path/to/satnav-streamvln/bin/python}"
export STREAMVLN_MODEL_ROOT="${STREAMVLN_MODEL_ROOT:-/path/to/streamvln-models}"
export STREAMVLN_TOKENIZER_PATH="${STREAMVLN_TOKENIZER_PATH:-/path/to/streamvln-models/LLaVA-Video-7B-Qwen2}"
export STREAMVLN_VISION_TOWER="${STREAMVLN_VISION_TOWER:-/path/to/streamvln-models/siglip-so400m-patch14-384}"

export SATNAV_STREAMVLN_TRAIN_DATA="${SATNAV_STREAMVLN_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_STREAMVLN_EVAL_EPISODES="${SATNAV_STREAMVLN_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_STREAMVLN_SCENES_DIR="${SATNAV_STREAMVLN_SCENES_DIR:-/path/to/scenes}"
export SATNAV_STREAMVLN_OUTPUT="${SATNAV_STREAMVLN_OUTPUT:-output/baselines/vlm/streamvln}"
```

各变量对应的来源为：

| 变量 | 路径来源 |
| --- | --- |
| `STREAMVLN_REPO` | 第 3 节 clone 的 StreamVLN 目录 |
| `STREAMVLN_PYTHON` | 第 2 节 `command -v python` 的输出 |
| `STREAMVLN_MODEL_ROOT` | 第 5 节选择的模型保存目录 |
| `STREAMVLN_TOKENIZER_PATH` | 下载后的 `LLaVA-Video-7B-Qwen2/` |
| `STREAMVLN_VISION_TOWER` | 下载后的 `siglip-so400m-patch14-384/` |
| `SATNAV_STREAMVLN_TRAIN_DATA` | 第 4 节校验通过的 `trajectory_data/` |
| `SATNAV_STREAMVLN_EVAL_EPISODES` | SatNav-v0.1 eval Episode 路径模板 |
| `SATNAV_STREAMVLN_SCENES_DIR` | 第 4 节校验通过的 GeoTIFF 目录 |
| `SATNAV_STREAMVLN_OUTPUT` | 训练和评测输出根目录 |

启动脚本会自动读取该文件。临时在终端显式设置的同名环境变量优先于 `.local/env.sh`。
本地数据、模型、上游代码、输出目录和环境路径均不应提交到 Git。

如需在当前终端后续命令中直接使用这些变量，加载本地配置：

```bash
source baselines/vlm/streamvln/.local/env.sh
```

## 7. 训练配置

默认训练参数位于 `baselines/vlm/streamvln/configs/train.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| Frames per window | 32 |
| History samples | 8 |
| Future actions | 4 |
| Epochs | 1 |
| Per-device batch size | 3 |
| Gradient accumulation | 2 |
| Learning rate | `2e-5` |
| Vision tower learning rate | `5e-6` |

训练使用 `baselines/vlm/streamvln/configs/zero2.json` 中的 DeepSpeed ZeRO-2，并更新
vision tower、multimodal projector 和 language model。常用参数可以直接通过 `train.sh`
覆盖，不需要修改公共配置。

训练命令中的 `--max-steps` 表示 optimizer step 数；评测命令中的同名参数表示每个 Episode
最多执行的 primitive action 数。

## 8. Smoke 训练

先使用单卡和一个 optimizer step 检查模型、数据与 checkpoint 写入：

```bash
CUDA_DEVICES=0 \
SATNAV_MAX_EPISODES=1 \
SATNAV_MAX_SAMPLES=1 \
bash baselines/vlm/streamvln/scripts/train.sh continue \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --max-steps 1 \
  --dataloader-workers 0 \
  --no-data-augmentation \
  --no-torch-compile \
  --output-dir output/baselines/vlm/streamvln/train/continue-smoke
```

`SATNAV_MAX_SAMPLES` 至少应等于 `GPU 数 × per-device batch size`，否则在
`dataloader_drop_last=true` 时无法形成一个完整 batch。

Smoke 完成后，确认输出目录中存在模型配置、权重和 trainer state。选择包含
`config.json` 与模型权重的目录作为后续 `--model-path`：

```bash
find output/baselines/vlm/streamvln/train/continue-smoke \
  -maxdepth 2 -name config.json -print
```

## 9. 完整训练

从官方 StreamVLN checkpoint 继续训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/train.sh continue \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/train/continue-v0-1
```

从 LLaVA-Video base model 开始训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/train.sh scratch \
  --trajectory-root "$SATNAV_STREAMVLN_TRAIN_DATA" \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/train/scratch-v0-1
```

显存不足时优先减小 `--batch-size`，并使用 `--gradient-accumulation` 保持所需的有效 batch
size。当前 launcher 不提供独立的 resume 参数；每次训练应使用独立输出目录。

## 10. 配置在线评测

设置待评测 checkpoint：

```bash
export STREAMVLN_CHECKPOINT=/path/to/trained-streamvln-checkpoint
```

检查 Episode、scene、tokenizer 和输出路径能否解析：

```bash
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run 只检查配置、数据选择和输出目录，不加载模型，也不执行环境交互。

## 11. 单 GPU Smoke 评测

选择 5 个 Episode，每个 Episode 最多执行 5 步：

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/5steps-1rank \
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
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit 5 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

所有 rank 成功退出后，launcher 会自动生成合并后的 `summary.json`。

## 13. 完整评测

完整 `val_seen` 评测使用 500 步上限和全部 Episode：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/streamvln/scripts/eval.sh \
  --model-path "$STREAMVLN_CHECKPOINT" \
  --tokenizer-path "$STREAMVLN_TOKENIZER_PATH" \
  --vision-tower "$STREAMVLN_VISION_TOWER" \
  --episodes "$SATNAV_STREAMVLN_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_STREAMVLN_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/streamvln/eval/continue-v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

完成后，将 split 和输出目录改为 `val_unseen` 再运行一次。
`SATNAV_STREAMVLN_EVAL_EPISODES` 中的 `{split}` 会自动替换为当前 split。

评测中断后，可以使用完全相同的参数和输出目录重新执行，并在命令末尾添加 `--resume`。
Resume 会跳过当前 rank 已经写入的 Episode。更换 checkpoint、数据、seed、GPU 数量或评测
参数时，应使用新的输出目录。

## 14. 常见问题

### 为什么提示 StreamVLN revision 不匹配？

确认 `STREAMVLN_REPO` 指向正确目录，并执行：

```bash
git -C "$STREAMVLN_REPO" checkout \
  60476e81f4c01b29f1a51a7469f1cb4addbc1d62
```

### 为什么找不到 tokenizer？

将 `STREAMVLN_TOKENIZER_PATH` 指向包含 `tokenizer_config.json` 的目录。评测时也可以显式传入
`--tokenizer-path`。不要只指定模型权重 shard 所在目录。

### 为什么 vision tower 加载失败？

确认 `STREAMVLN_VISION_TOWER` 指向完整的本地 SigLIP 模型目录，并且该目录与训练时使用的
vision tower 一致。

### 为什么数据校验提示 frame 数量不足？

检查 `annotations.json` 中的 `video` 路径是否对应当前 `images/`，以及 trajectory 生成过程
是否完整结束。不要组合来自不同 trajectory export 的 annotation 和图像目录。

### 为什么 Smoke 训练没有产生 optimizer step？

使用 `SATNAV_MAX_SAMPLES` 限制样本时，样本数不能小于 `GPU 数 × per-device batch size`。
同时确认选中的 trajectory 在 action 规范化后至少包含 4 个动作。

### 为什么启动训练后长时间没有输出？

第一次启动需要加载大模型、初始化 DeepSpeed 和编译模型。排查环境或数据时，可以使用
`--dataloader-workers 0 --no-data-augmentation --no-torch-compile` 运行 bounded smoke。

### 为什么重新评测时没有执行部分 Episode？

使用 `--resume` 时，evaluator 会跳过当前输出目录中已经完成的 Episode。只有继续完全相同
的评测时才应复用该目录；其他情况请指定新的 `--output-dir`。
