# SatNav NaVILA Baseline

本文介绍如何使用 SatNav-v0.1 trajectory 训练 NaVILA，并在 SatSim 中完成单卡或多卡在线
评测。NaVILA 使用独立 Python 环境和外部模型代码，不与 SatNav Core、Classic 或其他 VLM
baseline 共用环境。

开始前建议先阅读[环境安装](../../getting-started/INSTALLATION.md)、[模型训练](../README.md)和
[统一评测](../../evaluation/README.md)。

## 1. 模型与 SatNav 接口

NaVILA 根据导航指令、当前 RGB observation 和历史视觉信息生成文本动作。SatNav adapter 将
模型输出转换为四个 primitive action：

| 模型输出 | SatNav action | 环境行为 |
| --- | --- | --- |
| `stop` | `STOP` | 结束当前 Episode |
| `forward` | `MOVE_FORWARD` | 前进 10 m |
| `left` | `TURN_LEFT` | 左转 15° |
| `right` | `TURN_RIGHT` | 右转 15° |

默认使用 `compact` action format，要求模型只返回一个动作词。`sentence` format 保留 NaVILA
原有的自然语言输出方式；当输出包含更长距离或更大转角时，adapter 会将其拆分为多个
SatNav primitive action 并依次执行。

每次推理使用 8 帧历史 observation。历史不足时在前方补黑帧，历史较长时等间隔采样并始终
保留当前帧。

SatNav 当前适配的上游版本为
[`AnjieCheng/NaVILA@76b98f233dd0fff05dfcd69435eec6740febff9d`](https://github.com/AnjieCheng/NaVILA/commit/76b98f233dd0fff05dfcd69435eec6740febff9d)。
使用模型代码和权重前，请阅读 [UPSTREAM](../../../../baselines/vlm/navila/UPSTREAM.md)、
[NOTICE](../../../../baselines/vlm/navila/NOTICE) 和
[Upstream License](../../../../baselines/vlm/navila/LICENSE.upstream)。

## 2. 准备独立环境

在 SatNav 仓库根目录创建 NaVILA 环境：

```bash
conda env create -f baselines/vlm/navila/environment/conda.yml
conda activate satnav-navila
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装适配 Python 3.10、PyTorch 2.3、CUDA 12.x 和旧 CXX11 ABI 的 FlashAttention 2.5.8
wheel：

```bash
export NAVILA_FLASH_ATTN_WHEEL="${NAVILA_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
python -m pip install "$NAVILA_FLASH_ATTN_WHEEL"
```

如果当前平台无法使用该 wheel，应准备与当前 Python、PyTorch、CUDA 和 ABI 完全匹配的本地
wheel，再将 `NAVILA_FLASH_ATTN_WHEEL` 指向该文件。

继续安装其余依赖和 SatNav：

```bash
python -m pip install -r baselines/vlm/navila/requirements.txt
python -m pip install -e .
```

## 3. 准备上游代码

将 NaVILA clone 到 SatNav 仓库之外，并 checkout 到适配版本：

```bash
git clone https://github.com/AnjieCheng/NaVILA.git /path/to/NaVILA
git -C /path/to/NaVILA checkout --detach \
  76b98f233dd0fff05dfcd69435eec6740febff9d

python -m pip install --no-deps -e /path/to/NaVILA
bash baselines/vlm/navila/scripts/patch_environment.sh
```

`patch_environment.sh` 将固定上游版本提供的 Transformers 和 DeepSpeed 兼容文件安装到当前
NaVILA 环境，不会修改上游 checkout。不要在其他 VLM 环境中运行该脚本。

确认源码版本、环境和入口：

```bash
git -C /path/to/NaVILA rev-parse HEAD
git -C /path/to/NaVILA status --short

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
python -m baselines.vlm.navila.dataset --help
python -m baselines.vlm.navila.trainer --help
python -m baselines.vlm.navila.evaluate --help
python -m baselines.vlm.navila.checkpoint --help
command -v python
```

`rev-parse` 应输出第 1 节列出的 revision，`status --short` 应无输出。记录
`command -v python` 的结果，后续将其设置为 `NAVILA_PYTHON`。

## 4. 准备 SatNav 数据

### 4.1 Episode、GeoTIFF 与 Trajectory

按照 [Episode 数据下载](../../dataset/DATA_DOWNLOAD.md)、[卫星场景下载](../../applications/MAP_DOWNLOAD.md)和
[轨迹数据生成](../../applications/TRAJECTORY_GENERATION.md)准备 SatNav-v0.1。NaVILA 直接读取
trajectory 中的 JPEG：

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

检查 annotation、action、frame 数量与图片解码：

```bash
python -m baselines.vlm.navila.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --expected-episodes 105164 \
  --strict-frames \
  --decode-samples 64
```

正式训练开始前，launcher 还会自动检查完整 train split。缺失图片、越界路径、action 不合法
或 frame 数量不匹配都会直接终止训练，不会静默跳过样本。

## 5. 下载模型

选择模型保存目录：

```bash
export NAVILA_MODEL_ROOT=/path/to/navila-models
mkdir -p "$NAVILA_MODEL_ROOT"
```

### 5.1 SatNav 已发布 checkpoint

[SatNav Baseline Model Zoo](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
提供两个 NaVILA checkpoint：

- [scratch](https://huggingface.co/Eku127/navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4)：从 NaVILA pretrain checkpoint 开始训练；
- [continue](https://huggingface.co/Eku127/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4)：从 NaVILA SFT checkpoint 继续训练。

使用 baseline downloader 下载所需 checkpoint：

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo Eku127/navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4 \
  --model-root "$NAVILA_MODEL_ROOT"
```

评测时将下载后的目录传给 `--model-path`。发布目录包含推理所需文件，不包含 optimizer 与
scheduler state。

已发布 checkpoint 在完整 split 和 500 步上限下的参考结果如下：

| Checkpoint | Split | Episode 数 | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | `val_seen` | 4,574 | 93.05 | 27.59 | 18.10 | 18.02 |
| Scratch | `val_unseen` | 8,756 | 128.88 | 23.66 | 13.00 | 12.96 |
| Continue | `val_seen` | 4,574 | 93.88 | 35.00 | 24.99 | 24.88 |
| Continue | `val_unseen` | 8,756 | 123.49 | 31.60 | 18.57 | 18.44 |

### 5.2 训练起点

`continue` 训练使用 NaVILA SFT checkpoint：

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo a8cheng/navila-llama3-8b-8f \
  --model-root "$NAVILA_MODEL_ROOT"
```

`scratch` 训练从 NaVILA pretrain checkpoint 开始：

```bash
bash baselines/vlm/navila/scripts/download.sh \
  --repo a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
  --model-root "$NAVILA_MODEL_ROOT"
```

如果 Hugging Face 模型仓库要求授权，请先在当前环境完成登录。需要使用镜像时，可在下载
命令中添加 `--endpoint`。

下载完成后的主要路径为：

```text
navila-models/
├── navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4/
├── navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4/
├── navila-llama3-8b-8f/
└── navila-siglip-llama3-8b-v1.5-pretrain/
```

两个目录都应包含完整的模型配置、tokenizer、vision tower、multimodal projector 和全部
权重文件，不能只下载单独的权重 shard 或 adapter。

## 6. 配置本地路径

此时环境、上游代码、SatNav 数据和模型目录均已准备完成。复制 NaVILA 本地配置模板：

```bash
mkdir -p baselines/vlm/navila/.local
cp baselines/vlm/navila/local.env.example \
  baselines/vlm/navila/.local/env.sh
```

在 `baselines/vlm/navila/.local/env.sh` 中填写前面步骤得到的实际路径：

```bash
export NAVILA_REPO="${NAVILA_REPO:-/path/to/NaVILA}"
export NAVILA_PYTHON="${NAVILA_PYTHON:-/path/to/satnav-navila/bin/python}"
export NAVILA_MODEL_ROOT="${NAVILA_MODEL_ROOT:-/path/to/navila-models}"
export NAVILA_PRETRAIN_MODEL="${NAVILA_PRETRAIN_MODEL:-${NAVILA_MODEL_ROOT}/navila-siglip-llama3-8b-v1.5-pretrain}"
export NAVILA_SFT_MODEL="${NAVILA_SFT_MODEL:-${NAVILA_MODEL_ROOT}/navila-llama3-8b-8f}"

export SATNAV_NAVILA_TRAIN_DATA="${SATNAV_NAVILA_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_NAVILA_EVAL_EPISODES="${SATNAV_NAVILA_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_NAVILA_SCENES_DIR="${SATNAV_NAVILA_SCENES_DIR:-/path/to/scenes}"
export SATNAV_NAVILA_OUTPUT="${SATNAV_NAVILA_OUTPUT:-output/baselines/vlm/navila}"
```

各变量对应的来源为：

| 变量 | 路径来源 |
| --- | --- |
| `NAVILA_REPO` | 第 3 节 clone 的 NaVILA 目录 |
| `NAVILA_PYTHON` | 第 3 节 `command -v python` 的输出 |
| `NAVILA_MODEL_ROOT` | 第 5 节选择的模型保存目录 |
| `NAVILA_PRETRAIN_MODEL` | 下载后的 pretrain checkpoint |
| `NAVILA_SFT_MODEL` | 下载后的 SFT checkpoint |
| `SATNAV_NAVILA_TRAIN_DATA` | 第 4 节校验通过的 `trajectory_data/` |
| `SATNAV_NAVILA_EVAL_EPISODES` | SatNav-v0.1 eval Episode 路径模板 |
| `SATNAV_NAVILA_SCENES_DIR` | 第 4 节校验通过的 GeoTIFF 目录 |
| `SATNAV_NAVILA_OUTPUT` | 训练和评测输出根目录 |

启动脚本会自动读取该文件。临时在终端显式设置的同名环境变量优先于 `.local/env.sh`。
本地数据、模型、上游代码、输出目录和环境路径均不应提交到 Git。

如需在当前终端后续命令中直接使用这些变量，加载本地配置：

```bash
source baselines/vlm/navila/.local/env.sh
```

## 7. 训练配置与 Checkpoint 检查

训练前先确认 SFT checkpoint 可以完整加载：

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path "$NAVILA_SFT_MODEL" \
  --navila-repo "$NAVILA_REPO" \
  --device cuda:0
```

如需运行 `scratch`，将 `--model-path` 改为 `$NAVILA_PRETRAIN_MODEL` 再检查一次。缺失、形状
不匹配或无法加载的模型组件会直接报错。

默认训练参数位于 `baselines/vlm/navila/configs/train.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| Video frames | 8 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | `3e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 8 |

训练使用 `baselines/vlm/navila/configs/zero2.json` 中的 DeepSpeed ZeRO-2。默认更新 vision
tower、multimodal projector 和 language model；bounded smoke 可以通过
`--train-components projector` 只检查 projector 更新链路。

训练命令中的 `--max-steps` 表示 optimizer step 数；评测命令中的同名参数表示每个 Episode
最多执行的 primitive action 数。

## 8. Smoke 训练

先运行两个 optimizer step，检查数据、模型更新和完整 checkpoint 保存：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_SFT_MODEL" \
  --output-dir output/baselines/vlm/navila/train/continue-smoke \
  --max-steps 2 \
  --save-steps 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --warmup-ratio 0 \
  --dataloader-workers 0 \
  --train-components projector \
  --max-samples 16
```

Smoke 使用 8 张 GPU，因此 `--max-samples 16` 可以形成两个完整 global microbatch。调整 GPU
数或 batch size 时，样本上限至少应为 `GPU 数 × per-device batch size`。

训练完成后，确认输出可以重新加载，并且 projector 参数相对起始模型发生变化：

```bash
python -m baselines.vlm.navila.checkpoint \
  --model-path output/baselines/vlm/navila/train/continue-smoke \
  --compare-model "$NAVILA_SFT_MODEL" \
  --navila-repo "$NAVILA_REPO" \
  --device cuda:0
```

## 9. 完整训练

从 NaVILA SFT checkpoint 继续训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh continue \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_SFT_MODEL" \
  --output-dir output/baselines/vlm/navila/train/continue-v0-1
```

从 NaVILA pretrain checkpoint 开始训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/train.sh scratch \
  --gpus 8 \
  --trajectory-root "$SATNAV_NAVILA_TRAIN_DATA" \
  --model-path "$NAVILA_PRETRAIN_MODEL" \
  --output-dir output/baselines/vlm/navila/train/scratch-v0-1
```

显存不足时优先减小 `--batch-size`，并通过 `--gradient-accumulation` 保持所需的有效 batch
size。当前 launcher 不提供 resume 参数；每次正式训练应使用独立输出目录。

## 10. 配置在线评测

设置待评测 checkpoint。训练输出目录本身就是完整 checkpoint：

```bash
export NAVILA_CHECKPOINT=/path/to/trained-navila-checkpoint
```

检查 Episode、scene 和输出路径能否解析：

```bash
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --dry-run
```

Dry run 只检查配置、数据选择和输出目录，不加载模型，也不执行环境交互。

## 11. 单 GPU Smoke 评测

选择 4 个 Episode，每个 Episode 最多执行 5 步：

```bash
CUDA_DEVICES=0 \
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/5steps-1rank \
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
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

所有 rank 成功退出后，launcher 会自动生成合并后的 `summary.json`。

## 13. 完整评测

完整 `val_seen` 评测使用 500 步上限和全部 Episode：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/navila/scripts/eval.sh \
  --model-path "$NAVILA_CHECKPOINT" \
  --episodes "$SATNAV_NAVILA_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_NAVILA_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/navila/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

完成后，将 split 和输出目录改为 `val_unseen` 再运行一次。
`SATNAV_NAVILA_EVAL_EPISODES` 中的 `{split}` 会自动替换为当前 split。

评测中断后，可以使用完全相同的参数和输出目录重新执行，并在命令末尾添加 `--resume`。
Resume 会跳过当前 rank 已经写入的 Episode。更换 checkpoint、数据、seed、GPU 数量或评测
参数时，应使用新的输出目录。

## 14. 常见问题

### 为什么提示 NaVILA revision 不匹配？

确认 `NAVILA_REPO` 指向正确目录，并恢复到适配版本：

```bash
git -C "$NAVILA_REPO" checkout --detach \
  76b98f233dd0fff05dfcd69435eec6740febff9d
git -C "$NAVILA_REPO" status --short
```

`status --short` 应无输出。不要在该 checkout 中保存模型、日志或其他本地文件。

### 为什么 FlashAttention 无法导入？

确认 wheel 的 Python、PyTorch、CUDA 和 CXX11 ABI 与当前环境一致。重新创建干净环境后，先
安装 PyTorch 和 FlashAttention，再安装其余依赖，不要复用其他 VLM 的环境。

### 为什么环境 patch 失败？

确认当前环境使用固定的 Transformers 和 DeepSpeed 版本，并且 `NAVILA_REPO` 位于适配
revision。`patch_environment.sh` 只能在完成依赖安装后运行。

### 为什么 checkpoint 无法加载？

确认使用的是完整 NaVILA checkpoint，而不是单独的 projector、adapter 或部分权重。模型
目录应同时包含 tokenizer、vision tower、multimodal projector 和 language model 文件。

### 为什么数据校验提示 frame 数量不匹配？

检查 `annotations.json` 中的 `video` 路径是否对应当前 `images/`，以及 trajectory 生成过程
是否完整结束。不要组合来自不同 trajectory export 的 annotation 和图像目录。

### 为什么训练启动后长时间没有加载模型？

训练会先检查完整 SatNav-v0.1 trajectory。数据位于网络存储或低速磁盘时，该阶段可能需要
较长时间；建议将 annotation 和 JPEG 放在本地高速存储。

### 为什么重新评测时没有执行部分 Episode？

使用 `--resume` 时，evaluator 会跳过当前输出目录中已经完成的 Episode。只有继续完全相同
的评测时才应复用该目录；其他情况请指定新的 `--output-dir`。
