# SatNav Uni-NaVid Baseline

本文介绍如何使用 SatNav-v0.1 trajectory 训练 Uni-NaVid，并在 SatSim 中完成单卡或多卡
在线评测。Uni-NaVid 使用独立 Python 环境和外部模型代码，不与 SatNav Core、Classic 或其他
VLM baseline 共用环境。

开始前建议先阅读[环境安装](../../getting-started/INSTALLATION.md)、[模型训练](../README.md)和
[统一评测](../../evaluation/README.md)。

## 1. 模型与 SatNav 接口

Uni-NaVid 根据导航指令和连续 RGB observation 生成最多四个文本动作。SatNav adapter 将
模型输出转换为四个 primitive action：

| 模型输出 | SatNav action | 环境行为 |
| --- | --- | --- |
| `stop` | `STOP` | 结束当前 Episode |
| `forward` | `MOVE_FORWARD` | 前进 10 m |
| `left` | `TURN_LEFT` | 左转 15° |
| `right` | `TURN_RIGHT` | 右转 15° |

模型生成的动作会进入队列，再按照 SatNav 的逐步交互接口依次执行。评测过程中，adapter 会
持续维护新增 observation 和导航特征缓存，并在每个 Episode 开始时重置模型状态。

SatNav 当前适配的上游版本为
[`jzhzhang/Uni-NaVid@79ef5ea3fea14c205342d1ab070563d84c7a966a`](https://github.com/jzhzhang/Uni-NaVid/commit/79ef5ea3fea14c205342d1ab070563d84c7a966a)。
使用模型代码和权重前，请阅读 [UPSTREAM](../../../../baselines/vlm/uninavid/UPSTREAM.md)、
[NOTICE](../../../../baselines/vlm/uninavid/NOTICE) 和
[Upstream License](../../../../baselines/vlm/uninavid/LICENSE.upstream)。

## 2. 准备独立环境

在 SatNav 仓库根目录创建 Uni-NaVid 环境：

```bash
conda env create -f baselines/vlm/uninavid/environment/conda.yml
conda activate satnav-uninavid
export PYTHONNOUSERSITE=1

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装与 Python 3.9、PyTorch 2.5、CUDA 和本机 ABI 匹配的 FlashAttention wheel：

```bash
export UNINAVID_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl
test -f "$UNINAVID_FLASH_ATTN_WHEEL"
python -m pip install "$UNINAVID_FLASH_ATTN_WHEEL"
```

继续安装其余依赖和 SatNav：

```bash
python -m pip install -r baselines/vlm/uninavid/requirements.txt
python -m pip install -e .
python -m pip check
```

Uni-NaVid adapter 直接读取 SatNav trajectory 中的 JPEG，不需要将图像转换为 MP4，也不需要
额外安装 Decord。上游 Uni-NaVid 源码将在后续步骤中通过路径加载，不要将其安装为 Python
package。

检查关键依赖和入口：

```bash
python - <<'PY'
import accelerate
import deepspeed
import flash_attn
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
PY

python -m baselines.vlm.uninavid.dataset --help
python -m baselines.vlm.uninavid.trainer --help
python -m baselines.vlm.uninavid.evaluate --help
python -m baselines.vlm.uninavid.checkpoint --help
command -v python
```

记录 `command -v python` 的输出，后续将其设置为 `UNINAVID_PYTHON`。

## 3. 准备上游代码

将 Uni-NaVid clone 到 SatNav 仓库之外，并 checkout 到适配版本：

```bash
git clone https://github.com/jzhzhang/Uni-NaVid.git /path/to/Uni-NaVid
git -C /path/to/Uni-NaVid checkout --detach \
  79ef5ea3fea14c205342d1ab070563d84c7a966a
```

确认 checkout 位于正确 revision，且没有本地修改：

```bash
git -C /path/to/Uni-NaVid rev-parse HEAD
git -C /path/to/Uni-NaVid status --short
```

训练和评测启动时会检查当前 revision 和工作区状态。不要修改该 checkout，也不要在同一目录
中切换到其他 Uni-NaVid 版本后继续使用已有输出。

## 4. 准备 SatNav 数据

### 4.1 Episode、GeoTIFF 与 Trajectory

按照 [Episode 数据下载](../../dataset/DATA_DOWNLOAD.md)、[卫星场景下载](../../applications/MAP_DOWNLOAD.md)和
[轨迹数据生成](../../applications/TRAJECTORY_GENERATION.md)准备 SatNav-v0.1。Uni-NaVid 直接读取
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

检查 Uni-NaVid 的四动作窗口、action 和 frame 对齐：

```bash
python -m baselines.vlm.uninavid.dataset \
  "$SATNAV_TRAJECTORY_ROOT" \
  --expected-episodes 105164 \
  --expected-samples 1399366 \
  --strict-frames \
  --decode-samples 32
```

正式训练开始前，launcher 还会自动检查完整 train split。缺失图片、越界路径、action 不合法
或 frame 数量不足都会直接终止训练，不会静默跳过样本。

## 5. 下载模型

选择模型保存目录，并设置上一节准备的上游代码路径：

```bash
export UNINAVID_REPO=/path/to/Uni-NaVid
export UNINAVID_MODEL_ROOT=/path/to/uninavid-models
mkdir -p "$UNINAVID_MODEL_ROOT"
```

下载 Uni-NaVid 完整 checkpoint 和 EVA 权重：

```bash
bash baselines/vlm/uninavid/scripts/download.sh \
  --source-dir "$UNINAVID_REPO" \
  --model-root "$UNINAVID_MODEL_ROOT" \
  --model-revision main
```

如果 Hugging Face 模型仓库要求授权，请先在当前环境完成登录，再重新运行下载命令。下载器
支持断点复用，并会同时确认上游代码版本和 EVA 文件。

下载完成后的主要路径为：

```text
uninavid-models/
├── eva_vit_g.pth
└── huggingface/
    └── Uni-NaVid/
        └── uninavid-7b-full-224-video-fps-1-grid-2/
```

此外，image processor 位于上游代码目录：

```text
Uni-NaVid/uninavid/processor/clip-patch14-224/
```

## 6. 配置本地路径

此时环境、上游代码、SatNav 数据和模型目录均已准备完成。复制 Uni-NaVid 本地配置模板：

```bash
mkdir -p baselines/vlm/uninavid/.local
cp baselines/vlm/uninavid/local.env.example \
  baselines/vlm/uninavid/.local/env.sh
```

在 `baselines/vlm/uninavid/.local/env.sh` 中填写前面步骤得到的实际路径：

```bash
export UNINAVID_REPO="${UNINAVID_REPO:-/path/to/Uni-NaVid}"
export UNINAVID_MODEL_ROOT="${UNINAVID_MODEL_ROOT:-/path/to/uninavid-models}"
export UNINAVID_MODEL="${UNINAVID_MODEL:-/path/to/uninavid-models/huggingface/Uni-NaVid/uninavid-7b-full-224-video-fps-1-grid-2}"
export UNINAVID_EVA="${UNINAVID_EVA:-/path/to/uninavid-models/eva_vit_g.pth}"
export UNINAVID_PROCESSOR="${UNINAVID_PROCESSOR:-${UNINAVID_REPO}/uninavid/processor/clip-patch14-224}"
export UNINAVID_PYTHON="${UNINAVID_PYTHON:-/path/to/satnav-uninavid/bin/python}"
export UNINAVID_FLASH_ATTN_WHEEL="${UNINAVID_FLASH_ATTN_WHEEL:-/path/to/flash_attn.whl}"

export SATNAV_UNINAVID_TRAIN_DATA="${SATNAV_UNINAVID_TRAIN_DATA:-/path/to/trajectory_data}"
export SATNAV_UNINAVID_EVAL_EPISODES="${SATNAV_UNINAVID_EVAL_EPISODES:-/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json}"
export SATNAV_UNINAVID_SCENES_DIR="${SATNAV_UNINAVID_SCENES_DIR:-/path/to/scenes}"
export SATNAV_UNINAVID_OUTPUT="${SATNAV_UNINAVID_OUTPUT:-output/baselines/vlm/uninavid}"
```

各变量对应的来源为：

| 变量 | 路径来源 |
| --- | --- |
| `UNINAVID_REPO` | 第 3 节 clone 的 Uni-NaVid 目录 |
| `UNINAVID_MODEL_ROOT` | 第 5 节选择的模型保存目录 |
| `UNINAVID_MODEL` | 下载后的完整 Uni-NaVid checkpoint |
| `UNINAVID_EVA` | 下载后的 `eva_vit_g.pth` |
| `UNINAVID_PROCESSOR` | 上游代码中的 `clip-patch14-224/` |
| `UNINAVID_PYTHON` | 第 2 节 `command -v python` 的输出 |
| `UNINAVID_FLASH_ATTN_WHEEL` | 第 2 节安装的 FlashAttention wheel |
| `SATNAV_UNINAVID_TRAIN_DATA` | 第 4 节校验通过的 `trajectory_data/` |
| `SATNAV_UNINAVID_EVAL_EPISODES` | SatNav-v0.1 eval Episode 路径模板 |
| `SATNAV_UNINAVID_SCENES_DIR` | 第 4 节校验通过的 GeoTIFF 目录 |
| `SATNAV_UNINAVID_OUTPUT` | 训练和评测输出根目录 |

启动脚本会自动读取该文件。临时在终端显式设置的同名环境变量优先于 `.local/env.sh`。
本地数据、模型、上游代码、输出目录和环境路径均不应提交到 Git。

如需在当前终端后续命令中直接使用这些变量，加载本地配置：

```bash
source baselines/vlm/uninavid/.local/env.sh
```

## 7. 训练配置与 Checkpoint 检查

训练前先确认下载的完整 checkpoint 可以加载：

```bash
python -m baselines.vlm.uninavid.checkpoint \
  --model-path "$UNINAVID_MODEL" \
  --uninavid-repo "$UNINAVID_REPO" \
  --eva-path "$UNINAVID_EVA" \
  --processor-path "$UNINAVID_PROCESSOR" \
  --device cuda:0
```

该命令会检查 checkpoint、tokenizer、vision tower 和 FlashAttention 是否能够完整加载。
缺失、不兼容或形状不匹配的权重会直接报错。

默认训练参数位于 `baselines/vlm/uninavid/configs/train.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| Action window | 4 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 1 |
| Learning rate | `1e-5` |
| Warmup ratio | `0.03` |
| Save interval | 1,000 steps |
| DataLoader workers | 2 |

训练使用 `baselines/vlm/uninavid/configs/zero1.json` 中的 DeepSpeed ZeRO-1。默认更新 language
model 和 multimodal projector，同时保持 vision tower 冻结。训练结束后，launcher 会重新
加载完整输出，确认可用于后续评测。

训练命令中的 `--max-steps` 表示 optimizer step 数；评测命令中的同名参数表示每个 Episode
最多执行的 primitive action 数。

## 8. Smoke 训练

先运行两个 optimizer step，检查数据、模型更新和 checkpoint 保存：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/smoke \
  --max-steps 2 \
  --save-steps 1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --warmup-ratio 0 \
  --dataloader-workers 0 \
  --max-samples 16 \
  --disable-augmentation
```

Smoke 使用 8 张 GPU，因此 `--max-samples 16` 可以形成两个完整 global microbatch。调整 GPU
数或 batch size 时，样本上限至少应为 `GPU 数 × per-device batch size`。

训练成功后，输出目录包含可直接评测的完整模型、训练日志和 `checkpoint-*`。launcher 还会
自动确认 language model 与 projector 已更新，并且冻结的 vision tower 没有变化。

## 9. 完整训练与 Resume

使用完整 SatNav-v0.1 trajectory 训练：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/v0-1
```

默认训练一个 epoch。显存不足时可以减小 `--batch-size`，并使用
`--gradient-accumulation` 调整有效 batch size。

训练中断且输出目录中已经存在 `checkpoint-*` 时，使用完全相同的参数和输出目录重新执行，
并添加 `--resume`：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/train.sh \
  --gpus 8 \
  --trajectory-root "$SATNAV_UNINAVID_TRAIN_DATA" \
  --model-path "$UNINAVID_MODEL" \
  --eva-path "$UNINAVID_EVA" \
  --output-dir output/baselines/vlm/uninavid/train/v0-1 \
  --resume
```

Resume 只适用于继续同一次训练。更换模型、数据、配置、GPU 数量或采样参数时，应使用新的
输出目录。

## 10. 配置在线评测

设置待评测 checkpoint。训练输出目录本身就是完整 checkpoint：

```bash
export UNINAVID_CHECKPOINT=/path/to/trained-uninavid-checkpoint
```

检查 Episode、scene 和输出路径能否解析：

```bash
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
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
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 1 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/5steps-1rank \
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
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit 4 \
  --max-steps 5 \
  --gpus 2 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/5steps-2rank \
  --fail-on-episode-error
```

所有 rank 成功退出后，launcher 会自动生成合并后的 `summary.json`。

## 13. 完整评测

完整 `val_seen` 评测使用 500 步上限和全部 Episode：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
bash baselines/vlm/uninavid/scripts/eval.sh \
  --model-path "$UNINAVID_CHECKPOINT" \
  --eva-path "$UNINAVID_EVA" \
  --episodes "$SATNAV_UNINAVID_EVAL_EPISODES" \
  --scenes-dir "$SATNAV_UNINAVID_SCENES_DIR" \
  --split val_seen \
  --limit -1 \
  --max-steps 500 \
  --gpus 8 \
  --output-dir output/baselines/vlm/uninavid/eval/v0-1/val_seen/500steps-8rank \
  --fail-on-episode-error
```

完成后，将 split 和输出目录改为 `val_unseen` 再运行一次。
`SATNAV_UNINAVID_EVAL_EPISODES` 中的 `{split}` 会自动替换为当前 split。

评测中断后，可以使用完全相同的参数和输出目录重新执行，并在命令末尾添加 `--resume`。
Resume 会跳过当前 rank 已经写入的 Episode。更换 checkpoint、数据、seed、GPU 数量或评测
参数时，应使用新的输出目录。

## 14. 常见问题

### 为什么提示 Uni-NaVid revision 不匹配或工作区不干净？

确认 `UNINAVID_REPO` 指向正确目录，并恢复到适配版本：

```bash
git -C "$UNINAVID_REPO" checkout --detach \
  79ef5ea3fea14c205342d1ab070563d84c7a966a
git -C "$UNINAVID_REPO" status --short
```

`status --short` 应无输出。不要在该 checkout 中保存模型、日志或其他本地文件。

### 为什么模型下载失败？

确认当前机器可以访问 Hugging Face，并检查模型仓库是否要求登录或接受使用条款。完成授权后
重新运行相同下载命令即可复用已经下载的文件。

### 为什么找不到 EVA 或 processor？

`UNINAVID_EVA` 应指向下载器生成的 `eva_vit_g.pth`；`UNINAVID_PROCESSOR` 应指向固定上游
checkout 中包含 `preprocessor_config.json` 的 `clip-patch14-224/` 目录。

### 为什么 checkpoint 无法加载？

确认使用的是完整 Uni-NaVid checkpoint，而不是单独的 adapter 或部分权重。模型目录应同时
包含 config、tokenizer、权重索引和全部权重 shard。

### 为什么数据校验提示 frame 数量不足？

检查 `annotations.json` 中的 `video` 路径是否对应当前 `images/`，以及 trajectory 生成过程
是否完整结束。不要组合来自不同 trajectory export 的 annotation 和图像目录。

### 为什么训练启动后长时间没有加载模型？

训练会先检查完整 SatNav-v0.1 trajectory。数据位于网络存储或低速磁盘时，该阶段可能需要
较长时间；建议将 annotation 和 JPEG 放在本地高速存储。

### 为什么 Resume 被拒绝？

确认输出目录中存在 `checkpoint-*`，并使用与首次训练完全相同的模型、数据、配置、GPU 数量
和采样参数。需要改变实验设置时，请创建新的输出目录。

### 为什么重新评测时没有执行部分 Episode？

使用 `--resume` 时，evaluator 会跳过当前输出目录中已经完成的 Episode。只有继续完全相同
的评测时才应复用该目录；其他情况请指定新的 `--output-dir`。
