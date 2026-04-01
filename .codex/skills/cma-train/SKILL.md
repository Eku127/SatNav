---
name: cma-train
description: Full CMA training pipeline on SatNav ver_260317 data. Uses the same offline_trainer as seq2seq; only the model architecture and hyperparameters differ. Use when the user asks to train cma, run cma full training, 训练cma, 启动cma训练, or prepare cma for training.
---

# CMA Training (ver_260317)

唯一推荐路径：**离线 `offline_trainer`**，不启动仿真器，直接读取预渲染轨迹数据。

CMA 模型实现来自 [Zhu et al., 2020](https://arxiv.org/abs/2004.02857)（VLN-CE CMA 简化版，去掉深度编码器和进度监控器，核心 RGB-instruction 跨模态注意力保留）。

- 训练配置：`configs/baselines/cma.yaml`
- 多卡脚本：`scripts/cma/train_ddp.sh`
- 监控：**SwanLab**（默认 `cloud` 模式；可切 `local`）
- 推荐起点（8xH100）：
  - `IL.batch_size=4`（per GPU，global batch 32；CMA 显存比 seq2seq 高，建议从 4 开始）
  - `IL.lr=1e-4`
  - `IL.OFFLINE.num_workers=4`

## Key Paths

```text
REPO:           /mnt/data1/home/jiangjiajun/workspace/SatNav
DATASET:        /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317
TRAJ_DATA:      /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data
OUTPUT_ROOT:    output/cma
RUN_LOGS:       output/cma/logs
SWANLAB_LOGDIR: output/cma/swanlab     (仅 local/offline 模式写入)
LATEST_LINK:    output/cma/checkpoints/latest    (训练完成后自动更新)
VOCAB:          output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json
EMBEDDING:      output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz
TRAIN_CONFIG:   configs/baselines/cma.yaml
DDP_SCRIPT:     scripts/cma/train_ddp.sh
```

> Vocab 和 Embeddings 与 seq2seq 共用，直接复用已构建产物。

**Output 结构说明：**

```
output/cma/
├── checkpoints/
│   ├── latest -> cma-ddp-<timestamp>   ← 软链，始终指向最新成功训练
│   └── cma-ddp-<timestamp>/best.pth
├── logs/
│   └── cma-ddp-<timestamp>.log
├── results/                         ← 由 eval 步骤按需创建
│   └── cma-ddp-<timestamp>/
└── swanlab/                         ← 仅 local/offline SwanLab 模式写入
```

Conda env: `satnav`
Working dir: always `REPO` above.

---

## Workflow

### Step 1 — Vocab & Embeddings 前置检查

CMA 复用 seq2seq 的 vocab 和 embeddings；先确认已存在：

```bash
ls output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json 2>/dev/null && echo "EXISTS" || echo "MISSING"
ls output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

如果 MISSING，先运行 seq2seq-train skill 的 Step 2–4 构建这两个产物，CMA 即可直接使用。

---

### Step 2 — 验证训练配置

`configs/baselines/cma.yaml` 核心字段应满足：

```yaml
TRAINER_NAME: offline_trainer

IL:
  lr: 1e-4
  batch_size: 1     # 默认；推荐实际使用时覆盖为 4
  OFFLINE:
    annotations_path: /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json
    images_root: /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/images
  use_class_weighting: true
  class_weights:
    STOP: 2.0
    MOVE_FORWARD: 1.0
    TURN_LEFT: 1.5
    TURN_RIGHT: 1.5

MODEL:
  policy_name: cma
  INSTRUCTION_ENCODER:
    embedding_file: output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz
    vocab_size: 5395
    final_state_only: false   # CMA 需要所有时间步隐藏状态做 cross-attention
```

---

### Step 3 — 可选 Smoke Run（2 GPU）

先做 2 卡快速 smoke，验证 DDP、checkpoint 和 `latest` 软链：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

GPUS_PER_NODE=2 \
CUDA_DEVICES=0,1 \
MASTER_PORT=29601 \
NUM_EPOCHS=1 \
PER_GPU_BATCH_SIZE=1 \
NUM_WORKERS=0 \
USE_SWANLAB=true \
SWANLAB_MODE=local \
SWANLAB_EXP_NAME=cma-smoke-2gpu \
bash scripts/cma/train_ddp.sh
```

期望结果：

- 2 个 rank 都进入训练
- `output/cma/checkpoints/cma-smoke-2gpu/best.pth` 被写出
- `output/cma/checkpoints/latest -> cma-smoke-2gpu` 软链建立
- `output/cma/logs/cma-smoke-2gpu.log` 存在

---

### Step 4 — 正式 8-GPU 离线训练

建议在 server 98 上用 tmux 启动，默认使用 SwanLab `cloud` 模式：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

tmux new -s cma_ddp

CUDA_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_NODE=8 \
MASTER_PORT=29600 \
PER_GPU_BATCH_SIZE=4 \
LEARNING_RATE=1e-4 \
NUM_EPOCHS=10 \
NUM_WORKERS=4 \
USE_SWANLAB=true \
SWANLAB_MODE=cloud \
SWANLAB_PROJECT=baseline \
bash scripts/cma/train_ddp.sh
```

> 实验名默认为 `cma-ddp-g<N>-bs<B>-lr<LR>-<timestamp>`，可通过 `SWANLAB_EXP_NAME=my-exp` 自定义。

常用 override：

- 自定义实验名：`SWANLAB_EXP_NAME=cma-run1`
- 调整 batch size：`PER_GPU_BATCH_SIZE=2`（显存不足时）
- 切换 SwanLab 模式：`SWANLAB_MODE=local`
- 关闭 SwanLab：`USE_SWANLAB=false`

---

### Step 4.1 — Running Checks

```bash
tmux ls
tail -n 120 output/cma/logs/<EXP_NAME>.log
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv
```

期望现象：

- 8 个 rank 都进入 `offline_trainer`
- SwanLab 日志里出现 `Tracking run with swanlab`
- 8 张卡都有显存占用
- loss 从开头的 `~4` 逐步降低

---

### Step 4.2 — Completion Checks

```bash
readlink output/cma/checkpoints/latest
tail -n 40 output/cma/logs/<EXP_NAME>.log
ls output/cma/checkpoints/latest/
```

期望结果：

- 日志包含 `Offline training completed!`
- `output/cma/checkpoints/latest/best.pth` 存在
- `latest` 软链指向本次实验目录

---

### Step 5 — Evaluate After Training

训练完成后，`latest` 软链已自动更新，直接 eval：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh latest val_seen
```

指定特定实验的 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh <EXP_NAME> val_seen
```

Results saved to `output/cma/results/<EXP_NAME>/val_seen/`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Vocabulary file not found` | 确认 `output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json` 存在；缺失则先跑 seq2seq-train skill Step 2 |
| `embedding_file not found` | 确认 `output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz` 存在；缺失则先跑 seq2seq-train skill Step 4 |
| `CUDA OOM` | 减小 `PER_GPU_BATCH_SIZE`，例如 `4 -> 2 -> 1` |
| 8 个进程都落到 `cuda:0` | 直接用 `train_ddp.sh`，脚本自动注入 `DISTRIBUTED.enabled true` |
| `Please install swanboard` | `pip install swanlab[dashboard]` |
| 企业微信通知初始化失败 | Python 3.8 下 `WxWebhookCallback` 兼容性不稳定；SwanLab 基础记录不受影响 |
| `Error: Checkpoint not found: .../latest/best.pth` | 训练完成前 `latest` 不存在，需先完成一次训练 |
