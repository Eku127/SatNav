---
name: seq2seq-train
description: Full seq2seq training pipeline on SatNav ver_260317 data including env/data check, vocab build, GloVe download and embedding build, offline 8-GPU DDP launch, and SwanLab monitoring. Use when the user asks to train seq2seq, run seq2seq full training, 训练seq2seq, 启动seq2seq训练, or prepare seq2seq for training.
---

# Seq2Seq Training (ver_260317)

唯一推荐路径：**离线 `offline_trainer`**，不启动仿真器，直接读取预渲染轨迹数据。

- 训练配置：`configs/baselines/seq2seq_offline.yaml`
- Smoke 配置：`configs/baselines/seq2seq_offline_smoke.yaml`
- 多卡脚本：`scripts/seq2seq/train_offline_ddp.sh`
- 监控：**SwanLab**（默认 `cloud` 模式；可切 `local`）
- 8xH100 推荐起点：
  - `IL.batch_size=8`（per GPU，global batch 64）
  - `IL.lr=3e-4`
  - `IL.OFFLINE.num_workers=8`

## Key Paths

```text
REPO:           /mnt/data1/home/jiangjiajun/workspace/SatNav
DATASET:        /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317
TRAJ_DATA:      /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data
GLOVE_DIR:      /mnt/data3/jiangjiajun/dataset/glove
OUTPUT_ROOT:    output/seq2seq_offline
RUN_LOGS:       output/seq2seq_offline/logs
SWANLAB_LOGDIR: output/seq2seq_offline/swanlab     (仅 local/offline 模式写入)
LATEST_LINK:    output/seq2seq_offline/checkpoints/latest    (训练完成后自动更新)
VOCAB:          output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json
EMBEDDING:      output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz
SMOKE_SUBSET:   output/seq2seq_offline/artifacts/smoke/offline_annotations_128_260317.json
OFFLINE_CONFIG: configs/baselines/seq2seq_offline.yaml
SMOKE_CONFIG:   configs/baselines/seq2seq_offline_smoke.yaml
DDP_SCRIPT:     scripts/seq2seq/train_offline_ddp.sh
SMOKE_SCRIPT:   scripts/seq2seq/make_offline_smoke_subset.py
ORGANIZE_SCRIPT:scripts/seq2seq/organize_output.sh
```

**Output 结构说明：**

```
output/seq2seq_offline/
├── artifacts/                       ← 静态预处理产物（不随实验变化）
│   ├── vocab/train_vocab_<ver>.json
│   ├── embeddings/embeddings_glove50d_<ver>.json.gz
│   └── smoke/offline_annotations_128_<ver>.json
├── checkpoints/
│   ├── latest -> seq2seq-offline-<timestamp>   ← 软链，始终指向最新成功训练
│   └── seq2seq-offline-<timestamp>/best.pth
├── logs/
│   └── seq2seq-offline-<timestamp>.log
├── results/                         ← 由 eval 步骤按需创建
│   └── seq2seq-offline-<timestamp>/
└── swanlab/                         ← 仅 local/offline SwanLab 模式写入
```

Conda env: `satnav`
Working dir: always `REPO` above.

---

## Workflow

### Step 0 — Normalize Existing Output Layout

如果 `output/` 里有历史产物，先整理（同时自动迁移旧版 artifacts 路径）：

```bash
bash scripts/seq2seq/organize_output.sh
```

---

### Step 1 — Environment & Data Check

```bash
bash .codex/skills/seq2seq-train/scripts/check_env.sh
```

检查项：

- conda env / torch / GPU visibility
- train/eval episode files
- offline annotations / image directory
- vocab / GloVe / embedding
- offline config 和 DDP 脚本
- `latest` 软链状态
- SwanLab importability

Expected: all GREEN. Fix any RED items before continuing.

---

### Step 2 — Vocab

检查是否存在：

```bash
ls output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

如果 MISSING，构建：

```bash
mkdir -p output/seq2seq_offline/artifacts/vocab
python -m satnav.utils.build_vocab \
  --dataset /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/train/all_episodes.json \
  --output output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json
```

Expected vocab size: `5395`.

---

### Step 3 — GloVe Download (if absent)

检查：

```bash
ls /mnt/data3/jiangjiajun/dataset/glove/glove.6B.50d.txt 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

如果 MISSING：

```bash
bash .codex/skills/seq2seq-train/scripts/download_glove.sh
```

---

### Step 4 — Build GloVe Embeddings

检查：

```bash
ls output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

如果 MISSING：

```bash
bash .codex/skills/seq2seq-train/scripts/build_embeddings.sh
```

Expected coverage: `60-70%`.

---

### Step 5 — Verify Offline Config

`configs/baselines/seq2seq_offline.yaml` 应满足：

```yaml
TRAINER_NAME: offline_trainer

IL:
  lr: 3e-4
  batch_size: 8
  OFFLINE:
    annotations_path: /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json
    images_root: /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/images

MODEL:
  INSTRUCTION_ENCODER:
    embedding_file: output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz
    vocab_size: 5395
```

并确认：

- `BASE_TASK_CONFIG_PATH: configs/satnav_task.yaml`
- `SWANLAB.mode` 可被脚本 override
- `DISTRIBUTED.enabled` 在脚本中自动注入 `true`

---

### Step 6 — Optional Smoke Run (2 GPU)

先做 2 卡 smoke，验证 DDP、checkpoint 和 `latest` 软链：

```bash
python scripts/seq2seq/make_offline_smoke_subset.py \
  --src /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/trajectory_data/annotations.json \
  --dst output/seq2seq_offline/artifacts/smoke/offline_annotations_128_260317.json \
  --count 128 \
  --per-scene 2

source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

GPUS_PER_NODE=2 \
CUDA_DEVICES=0,1 \
MASTER_PORT=29531 \
NUM_EPOCHS=1 \
PER_GPU_BATCH_SIZE=1 \
NUM_WORKERS=0 \
USE_SWANLAB=true \
SWANLAB_MODE=local \
SWANLAB_EXP_NAME=seq2seq-offline-smoke-2gpu \
CONFIG_PATH=configs/baselines/seq2seq_offline_smoke.yaml \
bash scripts/seq2seq/train_offline_ddp.sh
```

期望结果：

- 2 个 rank 都进入训练
- `output/seq2seq_offline/checkpoints/seq2seq-offline-smoke-2gpu/best.pth` 被写出
- `output/seq2seq_offline/checkpoints/latest -> seq2seq-offline-smoke-2gpu` 软链建立
- `output/seq2seq_offline/logs/seq2seq-offline-smoke-2gpu.log` 存在
- `swanlab watch output/seq2seq_offline/swanlab` 可以看到本地实验

---

### Step 7 — Launch 8-GPU Offline Training

正式训练建议在 server 98 上用 tmux 启动，默认使用 SwanLab `cloud` 模式。

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

tmux new -s seq2seq_offline_ddp

CUDA_DEVICES=0,1,2,3,4,5,6,7 \
GPUS_PER_NODE=8 \
MASTER_PORT=29540 \
PER_GPU_BATCH_SIZE=8 \
LEARNING_RATE=3e-4 \
NUM_EPOCHS=10 \
NUM_WORKERS=8 \
USE_SWANLAB=true \
SWANLAB_MODE=cloud \
SWANLAB_PROJECT=baseline \
bash scripts/seq2seq/train_offline_ddp.sh
```

> 实验名默认为 `seq2seq-offline-ddp-g<N>-bs<B>-lr<LR>-<timestamp>`，可通过 `SWANLAB_EXP_NAME=my-exp` 自定义。

可选：同时开本地 watch 看板：

```bash
tmux new -s swanlab_seq2seq
swanlab watch output/seq2seq_offline/swanlab -h 0.0.0.0 -p 5092
```

常用 override：

- 自定义实验名：`SWANLAB_EXP_NAME=my-exp`
- 改输出根目录：`OUTPUT_ROOT=output/seq2seq_offline_debug`
- 切换 SwanLab 模式：`SWANLAB_MODE=local`
- 关闭 SwanLab：`USE_SWANLAB=false`
- 临时改参数：`PER_GPU_BATCH_SIZE=4 LEARNING_RATE=2e-4`

局域网访问（开了 local watch）：`http://10.246.152.98:5092`

---

### Step 7.1 — Running Checks

```bash
tmux ls
tail -n 120 output/seq2seq_offline/logs/<EXP_NAME>.log
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv
```

期望现象：

- 8 个 rank 都进入 `offline_trainer`
- SwanLab 日志里出现 `Tracking run with swanlab`
- 8 张卡都有显存占用，长轨迹时高卡可接近 `80GB`
- loss 从开头的 `~4` 逐步降到 `~2` 附近

---

### Step 7.2 — Completion Checks

```bash
# 查看最新训练的 EXP_NAME
readlink output/seq2seq_offline/checkpoints/latest

tail -n 40 output/seq2seq_offline/logs/<EXP_NAME>.log
ls output/seq2seq_offline/checkpoints/latest/
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

期望结果：

- 日志包含 `Offline training completed!`
- `output/seq2seq_offline/checkpoints/latest/best.pth` 存在
- `latest` 软链指向本次实验目录
- GPU 回到空闲状态

已验证过的一次完整运行参考：

- 实验名：`seq2seq-offline-0317-full-20260320-164653`（旧命名格式）
- 最终结果：`Best loss: 0.7765`

---

### Step 8 — Evaluate After Training

训练完成后，`latest` 软链已自动更新，直接 eval：

```bash
python run.py \
  --exp-config configs/baselines/seq2seq_offline.yaml \
  --run-type eval \
  EVAL.SPLIT val_seen
```

指定特定实验的 checkpoint：

```bash
python run.py \
  --exp-config configs/baselines/seq2seq_offline.yaml \
  --run-type eval \
  EVAL.SPLIT val_seen \
  EVAL.CKPT_PATH output/seq2seq_offline/checkpoints/<EXP_NAME>/best.pth
```

Results saved to `output/seq2seq_offline/results/<EXP_NAME>/`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Loading task config: configs/debug_vln_task.yaml` | `BASE_TASK_CONFIG_PATH` 必须在 YAML 里设置，不能通过 CLI |
| `Vocabulary file not found` | 运行 Step 2 构建 vocab |
| `embedding_file not found` | 运行 Step 4 构建 embeddings |
| `CUDA OOM` | 减小 `PER_GPU_BATCH_SIZE`，例如 `8 -> 4` |
| `TRAINER_NAME not specified` | 确认 `_base_: configs/default.yaml` |
| 8 个进程都落到 `cuda:0` | 直接用 `train_offline_ddp.sh`，脚本自动注入 `DISTRIBUTED.enabled true` |
| `Please install swanboard` | `pip install swanlab[dashboard]` |
| 企业微信通知初始化失败 | Python 3.8 下 `WxWebhookCallback` 兼容性不稳定；SwanLab 基础记录不受影响 |
| `Error: Checkpoint not found: .../latest/best.pth` | 训练完成前 `latest` 不存在，需先跑一次训练或手动建软链 |
| eval 后 `results/` 目录不存在 | 正常——`results/` 由 evaluator 按需创建，不再由训练脚本预建 |
