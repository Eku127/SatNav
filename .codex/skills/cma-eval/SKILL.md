---
name: cma-eval
description: Evaluate a trained CMA model on SatNav ver_260317 data using the online simulator. Supports eval by experiment name or checkpoint path, val_seen / val_unseen splits, and quick debug with limited episodes. Use when the user asks to eval cma, evaluate cma model, 评测cma, 运行cma评测, 查看cma结果, or mentions cma evaluation results.
---

# CMA Evaluation (ver_260317)

推荐入口：**`scripts/cma/eval.sh`**，调用 `run.py --run-type eval`（`offline_trainer.eval()`），
使用 **在线仿真器** 对完整 episode 进行 rollout，记录 SR / SPL / NE / path_length。

- 评测配置：`configs/baselines/cma_eval.yaml`
- 入口脚本：`scripts/cma/eval.sh`
- 依赖 `satnav` conda 环境（脚本内自动 activate）

## Key Paths

```text
REPO:           /mnt/data1/home/jiangjiajun/workspace/SatNav
DATASET:        /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317
SCENES_DIR:     /mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes
EVAL_EPISODES:  /mnt/data3/jiangjiajun/dataset/satnav_datasets/ver_260317/episodes/eval/{split}/all_episodes.json
                ├── val_seen/all_episodes.json    (7801 episodes: Amsterdam-1, Rome-1, NewYork-1)
                └── val_unseen/all_episodes.json  (当前暂无城市，后续引入新城市后生成)
VOCAB:          output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json
EMBEDDING:      output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz
CHECKPOINTS:    output/cma/checkpoints/
LATEST_LINK:    output/cma/checkpoints/latest        (软链，始终指向最新成功训练)
RESULTS_ROOT:   output/cma/results/
EVAL_CONFIG:    configs/baselines/cma_eval.yaml
EVAL_SCRIPT:    scripts/cma/eval.sh
```

**Output 结构：**

```
output/cma/results/
└── <EXP_NAME>/
    └── val_seen/
        ├── eval_ckpt_0_val_seen.json    ← 汇总指标（SR / SPL / NE / path_length）
        └── eval.log                     ← 完整运行日志
```

Conda env: `satnav`
Working dir: always `REPO` above.

---

## Workflow

### Step 1 — 确认 Checkpoint 存在

```bash
# 查看最新训练实验名
readlink output/cma/checkpoints/latest

# 确认 best.pth 存在
ls output/cma/checkpoints/latest/best.pth

# 或指定具体实验
ls output/cma/checkpoints/<EXP_NAME>/best.pth
```

如果 `latest` 不存在：先运行训练（参考 cma-train skill）。

---

### Step 2 — 快速调试（限制 episodes 数量）

用少量 episodes 验证 eval 流程是否正常：

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

# 用 latest checkpoint，跑 10 个 episode
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh latest val_seen 10
```

期望输出：
- 打印每条 episode 的 SPL / Success 进度
- `output/cma/results/latest/val_seen/eval_ckpt_0_val_seen.json` 写出
- 无报错退出（exit code 0）

---

### Step 3 — 完整 val_seen 评测

```bash
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate satnav
cd /mnt/data1/home/jiangjiajun/workspace/SatNav

# 按实验名评测（推荐）
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh <EXP_NAME> val_seen

# 等价：用 latest checkpoint
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh latest val_seen
```

7801 episodes，建议在 `tmux` 里运行：

```bash
tmux new -s cma_eval
CUDA_VISIBLE_DEVICES=0 bash scripts/cma/eval.sh <EXP_NAME> val_seen \
  2>&1 | tee output/cma/results/<EXP_NAME>/val_seen/eval.log
```

---

### Step 4 — 查看结果

```bash
# 查看汇总指标
cat output/cma/results/<EXP_NAME>/val_seen/eval_ckpt_0_val_seen.json

# 查看运行日志末尾（最终指标打印在此）
tail -n 30 output/cma/results/<EXP_NAME>/val_seen/eval.log
```

结果 JSON 格式：

```json
{
    "spl": 0.0,
    "success": 0.0,
    "distance_to_goal": 255.4,
    "path_length": 987.7,
    "steps_taken": 500.0,
    "num_episodes": 10,
    "split": "val_seen",
    "checkpoint_index": 0
}
```

关键指标含义：

| 字段 | 含义 |
|---|---|
| `success` | Success Rate (SR)：成功到达目标的比例 |
| `spl` | SPL：Success weighted by Path Length |
| `distance_to_goal` | NE：平均导航误差（米），越低越好 |
| `path_length` | 平均轨迹长度（米） |
| `steps_taken` | 平均步数（达到 500 = 未学会 STOP） |

---

## 脚本参数说明

`scripts/cma/eval.sh <exp_name_or_ckpt> [split] [max_episodes]`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `exp_name_or_ckpt` | — | 实验名（在 `checkpoints/` 下找 `best.pth`）或 `.pth` 文件绝对路径 |
| `split` | `val_seen` | 评测 split：`val_seen` / `val_unseen` |
| `max_episodes` | 全部 | 限制 episode 数量（调试用） |

环境变量 override：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CONFIG_PATH` | `configs/baselines/cma_eval.yaml` | 评测配置文件路径 |
| `OUTPUT_ROOT` | `output/cma` | 输出根目录 |
| `CUDA_DEVICES` | `0` | 使用的 GPU（eval 为单卡） |

示例：

```bash
# 换 GPU
CUDA_DEVICES=1 bash scripts/cma/eval.sh <EXP_NAME> val_seen

# 自定义输出目录
OUTPUT_ROOT=output/cma_debug bash scripts/cma/eval.sh <EXP_NAME> val_seen 50

# 直接指定 .pth 路径（backward compatible）
bash scripts/cma/eval.sh /abs/path/to/best.pth val_seen
```

---

## 直接用 run.py（高级用法）

```bash
python run.py \
  --exp-config configs/baselines/cma_eval.yaml \
  --run-type eval \
  EVAL.SPLIT val_seen \
  EVAL.CKPT_PATH output/cma/checkpoints/<EXP_NAME>/best.pth \
  RESULTS_DIR output/cma/results/<EXP_NAME>/val_seen \
  EVAL.EPISODE_COUNT 20
```

---

## Eval 内部流程说明

```
run.py --run-type eval
  └─ offline_trainer.eval()
       ├─ 1. OfflineTrajectoryDataset  ← 仅用于初始化 policy 的 obs/action space
       │       (加载 annotations.json，不加载图片)
       ├─ 2. _initialize_policy()      ← 创建 CMAPolicy（ResNet50 + BiLSTM + GRU + Cross-Attn）
       ├─ 3. Evaluator.evaluate_checkpoint()
       │       ├─ load_state_dict(best.pth)    ← 加载 checkpoint 权重
       │       ├─ SatNavDataset(eval episodes) ← 7801 val_seen episodes
       │       ├─ Env(cycle=False)             ← 在线仿真器
       │       ├─ _load_vocabulary()           ← vocab/train_vocab_260317.json
       │       └─ episode rollout loop
       │               ├─ env.reset()
       │               ├─ _tokenize_instruction()   ← text → token indices
       │               ├─ policy.act(deterministic=True)
       │               └─ env.step(action)
       └─ 4. _save_results()           ← eval_ckpt_0_val_seen.json
```

关键约定：
- `DATASET.DATA_PATH` 使用 `{split}` 占位符，由 `SatNavDataset` 运行时展开
- `val_unseen` 当前暂无 episodes（城市划分：全部 3 个 eval 城市均为 val_seen）
- 评测为单卡；不支持多卡分布式 rollout
- 每条 episode 最多 500 步（`ENVIRONMENT.MAX_EPISODE_STEPS`）
- SUCCESS_DISTANCE: Road=10m, Boundary=10m, LandmarkSet=3m

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Checkpoint not found: .../best.pth` | 先运行训练（cma-train skill）；或确认 `latest` 软链指向已完成训练的目录 |
| `FileNotFoundError: .../val_unseen/all_episodes.json` | 当前无 val_unseen 数据，只能评 `val_seen` |
| `Vocabulary file not found` | 确认 `output/seq2seq_offline/artifacts/vocab/train_vocab_260317.json` 存在；缺失则运行 seq2seq-train skill Step 2 |
| `embedding_file not found` | 确认 `output/seq2seq_offline/artifacts/embeddings/embeddings_glove50d_260317.json.gz` 存在；缺失则运行 seq2seq-train skill Step 4 |
| `steps_taken: 500`（所有 episode 走满步） | 模型未学会 STOP；调整 `class_weights.STOP` 或延长训练轮次后重训 |
| `distance_to_goal` 极大（>200m） | 模型导航能力差；检查训练 loss 曲线是否收敛 |
| `CUDA OOM` | eval 为单卡单 episode，通常不 OOM；若出现则检查其他进程占用 GPU 显存 |
| `RuntimeError: All episodes exhausted` | `EVAL.EPISODE_COUNT` 超出实际 episode 数量；设为 `-1` 或减小数值 |
| Scene map 相关 TIF 文件报错 | 确认 `DATASET.SCENES_DIR=/mnt/data3/.../scenes` 存在，且包含对应城市 TIF |
