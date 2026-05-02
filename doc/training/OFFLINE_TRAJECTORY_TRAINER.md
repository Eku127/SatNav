# 离线轨迹训练方案（可实施版）

基于 `ver_260317` 的 `trajectory_data/`，新增 `offline_trainer`，在训练阶段完全绕开仿真器，支持单机多卡 DDP。

---

## 1. 目标

当前 `RecollectTrainer` 的训练路径是：

```text
episode JSON -> ReferencePathFollower -> SatSim 渲染 -> 收集轨迹 -> GPU 更新
```

瓶颈明确：

1. 仿真器渲染是 CPU 密集型，GPU 经常等待数据。
2. 训练数据按 episode 串行生成，多卡扩展性差。

`ver_260317/trajectory_data` 已经提供预渲染轨迹和动作序列，因此可以替换为：

```text
annotations.json -> OfflineTrajectoryDataset -> DataLoader -> GPU / DDP 更新
```

---

## 2. 真实数据结论

已在仓库环境中抽样确认：

- `trajectory_data/annotations.json` 共 `94986` 条记录。
- 每条记录都包含 `video / instructions / actions`。
- `rgb/*.jpg` 实际数量与 `actions` 长度一致。
- 最后一帧是执行 `STOP` 后的重复帧，和前一帧相同。

示例：

```json
{
  "id": 0,
  "trajectory_id": "0",
  "steps": 24,
  "video": "images/Amsterdam-2_satnav_000000",
  "instructions": ["From the initial T-shaped intersection..."],
  "actions": [-1, 1, 1, ..., 1, 0]
}
```

---

## 3. 和当前在线训练保持一致的序列定义

这是离线实现最关键的地方。

当前 `RecollectTrainer` 不会把最终 `STOP` 作为 teacher action 参与训练，因此离线版必须遵守同一语义。

假设：

```text
actions = [-1, a1, a2, ..., ak, STOP]
frames  = [f0, f1, f2, ..., fk, f_stop]
```

其中：

- `f0` 是初始观测
- `f_stop` 与 `fk` 重复

离线训练应构造：

- `T = k`
- `observations = [f0, f1, ..., f{k-1}]`
- `prev_actions = [0, a1, ..., a{k-1}]`
- `teacher_actions = [a1, a2, ..., ak]`

也就是：

- 丢弃开头 `START=-1`
- 丢弃结尾 `STOP=0`
- 丢弃最后一张重复的 post-STOP 帧

这样才能和当前 `RecollectTrainer` 对齐。

---

## 4. 实现约束

### 4.1 observation 格式

离线数据集输出必须兼容现有 `collate_fn`：

- `observations["rgb"]`: `Tensor(T, H, W, C)`
- `observations["instruction"]`: `Tensor(T, max_instruction_len)`
- `prev_actions`: `Tensor(T,)`
- `teacher_actions`: `Tensor(T,)`
- `weights`: `Tensor(T,)`

注意：

- `instruction` 不能只返回 `(max_instruction_len,)`
- 现有 `collate_fn` 把每个 sensor 的第 0 维都视为时间维，因此离线版必须把同一条指令沿时间维重复 `T` 次

### 4.2 图像预处理

默认把 `448x448` 轨迹帧 resize 到 `224x224`：

- 与当前在线训练默认 observation space 一致
- 显著降低 CPU RAM、pin memory 和 GPU 显存压力
- 更适合 smoke test 和初版多卡落地

不要在 dataset 里做 ImageNet normalize。当前视觉编码器已经在前向内部负责：

```text
[0,255] uint8 -> [0,1] -> 可选 ImageNet normalize
```

因此 dataset 只返回 `uint8` 的 HWC 图像。

### 4.3 权重

在线路径已实现 inflection weighting，但旧版 `collate_fn` 没有真正传递 `weights`。

离线方案里一并修复：

- `collate_fn` 支持可选第 5 个返回值 `weights`
- `BaseILTrainer._update_agent()` 支持 `(T,)`、`(T,1)`、`(T,N)` 三种权重形状

---

## 5. 落地文件

### 新增

```text
satnav/dataset/offline_trajectory_dataset.py
satnav/training/offline_trainer.py
satnav/training/distributed.py
configs/baselines/seq2seq_offline_train.yaml
scripts/seq2seq/train_offline_ddp.sh
```

### 修改

```text
satnav/training/utils.py
satnav/training/base_il_trainer.py
satnav/dataset/__init__.py
satnav/training/__init__.py
```

---

## 6. 配置约定

使用独立 baseline 配置，避免污染现有在线训练配置：

```yaml
TRAINER_NAME: offline_trainer

IL:
  batch_size: 2
  OFFLINE:
    annotations_path: /.../trajectory_data/annotations.json
    images_root: /.../trajectory_data/images
    num_workers: 4
    pin_memory: true
    rgb_size: 224
    max_instruction_len: 200
    max_traj_len: 500
```

说明：

- `IL.batch_size` 仍然是 **每卡 batch**
- 8 卡下若配置 `batch_size: 2`，全局 batch 才是 `16`
- 初版不建议直接把每卡 batch 提到 `32`

---

## 7. 注册方式

不在 `run.py` 单点手工导入。

实际实现采用和现有 trainer 一致的注册方式：

- 在 `satnav/training/__init__.py` 中导入 `OfflineTrainer`
- 依赖 `@register_trainer("offline_trainer")` 完成注册

这样更符合当前仓库结构。

---

## 8. 训练与评估关系

- **训练**：可使用 `offline_trainer`
- **评估**：仍使用现有 `Evaluator` + 仿真器环境
- **checkpoint**：与在线训练格式兼容，可直接复用

也就是说：

- 离线训练负责提速
- 评估语义不变，仍以真实环境 rollout 为准

---

## 9. 冒烟测试建议

使用 `configs/baselines/seq2seq_offline_train.yaml`，并通过 CLI override 指向 smoke annotations。

步骤：

1. 复用/生成 train vocab
2. 从 `trajectory_data/annotations.json` 截取 `128` 条离线轨迹
3. 单卡训练 `1` 个 epoch
4. 验证 checkpoint 写出

建议命令（需先在配置中指向已准备好的 smoke annotations）：

```bash
python run.py \
  --exp-config configs/baselines/seq2seq_offline_train.yaml \
  --run-type train \
  IL.epochs 1 \
  IL.batch_size 1 \
  IL.OFFLINE.annotations_path output/seq2seq_offline/artifacts/smoke/offline_annotations_128_260404.json \
  IL.OFFLINE.num_workers 0 \
  CHECKPOINT_FOLDER output/seq2seq_offline/checkpoints/seq2seq_offline_smoke
```

通过标准：

- `Trainer: offline_trainer`
- `Creating offline dataset...`
- `Initializing policy...`
- `Epoch 1/1` 正常结束
- `output/seq2seq_offline/checkpoints/seq2seq_offline_smoke/best.pth` 写出

---

## 10. 当前结论

该方案可行，且已经具备工程落地条件。

关键不是“能不能做”，而是必须遵守以下三点：

1. 严格对齐在线训练的动作/帧语义，排除结尾 `STOP`
2. `instruction` 必须沿时间维重复，保持 `collate_fn` 兼容
3. 默认用 `224` resize 控资源，再逐步放大 batch 和分辨率

在这三个前提下，离线训练是当前 SatNav 上最直接、风险最低的提速路径。
