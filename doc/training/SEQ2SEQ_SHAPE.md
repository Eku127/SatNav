# Seq2Seq 数据处理与 Shape 变化全流程

本文档详细梳理了从 `RecollectionDataset` 数据收集到训练 Loss 计算过程中的数据流和 Shape 变化。

假设场景：
- **Batch Size (N)** = 2
- **Trajectory A 长度** = 3
- **Trajectory B 长度** = 2
- **RGB 尺寸** = (224, 224, 3)
- **动作空间** = 4 (STOP, FWD, LEFT, RIGHT)

---

## 1. `_collect_episode` (单条轨迹收集)
在 `RecollectionDataset` 内部，每条轨迹单独收集。

- **Trajectory A (T=3)**:
  - `rgb`: list of 3 tensors -> [(224, 224, 3), (224, 224, 3), (224, 224, 3)]
  - `prev_actions`: list of 3 ints -> [0, a1, a2]
  - `teacher_actions`: list of 3 ints -> [a1, a2, a3]

- **Trajectory B (T=2)**:
  - `rgb`: list of 2 tensors -> [(224, 224, 3), (224, 224, 3)]
  - `prev_actions`: list of 2 ints -> [0, b1]
  - `teacher_actions`: list of 2 ints -> [b1, b2]

---

## 2. `collate_fn` (Batch 组装与 Padding)
`DataLoader` 取出这 2 条轨迹，调用 `collate_fn`。

**第一步：Padding (对齐长度 T_max=3)**
- **Trajectory B** 被补齐到 3 步：
  - `rgb`: 补一张全白/全黑图
  - `actions`: 补 0 (STOP)

**第二步：Stack (堆叠)**
此时 shape 变成了 `(T, N, ...)`:
- `rgb`: `(3, 2, 224, 224, 3)`
- `prev_actions`: `(3, 2)`
- `teacher_actions`: `(3, 2)`

**第三步：Flatten (展平为 T*N)**
除了 `teacher_actions` 外，其他都展平，方便 CNN 处理：
- `rgb`: `(6, 224, 224, 3)`  -> (T*N, H, W, C)
- `prev_actions`: `(6, 1)`    -> (T*N, 1)
- `not_done_masks`: `(6, 1)`  -> (T*N, 1)
  - 内容：`[0, 0, 1, 1, 1, 1]` (每条轨迹第0步是0)

- `teacher_actions` **保留**：`(3, 2)` -> (T, N)

---

## 3. 模型前向 (Seq2SeqPolicy Forward)

输入：`rgb` (6, 224, 224, 3)

**第一步：特征提取 (ResNet + Embedding)**
- `rgb` -> ResNet -> `rgb_features`: `(6, 256)`
- `instruction` -> LSTM -> `inst_features`: `(2, 128)` (指令每条轨迹只有一个)
  - 这里会广播/复制成 `(6, 128)` 以匹配
- `prev_actions` -> Embedding -> `act_features`: `(6, 32)`

**第二步：拼接**
- `x = cat([rgb, inst, act])` -> `(6, 416)` (256+128+32)

**第三步：RNN 处理 (RNNStateEncoder)**
输入 `x` 是 `(6, 416)`，hidden state 是 `(1, 2, 512)`。
1. **Reshape**: 发现 `6 != 2`，拆回 `(3, 2, 416)`
2. **Loop (t=0 to 2)**:
   - **t=0**:
     - Mask `[0, 0]` -> 重置 hidden state 为 0
     - Input `(2, 416)` + Hidden `(2, 512)` -> RNN -> New Hidden `(2, 512)`
     - Output `(2, 512)`
   - **t=1**:
     - Mask `[1, 1]` -> 保持 hidden state
     - Input `(2, 416)` + Hidden `(2, 512)` -> RNN -> New Hidden `(2, 512)`
     - Output `(2, 512)`
   - **t=2**:
     - Mask `[1, 1]` -> 保持 hidden state
     - Input `(2, 416)` + Hidden `(2, 512)` -> RNN -> New Hidden `(2, 512)`
     - Output `(2, 512)`
3. **Flatten Back**:
   - 把 3 个 output 拼回去 -> `(6, 512)`

**第四步：Action Head**
- `(6, 512)` -> Linear -> `logits`: `(6, 4)`

---

## 4. Loss 计算

- **Logits**: `(6, 4)` -> Reshape -> `(3, 2, 4)` (T, N, Actions)
- **Teacher Actions**: `(3, 2)` (T, N)
- **Cross Entropy**:
  - `F.cross_entropy((3, 2, 4), (3, 2))`
  - 计算所有位置的 loss，包括 Pad 的位置 (0)
  - 返回 scalar loss

---

**核心总结**：数据流的关键在于 **Pad对齐 -> 展平处理(CNN) -> 恢复时序(RNN) -> 对齐标签(Loss)**。

