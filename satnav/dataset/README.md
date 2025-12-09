# SatNav 数据集模块

本模块提供了用于视觉语言导航（VLN）任务的数据集类。

## 数据集类别

### 1. `SatNavDataset` - 基础数据集加载器

**功能：** 从 JSON 文件加载原始的 episode 数据

**特点：**
- 支持从压缩的 JSON 文件（`.json.gz`）或普通 JSON 文件加载数据
- 解析 episode 信息：场景 ID、起始位置、目标点、参考路径、指令文本等
- 支持按场景或 episode ID 过滤数据
- 提供基础的 episode 迭代接口

**使用场景：**
- 数据加载和预处理
- 数据集信息查看和统计
- 作为其他数据集类的基础数据源

**数据格式：**
- 输入：JSON 格式的 episode 数据文件
- 输出：`VLNEpisode` 对象列表

---

### 2. `RecollectionDataset` - 训练数据集

**功能：** 为模仿学习训练实时收集轨迹数据

**特点：**
- 继承自 PyTorch 的 `IterableDataset`
- 使用 `ReferencePathFollower` 从参考路径提取 GT 动作序列
- 在环境中执行 GT 动作，实时收集观测数据（RGB 图像、指令等）
- 自动进行指令的 tokenization 处理
- **自动将 RGB 图像 resize 到 224×224**，减少 GPU 显存占用（约 4-5 倍）
- 支持预加载缓冲区机制，提高训练效率

**内存优化：**
- 默认将 RGB 图像 resize 到 224×224（`target_rgb_size=224`）
- 减少 CPU RAM 使用（preload buffer）
- 减少 GPU 显存占用（ResNet50 处理时）
- 加快 CPU→GPU 数据传输速度
- 与 ResNet50 的 ImageNet 预训练尺寸匹配

**使用场景：**
- 模仿学习（Imitation Learning）训练
- Teacher forcing 训练模式
- 需要实时环境交互的数据收集

**数据流程：**
1. 从 `SatNavDataset` 加载 episode
2. 使用路径跟随器提取 GT 动作序列
3. 在模拟器中执行动作，收集每一步的观测
4. 对指令进行 tokenization
5. 返回训练样本：`(observations, prev_actions, teacher_actions)`

---

## 数据集关系

```
SatNavDataset (基础数据加载)
    ↓
RecollectionDataset (训练数据准备)
    ↓
训练循环 (使用 DataLoader)
```

`RecollectionDataset` 内部使用 `SatNavDataset` 来获取 episode 信息，然后在此基础上进行轨迹收集和数据处理。

## 使用建议

- **数据查看和分析**：使用 `SatNavDataset`
- **模型训练**：使用 `RecollectionDataset`
- **自定义数据集**：可以继承 `SatNavDataset` 或 `RecollectionDataset` 来实现特定需求

