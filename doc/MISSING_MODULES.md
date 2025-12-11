# SatNav 缺失模块分析

## 文档说明

本文档基于对 SatNav、VLN-CE 和 Habitat-Lab 三个仓库的深入分析，识别出 SatNav 平台目前缺少的必要模块，并提供了实现优先级建议。

**分析日期**: 2024年12月

**参考仓库**:
- **SatNav**: 连续状态 VLN 评测平台（当前仓库）
- **VLN-CE**: Vision-and-Language Navigation in Continuous Environments
- **Habitat-Lab**: Facebook Research 的 Embodied AI 框架

---

## 1. 核心缺失模块

### 1.1 训练框架（Training Infrastructure）
**优先级**: ⭐⭐⭐⭐⭐ (最高)

**现状**: SatNav 目前只有评测环境，完全缺少训练能力。

**需要实现**:

#### 1.1.1 训练器基类
- **文件**: `satnav/trainers/base_trainer.py`
- **功能**:
  - 训练循环管理（epoch、iteration）
  - Checkpoint 保存和加载
  - 日志记录（控制台、文件）
  - 学习率调度
  - 梯度裁剪和优化器管理
  - 参考 VLN-CE 的 `BaseVLNCETrainer`

#### 1.1.2 具体训练器实现
- **DaggerTrainer** (`satnav/trainers/dagger_trainer.py`)
  - DAgger (Dataset Aggregation) 训练策略
  - 在线数据收集和模型更新
  - 专家策略集成
  
- **RecollectTrainer** (`satnav/trainers/recollect_trainer.py`)
  - 使用参考路径的 teacher forcing 训练
  - 适合快速原型验证
  
- **ILTrainer** (`satnav/trainers/il_trainer.py`)
  - 通用模仿学习训练器
  - 支持多种训练策略

#### 1.1.3 训练配置管理
- **文件**: `satnav/config/training_config.py`
- **配置项**:
  - 学习率、学习率调度策略
  - 批次大小、梯度累积步数
  - 优化器类型和参数（Adam、SGD等）
  - 训练轮数、验证频率
  - 早停策略

**参考实现**: 
- VLN-CE: `vlnce_baselines/common/base_il_trainer.py`
- VLN-CE: `vlnce_baselines/dagger_trainer.py`

---

### 1.2 基线模型（Baseline Models）
**优先级**: ⭐⭐⭐⭐⭐ (最高)

**现状**: 完全没有可训练的模型实现。

**需要实现**:

#### 1.2.1 策略网络基类
- **文件**: `satnav/models/policy.py`
- **功能**:
  - 定义策略网络接口
  - 前向传播逻辑
  - 动作预测
  - 参考 Habitat-Lab 的 `Policy` 基类

#### 1.2.2 编码器模块
- **指令编码器** (`satnav/models/encoders/instruction_encoder.py`)
  - LSTM/GRU 序列编码器
  - 可选：Transformer 编码器
  - 词嵌入层（支持预训练词向量）
  
- **视觉编码器** (`satnav/models/encoders/visual_encoder.py`)
  - ResNet/CNN 图像编码器
  - 支持预训练权重（ImageNet）
  - 特征提取和池化
  
- **状态编码器** (`satnav/models/encoders/state_encoder.py`)
  - RNN 状态编码器（GRU/LSTM）
  - 历史状态记忆

#### 1.2.3 具体模型实现
- **Seq2SeqPolicy** (`satnav/models/seq2seq_policy.py`)
  - 序列到序列模型
  - 最简单的基线模型
  - 适合快速验证训练流程
  
- **CMAPolicy** (`satnav/models/cma_policy.py`)
  - Cross-Modal Attention 模型
  - 视觉-语言注意力机制
  - VLN-CE 的标准基线

#### 1.2.4 模型工具
- **文件**: `satnav/models/utils.py`
- **功能**:
  - 模型权重初始化
  - Checkpoint 保存/加载
  - 模型参数统计
  - 模型可视化（可选）

**参考实现**:
- VLN-CE: `vlnce_baselines/models/seq2seq_policy.py`
- VLN-CE: `vlnce_baselines/models/cma_policy.py`
- VLN-CE: `vlnce_baselines/models/encoders/`

---

### 1.3 向量化环境（Vectorized Environment）
**优先级**: ⭐⭐⭐⭐ (高)

**现状**: 当前 `Env` 类是单进程实现，训练效率低。

**需要实现**:

#### 1.3.1 VectorEnv 实现
- **文件**: `satnav/core/vector_env.py`
- **功能**:
  - 多进程/多线程并行环境
  - 同步 reset/step 操作
  - 自动 episode 重置
  - 进程间通信（Pipe/Queue）
  - 参考 Habitat-Lab 的 `VectorEnv`

#### 1.3.2 环境构建工具
- **文件**: `satnav/utils/env_utils.py`
- **函数**:
  - `construct_envs()`: 批量创建向量化环境
  - GPU 设备分配
  - 场景分配策略（负载均衡）
  - 配置管理

**参考实现**:
- Habitat-Lab: `habitat/core/vector_env.py`
- VLN-CE: `vlnce_baselines/common/env_utils.py`

**性能提升**: 多环境并行可以显著加速训练（4-8倍速度提升）。

---

### 1.4 评估框架（Evaluation Framework）
**优先级**: ⭐⭐⭐ (中)

**现状**: 虽然有 `measures.py` 实现指标，但缺少完整的评估流程。

**需要实现**:

#### 1.4.1 评估器
- **文件**: `satnav/eval/evaluator.py`
- **功能**:
  - 批量运行评估
  - 指标聚合和统计
  - 结果保存（JSON/YAML格式）
  - 支持不同数据集划分（train/val_seen/val_unseen/test）

#### 1.4.2 评估脚本
- **文件**: `satnav/eval/run_eval.py`
- **功能**:
  - 命令行接口
  - Checkpoint 加载
  - 批量评估
  - 结果输出

#### 1.4.3 结果分析工具
- **文件**: `satnav/eval/analyzer.py`
- **功能**:
  - 失败案例分析
  - 路径可视化
  - 统计报告生成
  - 对比分析（不同模型/配置）

**参考实现**:
- VLN-CE: `vlnce_baselines/common/base_il_trainer.py` 中的 `eval()` 方法

---

### 1.5 数据增强（Data Augmentation）
**优先级**: ⭐⭐⭐ (中)

**现状**: 完全没有数据增强功能。

**需要实现**:

#### 1.5.1 路径增强
- **文件**: `satnav/augmentation/path_augmentation.py`
- **功能**:
  - 路径扰动（添加噪声）
  - 路径采样（从参考路径采样新路径）
  - 路径简化/复杂化

#### 1.5.2 指令增强
- **文件**: `satnav/augmentation/instruction_augmentation.py`
- **功能**:
  - 同义词替换
  - 指令重写（可选，需要NLP工具）
  - 指令扩展/压缩

#### 1.5.3 视觉增强
- **文件**: `satnav/augmentation/visual_augmentation.py`
- **注意**: 需要谨慎设计，避免破坏地理信息
- **功能**:
  - 颜色抖动（亮度、对比度）
  - 随机裁剪（需确保不超出地图边界）
  - 噪声添加

**参考实现**:
- VLN-CE: `vlnce_baselines/common/aux_losses.py` (部分增强逻辑)

---

### 1.6 回放缓冲区（Replay Buffer）
**优先级**: ⭐⭐⭐ (中)

**现状**: 没有轨迹存储和管理机制。

**需要实现**:

#### 1.6.1 RolloutStorage
- **文件**: `satnav/training/rollout_storage.py`
- **功能**:
  - 存储 (obs, action, reward, done) 序列
  - 支持批量采样
  - 支持 teacher forcing 数据
  - 内存高效管理

#### 1.6.2 轨迹管理
- **文件**: `satnav/training/trajectory_manager.py`
- **功能**:
  - 轨迹保存/加载（磁盘）
  - 轨迹过滤（按成功率、路径长度等）
  - 轨迹可视化
  - 轨迹统计分析

**参考实现**:
- VLN-CE: `vlnce_baselines/common/rollout_storage.py`
- VLN-CE: `vlnce_baselines/common/recollection_dataset.py`

---

### 1.7 日志和可视化（Logging & Visualization）
**优先级**: ⭐⭐⭐ (中)

**现状**: 只有基本的控制台输出。

**需要实现**:

#### 1.7.1 TensorBoard 集成
- **文件**: `satnav/utils/tensorboard_utils.py`
- **功能**:
  - 训练曲线记录（loss、metrics）
  - 图像可视化（观测、预测路径）
  - 超参数记录
  - 模型图可视化

#### 1.7.2 WandB 支持（可选）
- **文件**: `satnav/utils/wandb_utils.py`
- **功能**:
  - 实验跟踪
  - 超参数搜索
  - 结果对比

#### 1.7.3 视频生成工具
- **文件**: `satnav/utils/video_utils.py`
- **功能**:
  - 轨迹视频生成
  - 对比视频（预测路径 vs 参考路径）
  - 失败案例视频

**参考实现**:
- Habitat-Lab: `habitat_baselines/common/tensorboard_utils.py`

---

### 1.8 配置系统增强（Enhanced Config System）
**优先级**: ⭐⭐ (低-中)

**现状**: 配置系统较基础，只有任务配置。

**需要实现**:

#### 1.8.1 实验配置管理
- **文件**: `satnav/config/experiment_config.py`
- **功能**:
  - 训练/评估/推理配置分离
  - 配置继承和覆盖机制
  - 命令行参数解析
  - 配置验证

#### 1.8.2 模型配置
- **文件**: `satnav/config/model_config.py`
- **功能**:
  - 模型架构配置
  - 编码器配置（指令、视觉、状态）
  - 超参数配置

**参考实现**:
- VLN-CE: `vlnce_baselines/config/default.py`

---

### 1.9 推理框架（Inference Framework）
**优先级**: ⭐⭐⭐ (中)

**现状**: 没有标准化的推理流程。

**需要实现**:

#### 1.9.1 推理器
- **文件**: `satnav/inference/inferencer.py`
- **功能**:
  - 批量推理
  - 结果格式化
  - 支持不同输出格式（R2R、RxR、SatNav等）
  - Checkpoint 加载

#### 1.9.2 推理脚本
- **文件**: `satnav/inference/run_inference.py`
- **功能**:
  - 命令行接口
  - 测试集推理
  - 结果保存（JSON/JSONL格式）

**参考实现**:
- VLN-CE: `vlnce_baselines/common/base_il_trainer.py` 中的 `inference()` 方法

---

### 1.10 非学习智能体（Non-Learning Agents）
**优先级**: ⭐⭐ (低)

**现状**: 只有 `PathFollower`，缺少其他基线智能体。

**需要实现**:

#### 1.10.1 随机智能体
- **文件**: `satnav/agents/random_agent.py`
- **功能**: 随机动作选择，用于基线对比

#### 1.10.2 最短路径跟随器
- **文件**: `satnav/agents/shortest_path_follower.py`
- **功能**: 使用参考路径的oracle智能体

#### 1.10.3 贪心智能体
- **文件**: `satnav/agents/greedy_agent.py`
- **功能**: 贪心策略（总是朝目标方向移动）

**参考实现**:
- VLN-CE: `vlnce_baselines/nonlearning_agents.py`

---

## 2. 实现优先级建议

### 2.1 第一阶段：核心训练能力（必须）
**目标**: 实现基本的训练-评估流程

1. ✅ **训练框架**
   - BaseTrainer 基类
   - DaggerTrainer 实现
   - 基础训练配置

2. ✅ **基线模型**
   - Seq2SeqPolicy（最简单的模型）
   - 指令编码器 + 视觉编码器 + 状态编码器
   - 动作预测头

3. ✅ **评估框架**
   - Evaluator 类
   - 批量评估脚本
   - 结果保存

**预期时间**: 2-3周

**验证标准**: 能够成功训练一个简单模型并在验证集上评估

---

### 2.2 第二阶段：提升训练效率（重要）
**目标**: 加速训练，提升开发体验

1. ✅ **向量化环境**
   - VectorEnv 实现
   - 多进程环境支持
   - 环境构建工具

2. ✅ **回放缓冲区**
   - RolloutStorage
   - 轨迹管理

3. ✅ **日志和可视化**
   - TensorBoard 集成
   - 训练曲线可视化
   - 视频生成

**预期时间**: 1-2周

**性能目标**: 训练速度提升 4-8倍

---

### 2.3 第三阶段：增强功能（推荐）
**目标**: 提升模型性能和训练稳定性

1. ✅ **数据增强**
   - 路径增强
   - 指令增强
   - 视觉增强（谨慎实现）

2. ✅ **推理框架**
   - 推理器实现
   - 测试集推理脚本

3. ✅ **更多基线模型**
   - CMAPolicy

**预期时间**: 2-3周

---

### 2.4 第四阶段：完善生态（可选）
**目标**: 提升平台完整性和易用性

1. ✅ **非学习智能体**
   - 随机智能体
   - 最短路径跟随器
   - 贪心智能体

2. ✅ **配置系统增强**
   - 实验配置管理
   - 模型配置系统

3. ✅ **文档和示例**
   - 训练教程
   - 模型开发指南
   - API 文档

**预期时间**: 1-2周

---

## 3. 模块依赖关系

```
训练框架 (1.1)
    ├── 基线模型 (1.2) ──┐
    │                    │
    ├── 向量化环境 (1.3) │
    │                    │
    ├── 回放缓冲区 (1.6) │──> 完整训练流程
    │                    │
    └── 日志可视化 (1.7) │
                         │
评估框架 (1.4) ──────────┘
    │
    └──> 推理框架 (1.9)

数据增强 (1.5) ──> 提升模型性能

非学习智能体 (1.10) ──> 基线对比

配置系统增强 (1.8) ──> 提升易用性
```

---

## 4. 与现有代码的集成

### 4.1 已有模块
- ✅ `satnav/core/env.py` - 环境类（单进程）
- ✅ `satnav/task/vln_task.py` - 任务定义
- ✅ `satnav/task/measures.py` - 评价指标
- ✅ `satnav/dataset/satnav_dataset.py` - 数据集加载
- ✅ `satnav/sims/satsim_wrapper.py` - 仿真器包装
- ✅ `satnav/navigation/path_follower.py` - 路径跟随器

### 4.2 需要扩展的模块
- 🔄 `satnav/core/env.py` - 需要支持向量化
- 🔄 `satnav/core/config.py` - 需要支持训练配置
- 🔄 `satnav/task/measures.py` - 可能需要添加更多指标

### 4.3 新增模块
- 🆕 `satnav/trainers/` - 训练框架（全新）
- 🆕 `satnav/models/` - 模型实现（全新）
- 🆕 `satnav/core/vector_env.py` - 向量化环境（全新）
- 🆕 `satnav/eval/` - 评估框架（全新）
- 🆕 `satnav/inference/` - 推理框架（全新）
- 🆕 `satnav/training/` - 训练工具（全新）
- 🆕 `satnav/augmentation/` - 数据增强（全新）
- 🆕 `satnav/agents/` - 非学习智能体（全新）

---

## 5. 参考资源

### 5.1 代码参考
- **VLN-CE**: https://github.com/jacobkrantz/VLN-CE
  - 训练框架: `vlnce_baselines/common/base_il_trainer.py`
  - 模型实现: `vlnce_baselines/models/`
  - 环境工具: `vlnce_baselines/common/env_utils.py`

- **Habitat-Lab**: https://github.com/facebookresearch/habitat-lab
  - 向量化环境: `habitat/core/vector_env.py`
  - 配置系统: `habitat/config/default.py`
  - 基线实现: `habitat_baselines/`

### 5.2 论文参考
- **VLN-CE**: "Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments" (ECCV 2020)

### 5.3 设计原则
- 保持 SatNav 的简洁性，避免过度设计
- 优先实现核心功能，逐步扩展
- 参考但不完全复制 VLN-CE/Habitat-Lab 的实现
- 适配 SatNav 的地理坐标系统特点

---

## 6. 总结

SatNav 目前是一个功能完整的**评测平台**，但缺少**训练能力**。要实现完整的"训练-评估-推理"流程，最关键的缺失模块是：

1. **训练框架** - 使平台具备训练能力
2. **基线模型** - 提供可训练的模型实现
3. **向量化环境** - 提升训练效率

建议按照上述优先级逐步实现，先完成第一阶段的核心功能，验证训练流程可行后，再逐步添加其他模块。

---

## 7. 架构规划：模型组织方案

### 7.1 模型分类和放置策略

SatNav 需要支持三类模型，每类有不同的组织方式：

#### 7.1.1 Baseline 模型（SatNav 自己的）
**模型**: `random`, `seq2seq`, `CMA`

**放置位置**: `satnav/models/baselines/`

**特点**:
- 完全集成在 SatNav 代码库中
- 使用 SatNav 的统一接口和配置系统
- 与 SatNav 核心代码紧密耦合

**理由**: 这些是 SatNav 平台的核心基线模型，应该作为平台的一部分。

---

#### 7.1.2 外部导航大模型（第三方）
**模型**: `navid`, `navila`, `streamvln` 等

**放置策略**: **推荐放在 repo 外**

**方案A：外部仓库 + 适配器（推荐）**
```
workspace/
├── SatNav/                      # SatNav主仓库
│   └── satnav/
│       └── models/
│           └── adapters/        # 适配器在SatNav内
│
├── navid/                       # NavID外部仓库（独立维护）
│   └── ...
│
├── navila/                      # NavILA外部仓库（独立维护）
│   └── ...
│
└── streamvln/                   # StreamVLN外部仓库（独立维护）
    └── ...
```

**优点**:
- ✅ 保持 SatNav 代码库简洁
- ✅ 外部模型可以独立更新和维护
- ✅ 避免依赖冲突和版本管理问题
- ✅ 外部模型可以有自己的依赖和配置

**适配器接口**:
```python
# satnav/models/adapters/base_adapter.py
from abc import ABC, abstractmethod
from satnav.core.episode import VLNEpisode

class BaseModelAdapter(ABC):
    """外部模型适配器基类，统一外部模型的接口"""
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, config: dict):
        """加载外部模型"""
        pass
    
    @abstractmethod
    def predict_action(self, observations: dict) -> str:
        """预测动作（适配SatNav接口）
        
        Args:
            observations: SatNav标准观测格式
                {
                    "rgb": np.ndarray,      # (H, W, 3)
                    "instruction": {
                        "text": str
                    }
                }
        
        Returns:
            action: str  # "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"
        """
        pass
    
    @abstractmethod
    def reset(self, episode: VLNEpisode):
        """重置模型状态（开始新episode）"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """获取模型信息（用于日志和调试）"""
        pass
```

**适配器示例**:
```python
# satnav/models/adapters/navid_adapter.py
import sys
from pathlib import Path

# 添加外部模型路径
NAVID_PATH = Path(__file__).parent.parent.parent.parent.parent / "navid"
sys.path.insert(0, str(NAVID_PATH))

from satnav.models.adapters.base_adapter import BaseModelAdapter
from navid.model import NavIDModel  # 外部模型的导入

class NavIDAdapter(BaseModelAdapter):
    """NavID模型适配器"""
    
    def __init__(self, checkpoint_path: str, config: dict):
        # 加载外部模型
        self.model = NavIDModel.from_checkpoint(checkpoint_path, config)
        # 转换配置格式
        self.navid_config = self._convert_config(config)
    
    def predict_action(self, observations: dict) -> str:
        # 转换观测格式
        navid_obs = self._convert_observations(observations)
        # 调用外部模型
        navid_action = self.model.predict(navid_obs)
        # 转换动作格式
        return self._convert_action(navid_action)
    
    def _convert_observations(self, satnav_obs: dict):
        """将SatNav观测转换为NavID格式"""
        # 实现转换逻辑
        pass
    
    def _convert_action(self, navid_action) -> str:
        """将NavID动作转换为SatNav格式"""
        # 实现转换逻辑
        pass
```

**方案B：Git Submodule（备选）**
如果希望外部模型在同一个仓库中管理：

```bash
# 添加外部模型为submodule
git submodule add <navid-repo-url> models/navid
git submodule add <navila-repo-url> models/navila
```

**结构**:
```
SatNav/
├── satnav/
└── models/                      # 外部模型目录
    ├── navid/                   # Git submodule
    ├── navila/                  # Git submodule
    └── streamvln/               # Git submodule
```

**缺点**:
- ⚠️ 增加仓库大小
- ⚠️ 需要管理submodule更新
- ⚠️ 可能引入依赖冲突

**推荐**: 优先使用方案A（外部仓库 + 适配器）

---

#### 7.1.3 用户自定义模型
**模型**: `mymodel`（用户自己的模型）

**放置位置**: `satnav/models/custom/mymodel/`

**特点**:
- 放在 SatNav 代码库内
- 遵循 SatNav 的模型接口规范
- 可以使用 SatNav 的共享组件（编码器等）

**结构**:
```
satnav/models/custom/
├── __init__.py
├── mymodel/
│   ├── __init__.py
│   ├── model.py                 # 模型实现
│   ├── config.py                # 模型配置类
│   ├── encoders.py              # 自定义编码器（可选）
│   └── README.md                # 模型说明文档
└── README.md                    # 自定义模型开发指南
```

**实现示例**:
```python
# satnav/models/custom/mymodel/model.py
from satnav.models.base import BasePolicy
from satnav.core.episode import VLNEpisode

class MyModel(BasePolicy):
    """用户自定义模型"""
    
    def __init__(self, config):
        super().__init__(config)
        # 实现模型架构
        self.encoder = ...
        self.decoder = ...
    
    def forward(self, observations, hidden_state=None):
        """前向传播"""
        # 实现推理逻辑
        pass
    
    def predict_action(self, observations: dict) -> str:
        """预测动作"""
        # 实现动作预测
        pass
```

---

### 7.2 模型注册机制

为了统一管理所有模型，实现一个模型注册系统：

```python
# satnav/models/registry.py
from typing import Dict, Type, Optional
from satnav.models.base import BasePolicy
from satnav.models.adapters.base_adapter import BaseModelAdapter

class ModelRegistry:
    """模型注册表"""
    
    _baseline_models: Dict[str, Type[BasePolicy]] = {}
    _custom_models: Dict[str, Type[BasePolicy]] = {}
    _external_adapters: Dict[str, Type[BaseModelAdapter]] = {}
    
    @classmethod
    def register_baseline(cls, name: str, model_class: Type[BasePolicy]):
        """注册Baseline模型"""
        cls._baseline_models[name] = model_class
    
    @classmethod
    def register_custom(cls, name: str, model_class: Type[BasePolicy]):
        """注册自定义模型"""
        cls._custom_models[name] = model_class
    
    @classmethod
    def register_external(cls, name: str, adapter_class: Type[BaseModelAdapter]):
        """注册外部模型适配器"""
        cls._external_adapters[name] = adapter_class
    
    @classmethod
    def get_model(cls, name: str, model_type: str = "auto"):
        """获取模型类
        
        Args:
            name: 模型名称
            model_type: "baseline", "custom", "external", "auto"
        """
        if model_type == "auto":
            # 自动检测模型类型
            if name in cls._baseline_models:
                return cls._baseline_models[name]
            elif name in cls._custom_models:
                return cls._custom_models[name]
            elif name in cls._external_adapters:
                return cls._external_adapters[name]
            else:
                raise ValueError(f"Model '{name}' not found")
        elif model_type == "baseline":
            return cls._baseline_models.get(name)
        elif model_type == "custom":
            return cls._custom_models.get(name)
        elif model_type == "external":
            return cls._external_adapters.get(name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

# 在 satnav/models/__init__.py 中注册所有模型
from satnav.models.registry import ModelRegistry

# 注册Baseline模型
from satnav.models.baselines.random_agent import RandomAgent
from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy
from satnav.models.baselines.cma_policy import CMAPolicy

ModelRegistry.register_baseline("random", RandomAgent)
ModelRegistry.register_baseline("seq2seq", Seq2SeqPolicy)
ModelRegistry.register_baseline("cma", CMAPolicy)

# 注册自定义模型
from satnav.models.custom.mymodel.model import MyModel
ModelRegistry.register_custom("mymodel", MyModel)

# 注册外部模型适配器
from satnav.models.adapters.navid_adapter import NavIDAdapter
from satnav.models.adapters.navila_adapter import NavILAAdapter
from satnav.models.adapters.streamvln_adapter import StreamVLNAdapter

ModelRegistry.register_external("navid", NavIDAdapter)
ModelRegistry.register_external("navila", NavILAAdapter)
ModelRegistry.register_external("streamvln", StreamVLNAdapter)
```

---

### 7.3 路径规划：SatNav vs satnav

#### 7.3.1 目录职责划分

**`SatNav/` (项目根目录)**:
- 项目级别的配置和脚本
- 数据目录（datasets, scene_datasets）
- 输出目录（checkpoints, outputs）
- 文档和示例
- 外部依赖管理

**`satnav/` (Python包目录)**:
- 核心代码实现
- 可导入的Python模块
- 内部API和接口

#### 7.3.2 推荐的目录结构

```
SatNav/                          # 项目根目录
├── satnav/                      # Python包（核心代码）
│   ├── core/                    # 核心组件
│   ├── models/                   # 模型实现
│   │   ├── baselines/           # Baseline模型
│   │   ├── custom/               # 自定义模型
│   │   └── adapters/            # 外部模型适配器
│   ├── trainers/                # 训练框架
│   ├── eval/                     # 评估框架
│   └── ...
│
├── configs/                     # 配置文件（项目级别）
│   ├── baselines/                # Baseline配置
│   ├── external/                 # 外部模型配置
│   └── custom/                   # 自定义模型配置
│
├── data/                        # 数据目录（项目级别）
│   ├── datasets/
│   └── scene_datasets/
│
├── checkpoints/                 # Checkpoints（项目级别）
│   ├── baselines/
│   ├── external/
│   └── custom/
│
├── outputs/                     # 输出目录（项目级别）
│   ├── logs/
│   ├── videos/
│   └── results/
│
├── scripts/                     # 脚本目录（项目级别）
│   ├── train.py
│   ├── eval.py
│   └── inference.py
│
└── models/                      # 🆕 外部模型目录（可选，如果使用submodule）
    ├── navid/
    ├── navila/
    └── streamvln/
```

---

### 7.4 配置文件组织

#### 7.4.1 配置文件结构

```
configs/
├── vln_task.yaml                # 基础任务配置
│
├── baselines/                   # Baseline模型配置
│   ├── random.yaml
│   ├── seq2seq.yaml
│   └── cma.yaml
│
├── external/                    # 外部模型配置
│   ├── navid.yaml
│   ├── navila.yaml
│   └── streamvln.yaml
│
└── custom/                      # 自定义模型配置
    └── mymodel.yaml
```

#### 7.4.2 配置文件示例

**Baseline配置** (`configs/baselines/seq2seq.yaml`):
```yaml
MODEL:
  TYPE: seq2seq
  INSTRUCTION_ENCODER:
    hidden_size: 128
    embedding_size: 50
  VISUAL_ENCODER:
    backbone: resnet50
    output_size: 256
  SEQ2SEQ:
    hidden_size: 512
    rnn_type: GRU
    use_prev_action: true
```

**外部模型配置** (`configs/external/navid.yaml`):
```yaml
MODEL:
  TYPE: navid
  ADAPTER: navid_adapter
  
  # 外部模型路径（相对于workspace根目录）
  EXTERNAL_MODEL_PATH: ../navid/checkpoints/navid.pth
  EXTERNAL_CONFIG_PATH: ../navid/configs/navid_config.yaml
  
  # 外部模型特定配置
  EXTERNAL_MODEL_ARGS:
    device: cuda
    batch_size: 1
```

**自定义模型配置** (`configs/custom/mymodel.yaml`):
```yaml
MODEL:
  TYPE: mymodel
  CUSTOM_MODEL_PATH: satnav.models.custom.mymodel
  
  # 自定义模型参数
  MYMODEL_ARGS:
    hidden_size: 512
    num_layers: 3
```

---

### 7.5 使用示例

#### 7.5.1 使用Baseline模型

```python
from satnav.models.registry import ModelRegistry
from satnav.core.config import load_config

# 加载配置
config = load_config("configs/baselines/seq2seq.yaml")

# 获取模型
model_class = ModelRegistry.get_model("seq2seq")
model = model_class.from_config(config)
```

#### 7.5.2 使用外部模型

```python
from satnav.models.registry import ModelRegistry
from satnav.core.config import load_config

# 加载配置
config = load_config("configs/external/navid.yaml")

# 获取适配器
adapter_class = ModelRegistry.get_model("navid", model_type="external")
adapter = adapter_class(
    checkpoint_path=config.MODEL.EXTERNAL_MODEL_PATH,
    config=config.MODEL.EXTERNAL_MODEL_ARGS
)

# 使用适配器（接口与Baseline模型一致）
action = adapter.predict_action(observations)
```

#### 7.5.3 使用自定义模型

```python
from satnav.models.registry import ModelRegistry
from satnav.core.config import load_config

# 加载配置
config = load_config("configs/custom/mymodel.yaml")

# 获取模型
model_class = ModelRegistry.get_model("mymodel", model_type="custom")
model = model_class.from_config(config)
```

---

### 7.6 总结和建议

#### 推荐方案总结

1. **Baseline模型** (`random`, `seq2seq`, `CMA`)
   - ✅ 放在 `satnav/models/baselines/`
   - ✅ 完全集成在SatNav中

2. **外部模型** (`navid`, `navila`, `streamvln`)
   - ✅ **推荐**: 放在repo外（独立仓库）
   - ✅ 通过适配器接入（`satnav/models/adapters/`）
   - ✅ 保持SatNav代码库简洁

3. **自定义模型** (`mymodel`)
   - ✅ 放在 `satnav/models/custom/mymodel/`
   - ✅ 遵循SatNav接口规范

4. **路径规划**
   - ✅ `SatNav/`: 项目级别（配置、数据、输出）
   - ✅ `satnav/`: Python包（核心代码）

5. **模型注册**
   - ✅ 统一的注册机制
   - ✅ 支持自动发现和加载

---

## 附录：文件结构建议

### A.1 项目根目录结构（SatNav/）

```
SatNav/                          # 项目根目录
├── satnav/                      # Python包目录（核心代码）
│   ├── __init__.py
│   ├── core/                    # 核心组件
│   ├── task/                    # 任务定义
│   ├── dataset/                 # 数据集加载
│   ├── sims/                    # 仿真器
│   ├── navigation/              # 导航算法
│   ├── models/                  # 模型实现（见A.2）
│   ├── trainers/                # 训练框架
│   ├── eval/                    # 评估框架
│   ├── inference/               # 推理框架
│   ├── training/                # 训练工具
│   ├── augmentation/            # 数据增强
│   ├── agents/                  # 非学习智能体
│   └── utils/                   # 工具函数
│
├── models/                      # 🆕 外部模型目录（可选）
│   ├── navid/                   # NavID模型（外部仓库或子模块）
│   ├── navila/                  # NavILA模型（外部仓库或子模块）
│   ├── streamvln/               # StreamVLN模型（外部仓库或子模块）
│   └── README.md                # 外部模型使用说明
│
├── configs/                     # 配置文件目录
│   ├── vln_task.yaml            # 任务配置
│   ├── baselines/               # Baseline模型配置
│   │   ├── random.yaml
│   │   ├── seq2seq.yaml
│   │   └── cma.yaml
│   ├── external/                # 🆕 外部模型配置
│   │   ├── navid.yaml
│   │   ├── navila.yaml
│   │   └── streamvln.yaml
│   └── custom/                  # 🆕 自定义模型配置
│       └── mymodel.yaml
│
├── data/                        # 数据目录
│   ├── datasets/                # 数据集文件
│   └── scene_datasets/         # 场景数据
│
├── checkpoints/                 # 🆕 模型检查点目录
│   ├── baselines/               # Baseline模型checkpoints
│   ├── external/                # 外部模型checkpoints
│   └── custom/                  # 自定义模型checkpoints
│
├── outputs/                     # 🆕 输出目录（重命名自output/）
│   ├── logs/                    # 训练日志
│   ├── videos/                  # 视频输出
│   ├── visualizations/          # 可视化结果
│   └── results/                 # 评估结果
│
├── scripts/                     # 🆕 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── eval.py                  # 评估脚本
│   ├── inference.py             # 推理脚本
│   └── setup_external_models.py # 外部模型设置脚本
│
├── examples/                    # 示例代码
├── tests/                       # 测试代码
├── applications/                # 应用工具
├── doc/                         # 文档目录
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装脚本
└── README.md                    # 项目说明
```

### A.2 模型目录详细结构（satnav/models/）

```
satnav/models/
├── __init__.py
├── base.py                      # 🆕 模型基类和接口定义
│
├── baselines/                   # 🆕 SatNav自己的Baseline模型
│   ├── __init__.py
│   ├── random_agent.py         # 随机智能体
│   ├── seq2seq_policy.py        # Seq2Seq模型
│   ├── cma_policy.py            # CMA模型
│   └── README.md                # Baseline模型说明
│
├── custom/                      # 🆕 用户自定义模型
│   ├── __init__.py
│   ├── mymodel/                 # 用户自己的模型
│   │   ├── __init__.py
│   │   ├── model.py             # 模型实现
│   │   ├── config.py             # 模型配置
│   │   └── README.md            # 模型说明
│   └── README.md                # 自定义模型开发指南
│
├── adapters/                    # 🆕 外部模型适配器
│   ├── __init__.py
│   ├── base_adapter.py          # 适配器基类
│   ├── navid_adapter.py         # NavID适配器
│   ├── navila_adapter.py        # NavILA适配器
│   ├── streamvln_adapter.py     # StreamVLN适配器
│   └── README.md                # 适配器使用说明
│
├── encoders/                    # 编码器模块（共享）
│   ├── __init__.py
│   ├── instruction_encoder.py
│   ├── visual_encoder.py
│   └── state_encoder.py
│
└── utils.py                     # 模型工具函数
```

### A.3 外部模型集成方案

#### 方案A：外部模型放在repo外（推荐）

**优点**:
- 保持SatNav代码库的简洁性
- 外部模型可以独立维护和更新
- 避免依赖冲突

**结构**:
```
workspace/
├── SatNav/                      # SatNav主仓库
│   └── satnav/
│       └── models/
│           └── adapters/        # 适配器在SatNav内
│
├── navid/                       # NavID外部仓库
│   └── ...
│
├── navila/                      # NavILA外部仓库
│   └── ...
│
└── streamvln/                   # StreamVLN外部仓库
    └── ...
```

**适配器接口**:
```python
# satnav/models/adapters/base_adapter.py
class BaseModelAdapter(ABC):
    """外部模型适配器基类"""
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, config: dict):
        """加载外部模型"""
        pass
    
    @abstractmethod
    def predict_action(self, observations: dict) -> str:
        """预测动作（适配SatNav接口）"""
        pass
    
    @abstractmethod
    def reset(self, episode: VLNEpisode):
        """重置模型状态"""
        pass
```

#### 方案B：外部模型作为子模块（备选）

**结构**:
```
SatNav/
├── satnav/
└── models/                      # 外部模型目录
    ├── navid/                   # Git submodule或独立目录
    ├── navila/
    └── streamvln/
```

**使用Git Submodule**:
```bash
git submodule add <navid-repo-url> models/navid
git submodule add <navila-repo-url> models/navila
```

### A.4 模型注册机制

```python
# satnav/models/__init__.py
from satnav.models.registry import ModelRegistry

# 注册Baseline模型
ModelRegistry.register("random", "satnav.models.baselines.random_agent", "RandomAgent")
ModelRegistry.register("seq2seq", "satnav.models.baselines.seq2seq_policy", "Seq2SeqPolicy")
ModelRegistry.register("cma", "satnav.models.baselines.cma_policy", "CMAPolicy")

# 注册自定义模型
ModelRegistry.register("mymodel", "satnav.models.custom.mymodel.model", "MyModel")

# 注册外部模型（通过适配器）
ModelRegistry.register("navid", "satnav.models.adapters.navid_adapter", "NavIDAdapter")
ModelRegistry.register("navila", "satnav.models.adapters.navila_adapter", "NavILAAdapter")
```

### A.5 配置文件组织

```yaml
# configs/baselines/seq2seq.yaml
MODEL:
  TYPE: seq2seq
  INSTRUCTION_ENCODER:
    hidden_size: 128
  ...

# configs/external/navid.yaml
MODEL:
  TYPE: navid
  ADAPTER: navid_adapter
  EXTERNAL_MODEL_PATH: ../navid/checkpoints/navid.pth
  EXTERNAL_CONFIG_PATH: ../navid/configs/navid_config.yaml
  ...

# configs/custom/mymodel.yaml
MODEL:
  TYPE: mymodel
  CUSTOM_MODEL_PATH: satnav.models.custom.mymodel
  ...
```

---

## 附录：实现检查清单

### 第一阶段检查清单
- [ ] BaseTrainer 基类实现
- [ ] DaggerTrainer 实现
- [ ] Seq2SeqPolicy 模型实现
- [ ] 指令编码器实现
- [ ] 视觉编码器实现
- [ ] 状态编码器实现
- [ ] Evaluator 类实现
- [ ] 评估脚本实现
- [ ] 基础训练配置
- [ ] 训练脚本（run.py）

### 第二阶段检查清单
- [ ] VectorEnv 实现
- [ ] 环境构建工具
- [ ] RolloutStorage 实现
- [ ] 轨迹管理器实现
- [ ] TensorBoard 集成
- [ ] 视频生成工具

### 第三阶段检查清单
- [ ] 路径增强实现
- [ ] 指令增强实现
- [ ] 视觉增强实现（谨慎）
- [ ] 推理器实现
- [ ] 推理脚本实现
- [ ] CMAPolicy 实现

### 第四阶段检查清单
- [ ] 随机智能体实现
- [ ] 最短路径跟随器实现
- [ ] 贪心智能体实现
- [ ] 实验配置管理
- [ ] 模型配置系统
- [ ] 训练教程文档
- [ ] API 文档

---

**文档维护**: 本文档应随着项目进展定期更新，标记已完成模块并调整优先级。

**最后更新**: 2024年12月

