# Feature Importance Experiments

## 目标

验证假设："模型发现根据 prev_action 预测比理解 RGB+instruction 更容易"

## 实验方法

### 1. Ablation Study（特征移除实验）

**方法**：逐个移除特征，观察预测概率的变化

**结果**：
- **RGB移除影响**: 0.1986 (HIGH importance)
- **Instruction移除影响**: 0.0092 (LOW importance)
- **RNN State重置影响**: 0.0000 (LOW importance)

**结论**：
- ✓ 模型主要依赖 RGB 视觉特征
- ✗ 模型几乎不使用 instruction
- ✗ 模型不使用 RNN 历史状态（第一步时）

### 2. Feature Contribution Analysis（特征贡献分析）

**方法**：在多步中测试特征移除的影响

**结果**：
- **Instruction平均影响**: 0.0103 (Low impact)
- **RGB平均影响**: 0.1773 (High impact)
- **RNN History平均影响**: 0.0000 (Low impact)
- **Action变化率**: 0% (移除任何特征都不改变预测的动作)

**结论**：
- 模型主要使用 RGB，但即使移除 RGB，预测的动作也不变
- 这说明模型可能学到了一个简单的先验分布

### 3. Prev_Action Pattern Test（Prev_Action模式测试）

**方法**：
- Test 1: 相同 RGB+Instruction，不同动作历史
- Test 2: 不同 RGB+Instruction，相同 Prev_Action

**结果**：

#### Test 1: 相同观察，不同历史
- Scenario A (历史=[MOVE_FORWARD, MOVE_FORWARD, MOVE_FORWARD]): 预测 MOVE_FORWARD
- Scenario B (历史=[TURN_LEFT, TURN_LEFT, TURN_LEFT]): 预测 MOVE_FORWARD
- **差异**: 0.0000
- **结论**: 模型忽略动作历史！

#### Test 2: 不同观察，相同 Prev_Action
- Observation 1: 预测 MOVE_FORWARD
- Observation 2: 预测 MOVE_FORWARD
- **差异**: 0.0000
- **结论**: 模型忽略 RGB+Instruction！

## 关键发现

### 1. 模型行为异常

即使正确更新了 RNN state，模型仍然：
- 忽略动作历史（不同历史 → 相同预测）
- 忽略观察变化（不同观察 → 相同预测）

### 2. 可能的原因

#### 原因1: 模型学到了简单的先验分布
- 训练数据中 `MOVE_FORWARD` 占 89.6%
- 模型可能直接学到了 "总是预测 MOVE_FORWARD"
- 这解释了为什么改变任何输入都不影响预测

#### 原因2: 数据缺乏多样性
- 训练数据只有 11 个相同的 episode
- 模型无法学习到观察 → 动作的映射
- 只能学到数据中的统计模式（MOVE_FORWARD 占主导）

#### 原因3: RNN State 没有真正编码历史
- 虽然 RNN state 在更新，但可能没有学到有用的历史信息
- 因为所有步骤的观察都相同（同一轨迹），RNN 无法区分不同时间步

## 验证"模型使用 prev_action 作为捷径"的结论

### 实验证据

虽然直接测试显示模型忽略了历史，但这可能是因为：

1. **训练时的行为**：
   - 训练时，模型看到的是 ground truth `prev_action`
   - 模型学到：`prev_action=1 (MOVE_FORWARD) → action=1 (MOVE_FORWARD)`
   - 这是一个简单的模式匹配

2. **评估时的行为**：
   - 评估时，模型预测 `action=1`
   - 下一个时间步，`prev_action=1`
   - 模型继续预测 `action=1`
   - 结果：一直往前走

3. **为什么测试显示"忽略历史"**：
   - 测试中，我们手动设置了不同的 `prev_action`
   - 但模型可能没有真正学到"根据 `prev_action` 决定动作"
   - 而是学到了"根据训练数据中的模式预测"
   - 因为训练数据中 `prev_action=1` 后几乎总是 `action=1`

## 结论

### 原始假设验证

**假设**: "模型发现根据 prev_action 预测比理解 RGB+instruction 更容易"

**验证结果**: **部分正确**

1. ✓ **模型确实不使用 RGB+Instruction 进行导航**
   - RGB 影响较大（0.1986），但移除后预测不变
   - Instruction 影响很小（0.0092）

2. ✓ **模型主要依赖统计模式**
   - 训练数据中 `MOVE_FORWARD` 占主导
   - 模型学到了"总是预测 MOVE_FORWARD"

3. ✗ **但模型也没有真正使用 prev_action**
   - 测试显示改变 `prev_action` 不影响预测
   - 模型可能只是学到了数据中的统计分布

### 真正的结论

**模型没有学到导航能力，而是学到了数据中的统计模式。**

具体来说：
- 训练数据中 `MOVE_FORWARD` 占 89.6%
- 模型学到：预测 `MOVE_FORWARD` 可以获得低 loss
- 模型不需要理解 RGB、Instruction 或 `prev_action`
- 只需要预测最常见的动作即可

### 解决方案

1. **生成更多样化的数据**（必须）
   - 不同轨迹、不同指令
   - 平衡的动作分布

2. **强制模型使用观察**
   - 尝试 `use_prev_action=False`
   - 增加 RGB 和 Instruction 的权重

3. **使用 Scheduled Sampling**
   - 让模型看到自己的错误
   - 减少 exposure bias

## 实验脚本

所有实验脚本位于 `experiments/` 目录：

1. `feature_importance_analysis.py`: Ablation study 和特征贡献分析
2. `test_prev_action_vs_features.py`: Prev_Action 模式测试
3. `visualize_feature_usage.py`: 特征使用可视化（待完善）

## 运行实验

```bash
# Ablation study
python experiments/feature_importance_analysis.py

# Prev_Action pattern test
python experiments/test_prev_action_vs_features.py
```

