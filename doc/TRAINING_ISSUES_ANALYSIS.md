# Seq2Seq训练问题分析

## 核心问题总结

基于调试和分析，seq2seq训练不好的主要原因有以下几个，按重要性排序：

## 🔴 问题1：数据完全重复（最严重

### 现象
- **11个训练episodes完全相同**
- 相同的起点、终点、路径、instruction
- 模型只能看到一条轨迹

### 影响
1. **无法学习泛化**：模型只能记忆这一条路径
2. **无法学习视觉-语言-动作对应关系**：因为没有变化，模型学不到"看到什么图像/听到什么指令时应该做什么动作"
3. **过度依赖prev_action**：因为路径固定，模型学会了"如果prev_action=MOVE_FORWARD，就继续MOVE_FORWARD"

### 证据
- 调试显示：模型几乎不使用instruction（zero out instruction时预测差异只有0.0084）
- 模型主要依赖prev_action和RNN state
- 即使RGB在变化，模型仍然预测MOVE_FORWARD

### 解决方案
**必须生成更多样化的数据**：
- 不同的起点/终点
- 不同的路径（需要转弯的路径）
- 不同的指令文本
- 至少需要10-20条不同的轨迹

---

## 🟡 问题2：数据不平衡

### 现象
- MOVE_FORWARD: 89.6% (660/737)
- TURN_RIGHT: 9.0% (66/737)
- TURN_LEFT: 1.5% (11/737)
- STOP: 0%

### 影响
- 即使有class weights，模型仍然倾向于预测MOVE_FORWARD
- TURN_LEFT样本太少，模型难以学习

### 解决方案
- ✅ 已实现class weights（MOVE_FORWARD: 0.1, TURN_LEFT: 10.0）
- ✅ 已实现inflection weighting
- ⚠️ 但数据不平衡仍然严重，需要更多TURN样本

---

## 🟡 问题3：模型架构问题 - 不使用Instruction

### 现象
- 调试显示：zero out instruction时，预测几乎不变（差异0.0084）
- Instruction embedding在所有步骤中完全相同（因为instruction不变）
- 模型主要依赖prev_action和RNN state

### 原因分析
1. **数据单一**：因为所有episodes的instruction相同，模型学不到instruction的作用
2. **特征融合问题**：Instruction embedding (128维) vs RGB embedding (256维)，可能被RGB主导
3. **RNN记忆问题**：RNN state可能记住了"总是MOVE_FORWARD"，忽略了当前观察

### 解决方案
1. **增加数据多样性**（最重要）
2. **检查特征归一化**：确保instruction和RGB特征尺度匹配
3. **考虑使用attention机制**：让模型关注instruction中的关键信息

---

## 🟢 问题4：训练方式问题 - Teacher Forcing

### 现象
- Teacher forcing：模型看到的是ground truth prev_action
- 模型学会了"如果prev_action=MOVE_FORWARD，就继续MOVE_FORWARD"
- 没有学习"根据当前观察（RGB+instruction）决定动作"

### 影响
- 训练和评估的分布不匹配
- 模型过度依赖prev_action，而不是当前观察

### 解决方案
- ✅ 已实现Scheduled Sampling（但当前配置中use_scheduled_sampling=false）
- 可以考虑启用Scheduled Sampling
- 或者使用DAgger（但需要更多数据）

---

## 🟢 问题5：use_prev_action=True的影响

### 现象
- prev_action embedding (32维) 被添加到输入
- 模型可能过度依赖prev_action

### 影响
- 如果数据单一，模型更容易学会"根据prev_action预测下一个action"
- 而不是"根据当前观察预测action"

### 解决方案
- 可以尝试设置`use_prev_action: false`
- 强制模型只使用RGB和instruction
- 但这需要重新训练

---

## 根本原因排序

### 🔴 最严重：数据完全重复
**这是最根本的问题**。即使有最好的模型架构和训练方法，如果数据只有一条轨迹，模型也无法学习到有用的知识。

### 🟡 次要：数据不平衡
虽然严重，但可以通过权重机制缓解。

### 🟡 次要：模型不使用Instruction
这很可能是数据单一导致的。如果数据多样化，模型应该能学会使用instruction。

### 🟢 轻微：训练方式问题
Teacher forcing和use_prev_action的影响相对较小，主要是数据问题的表现。

---

## 建议的解决方案（按优先级）

### 1. 🔴 立即解决：生成多样化数据（必须）
- 创建至少10-20条不同的轨迹
- 包含不同的起点/终点
- 包含需要转弯的路径
- 包含不同的指令文本

### 2. 🟡 短期：启用Scheduled Sampling
- 设置`use_scheduled_sampling: true`
- - 让模型看到自己的预测，减少对teacher forcing的依赖

### 3. 🟡 短期：尝试use_prev_action=false
- 强制模型只使用RGB和instruction
- 避免过度依赖prev_action

### 4. 🟢 长期：考虑DAgger
- 如果数据多样化后仍然有问题，可以考虑实现DAgger
- 让模型看到错误轨迹上的observations

---

## 结论

**主要原因是数据完全重复**。这是最根本的问题，必须首先解决。

其他问题（数据不平衡、模型不使用instruction、训练方式）都是数据问题的表现或次要问题。

**建议**：
1. 先解决数据问题（生成多样化数据）
2. 然后重新训练
3. 如果仍然有问题，再考虑调整模型架构和训练方式

