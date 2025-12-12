# Seq2Seq Model Comparison: SatNav vs VLN-CE

## 概述

本文档对比了SatNav和VLN-CE中Seq2Seq模型的结构设计，分析差异和潜在问题。

## 模型架构对比

### 1. 整体结构

| 组件 | VLN-CE | SatNav | 差异说明 |
|------|--------|--------|----------|
| Instruction Encoder | ✓ LSTM (128) | ✓ LSTM (128) | **一致** |
| Depth Encoder | ✓ ResNet50 (128) | ✗ 无 | **合理差异** - SatNav使用卫星图像，无depth信息 |
| RGB Encoder | ✓ ResNet50 (256) | ✓ ResNet50 (256) | **一致** |
| Prev Action Embedding | ✓ 32维 (可选) | ✓ 32维 (可选) | **一致** |
| RNN State Encoder | ✓ GRU (512) | ✓ GRU (512) | **一致** |
| Progress Monitor | ✓ 可选 (默认False) | ✗ 无 | **缺失** - 见下文分析 |

### 2. 详细参数对比

#### Instruction Encoder
| 参数 | VLN-CE | SatNav | 状态 |
|------|--------|--------|------|
| embedding_size | 50 | 50 | ✓ 一致 |
| hidden_size | 128 | 128 | ✓ 一致 |
| rnn_type | LSTM | LSTM | ✓ 一致 |
| bidirectional | False | False | ✓ 一致 |
| final_state_only | True | True | ✓ 一致 |
| use_pretrained_embeddings | True | True | ✓ 一致 |

#### RGB Encoder
| 参数 | VLN-CE | SatNav | 状态 |
|------|--------|--------|------|
| cnn_type | ResNet50 | ResNet50 | ✓ 一致 |
| output_size | 256 | 256 | ✓ 一致 |
| trainable | False | False | ✓ 一致 |
| normalize_visual_inputs | False | False | ✓ 一致 |

#### RNN State Encoder
| 参数 | VLN-CE | SatNav | 状态 |
|------|--------|--------|------|
| hidden_size | 512 | 512 | ✓ 一致 |
| rnn_type | GRU | GRU | ✓ 一致 |
| num_layers | 1 | 1 | ✓ 一致 |

#### Seq2Seq配置
| 参数 | VLN-CE | SatNav | 状态 |
|------|--------|--------|------|
| use_prev_action | False (默认) | True (默认) | ⚠️ **差异** |

## 关键差异分析

### 1. Depth Encoder缺失 ✓ (合理)

**VLN-CE**: 使用ResNet50编码depth图像 (output_size=128)  
**SatNav**: 无depth encoder

**分析**: 
- ✅ **合理**: SatNav使用卫星俯视图，没有depth信息
- ✅ **不影响**: 模型结构仍然完整，只是输入模态不同

### 2. Progress Monitor缺失 ⚠️ (潜在问题)

**VLN-CE**: 
```python
self.progress_monitor = nn.Linear(
    hidden_size, 1
)
# 作为auxiliary loss使用
progress_hat = torch.tanh(self.progress_monitor(x))
progress_loss = F.mse_loss(
    progress_hat.squeeze(1),
    observations["progress"],
    reduction="none",
)
AuxLosses.register_loss("progress_monitor", progress_loss, alpha)
```

**SatNav**: 无progress monitor

**分析**:
- ⚠️ **影响**: Progress Monitor是一个auxiliary loss，帮助模型学习导航进度
- ⚠️ **重要性**: 在VLN-CE中是可选的（默认False），但在一些配置中启用（如`seq2seq_pm.yaml`）
- ✅ **可接受**: 如果不需要progress supervision，可以不加
- 💡 **建议**: 如果训练效果不理想，可以考虑添加

**Progress Monitor的作用**:
- 预测导航进度（0.0到1.0）
- 作为auxiliary loss帮助训练
- 帮助模型理解当前位置相对于目标的进度

### 3. use_prev_action默认值不同 ⚠️ (配置差异)

**VLN-CE**: `use_prev_action: False` (默认)  
**SatNav**: `use_prev_action: True` (默认)

**分析**:
- ⚠️ **差异**: 默认配置不同
- ✅ **影响**: 使用prev_action通常能提升性能
- 💡 **建议**: 保持True是合理的，但应该明确文档说明

## 训练方式对比

| 特性 | VLN-CE | SatNav | 状态 |
|------|--------|--------|------|
| Teacher Forcing | ✓ | ✓ | 一致 |
| Scheduled Sampling | ✗ | ✓ | **SatNav更先进** |
| DAgger | ✓ | ✗ | VLN-CE支持 |
| Progress Monitor Loss | ✓ (可选) | ✗ | SatNav缺失 |

## 潜在问题总结

### ✅ 无问题的差异
1. **Depth Encoder缺失**: 合理，因为SatNav使用卫星图像
2. **use_prev_action默认值**: 配置差异，不影响功能

### ⚠️ 需要注意的差异
1. **Progress Monitor缺失**: 
   - 如果训练效果不理想，可以考虑添加
   - 需要确保observations中有"progress"字段
   - 需要实现AuxLosses机制

## 建议

### 1. 短期（可选）
- 保持当前结构，因为Progress Monitor在VLN-CE中也是可选的
- 如果训练效果良好，无需修改

### 2. 中期（如果效果不理想）
- 考虑添加Progress Monitor作为auxiliary loss
- 需要：
  - 在observations中添加progress字段
  - 实现AuxLosses机制
  - 添加progress_monitor模块

### 3. 长期（优化）
- 考虑实现DAgger训练器（如果需要）
- 保持Scheduled Sampling（已经实现，比VLN-CE更先进）

## 代码质量对比

| 方面 | VLN-CE | SatNav | 评价 |
|------|--------|--------|------|
| 代码结构 | ✓ | ✓ | 都很好 |
| 文档注释 | ✓ | ✓ | SatNav注释更详细 |
| 可配置性 | ✓ | ✓ | 都支持配置 |
| 扩展性 | ✓ | ✓ | 都易于扩展 |

## 结论

**整体评价**: SatNav的Seq2Seq模型结构设计**基本正确**，与VLN-CE的核心架构一致。

**主要差异**:
1. ✅ Depth Encoder缺失 - **合理**（卫星图像无depth）
2. ⚠️ Progress Monitor缺失 - **可选**（VLN-CE中也是可选的）
3. ⚠️ use_prev_action默认值 - **配置差异**（不影响功能）

**建议**: 
- 如果当前训练效果良好，无需修改
- 如果效果不理想，可以考虑添加Progress Monitor
- Scheduled Sampling的实现比VLN-CE更先进，这是优势

**模型结构本身没有问题**，主要差异都是合理的或可选的。

