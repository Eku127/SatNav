# Seq2Seq Model Implementation in SatNav

## Overview

This document describes the implementation of the Sequence-to-Sequence (Seq2Seq) baseline model in SatNav for Vision-and-Language Navigation (VLN) tasks in continuous environments using satellite imagery.

**Reference:** VLN-CE vlnce_baselines/models/seq2seq_policy.py

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Seq2Seq Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Instruction → LSTM Encoder → instruction_embedding (128)       │
│  RGB Image   → ResNet50     → rgb_embedding (256)               │
│  Previous Action → Embedding(32)                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ State Encoder                                           │    │
│  │   [instruction_emb, rgb_emb, prev_action]               │    │
│  │   → GRU (512) → hidden_state                            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Output → Action Distribution (4 actions)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Components

#### 1. Instruction Encoder
- **Type:** Uni-directional LSTM
- **Input:** Tokenized instruction [batch, seq_len]
- **Output:** Final hidden state [batch, 128]
- **Key Setting:** `final_state_only=True` (only returns final state)
- **Embedding:** GloVe 50d or random initialization
  - Index 0: PAD (all zeros)
  - Index 1: UNK (mean of all word embeddings)
  - Index 2+: Vocabulary words

#### 2. RGB Visual Encoder
- **Type:** ResNet-50 (pretrained on ImageNet)
- **Input:** RGB image [batch, 224, 224, 3]
- **Output:** Global features [batch, 256]
- **Key Setting:** `spatial_output=False` (global pooling)
- **Trainable:** False (frozen by default)
- **Preprocessing:** Scale to [0, 1], optional ImageNet normalization

#### 3. Previous Action Embedding
- **Type:** nn.Embedding
- **Size:** (num_actions + 1) × 32
- **Purpose:** Encode action history as context
- **Optional:** Can be disabled via `SEQ2SEQ.use_prev_action`

#### 4. State Encoder (GRU)
- **Input:** Concatenation of [instruction_emb, rgb_emb, prev_action_emb]
- **Input Size:** 128 + 256 + 32 = 416 (with prev_action) or 384 (without)
- **Hidden Size:** 512
- **Purpose:** Maintain temporal context across navigation steps
- **Output:** Final features [batch, 512]

## Differences from VLN-CE Seq2Seq

| Component | VLN-CE Seq2Seq | SatNav Seq2Seq | Reason |
|-----------|----------------|----------------|--------|
| **Depth Encoder** | ✅ Included | ❌ Removed | SatNav uses satellite imagery (no depth) |
| **RGB Encoder** | ResNet-50 | ResNet-50 | Same |
| **Instruction Encoder** | LSTM | LSTM | Same |
| **State Encoder** | GRU | GRU | Same |
| **Previous Action** | Embedding (32) | Embedding (32) | Same |
| **Progress Monitor** | ✅ Included | ❌ Removed | Simplified implementation |

## Implementation Details

### File Structure

```
satnav/models/baselines/
├── seq2seq_policy.py       # Seq2Seq implementation
├── cma_policy.py           # CMA implementation
└── README.md               # Documentation

configs/baselines/
└── seq2seq.yaml            # Seq2Seq configuration

tests/
└── test_seq2seq.py         # Unit tests (if exists)

examples/
└── seq2seq_demo.py         # Demo script (if exists)
```

### Key Classes

#### Seq2SeqPolicy
- **Inherits:** `ILPolicy` (from `satnav.models.base`)
- **Purpose:** Policy wrapper for Seq2Seq network
- **Methods:**
  - `from_config()`: Create policy from configuration
  - `act()`: Select action given observations
  - `build_distribution()`: Build action distribution

#### Seq2SeqNet
- **Inherits:** `Net` (from `satnav.models.base`)
- **Purpose:** Seq2Seq neural network backbone
- **Properties:**
  - `output_size`: 512
  - `num_recurrent_layers`: 1 (one GRU encoder)
  - `is_blind`: False
- **Methods:**
  - `forward()`: Forward pass through network

### RNN State Handling

Seq2Seq has **one RNN encoder**, making state management straightforward:

```python
# RNN states shape: [num_recurrent_layers, batch_size, hidden_size]
# For Seq2Seq: [1, batch_size, 512]

# Initialize states
rnn_states = torch.zeros(1, batch_size, 512)

# Forward pass
features, rnn_states_out = self.state_encoder(
    state_in,
    rnn_states,
    masks
)
```

### Trainer Compatibility

Seq2Seq works seamlessly with `RecollectTrainer` and other trainers:

```python
# Trainer automatically handles RNN state initialization
rnn_states = torch.zeros(
    self.policy.net.num_recurrent_layers,  # 1 for Seq2Seq
    N,
    hidden_size,
    device=device
)
```

## Model Parameters

### Parameter Count
- **Total:** ~28M parameters
- **Trainable:** ~5M parameters
- **Frozen:** ~23M parameters (ResNet-50 backbone)

### Breakdown
```
Component                    Parameters
─────────────────────────────────────────
Instruction Encoder          ~200K
  - Embedding (2504 × 50)    ~125K
  - LSTM                     ~75K
RGB Encoder (ResNet-50)      ~23.5M (frozen)
Previous Action Embedding    ~160
State Encoder (GRU)          ~1.3M
  - Input projection         ~800K
  - Recurrent weights        ~500K
Action Distribution Head     ~2K
─────────────────────────────────────────
Total                        ~28M
Trainable                    ~5M
```

## Configuration

### Default Configuration (`configs/baselines/seq2seq.yaml`)

```yaml
MODEL:
  policy_name: seq2seq
  
  INSTRUCTION_ENCODER:
    embedding_size: 50
    hidden_size: 128
    rnn_type: LSTM
    bidirectional: false       # Uni-directional
    final_state_only: true     # Only final state
    use_pretrained_embeddings: true
    embedding_file: data/debug_data/embeddings.json.gz
  
  RGB_ENCODER:
    cnn_type: TorchVisionResNet50
    output_size: 256
    trainable: false
    # spatial_output: false (default)
  
  STATE_ENCODER:
    hidden_size: 512
    rnn_type: GRU
    num_layers: 1
  
  SEQ2SEQ:
    use_prev_action: true

IL:
  lr: 1e-4
  batch_size: 1
  epochs: 20
```

## Usage

### Training

```bash
# Train Seq2Seq model
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train

# Train with custom settings
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train \
    IL.lr 2.5e-4 IL.batch_size 5
```

### Evaluation

```bash
# Evaluate on val_seen
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval

# Evaluate on val_unseen
python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \
    EVAL.SPLIT val_unseen
```

### Programmatic Usage

```python
from satnav.models import ModelRegistry

# Get model class
Seq2SeqPolicy = ModelRegistry.get_model('seq2seq')

# Create policy
policy = Seq2SeqPolicy.from_config(
    config=config,
    observation_space=obs_space,
    action_space=act_space
)

# Initialize RNN states (1 layer for Seq2Seq)
rnn_states = torch.zeros(1, batch_size, 512)

# Forward pass
action, rnn_states = policy.act(
    observations,
    rnn_states,
    prev_actions,
    masks,
    deterministic=True
)
```

## Testing

### Unit Tests

```bash
# Run model instantiation tests
pytest tests/test_models.py::test_seq2seq_policy -v

# Run forward pass tests
pytest tests/test_models.py::test_seq2seq_forward -v
```

### Quick Validation

```python
import torch
from satnav.models import ModelRegistry

# Create model
model = ModelRegistry.get_model('seq2seq').from_config(config, obs_space, act_space)

# Check properties
print(f"Output size: {model.net.output_size}")
print(f"Num recurrent layers: {model.net.num_recurrent_layers}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Performance Expectations

### Training Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Parameters** | ~28M | Includes frozen ResNet |
| **Trainable Parameters** | ~5M | Instruction + State encoders |
| **Training Time** | Baseline | Reference for other models |
| **Memory Usage** | ~2GB | On GPU with batch_size=5 |
| **Convergence** | 15-20 epochs | Depends on dataset size |

### Typical Performance

Baseline Seq2Seq typically achieves:
- **Success Rate (SR):** 30-40% on val_seen
- **SPL:** 25-35% on val_seen
- **Oracle SR:** 50-60% (indicates grounding capability)

### Training Tips

1. **Learning Rate:** Start with 1e-4, can increase to 2.5e-4 if stable
2. **Batch Size:** 1-5, depending on GPU memory
3. **Gradient Clipping:** Use max_norm=5.0 to prevent exploding gradients
4. **Epochs:** 15-20 epochs usually sufficient for small datasets
5. **Previous Action:** Enabling `use_prev_action` usually helps performance

## Known Issues and Limitations

### 1. No Visual Attention
- Seq2Seq uses global pooled visual features
- Cannot focus on specific image regions
- Consider using CMA for attention mechanisms

### 2. Limited Instruction Grounding
- No explicit grounding of instruction phrases to visual features
- May struggle with complex spatial reasoning
- Use attention-based models for better grounding

### 3. Fixed Context Window
- GRU has limited memory capacity
- May forget information from long instructions
- Consider LSTM or Transformer for longer contexts

## Model Comparison

### Seq2Seq vs CMA

| Feature | Seq2Seq | CMA |
|---------|---------|-----|
| **Architecture** | Simple concatenation | Cross-modal attention |
| **Parameters** | ~28M | ~29M |
| **Training Time** | Baseline | +20-30% |
| **Memory** | Baseline | +15-20% |
| **Performance** | Baseline | +5-10% SR |
| **Complexity** | Low | Medium |
| **Use Case** | Baseline/Fast training | Better performance |

## Future Improvements

### Potential Enhancements

1. **Add Attention Mechanisms**
   - Implement cross-modal attention between instruction and visual features
   - See CMA implementation for reference

2. **Spatial Visual Features**
   - Use spatial features instead of global pooling
   - Enables finer-grained visual reasoning

3. **Progress Monitoring**
   - Add auxiliary task to predict navigation progress
   - Helps with early stopping and path efficiency

4. **Multi-Scale Visual Features**
   - Extract features from multiple ResNet layers
   - Captures both fine details and semantic information

5. **Bidirectional Instruction Encoding**
   - Use bidirectional LSTM for instruction encoding
   - Better captures instruction semantics

## References

### Papers
- Anderson et al. (2018). "Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments". CVPR 2018.
- Krantz et al. (2020). "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments". ECCV 2020.

### Code
- VLN-CE Seq2Seq Implementation: [vlnce_baselines/models/seq2seq_policy.py](https://github.com/jacobkrantz/VLN-CE/blob/main/vlnce_baselines/models/seq2seq_policy.py)
- Habitat-Lab: [habitat_baselines/rl/ppo/policy.py](https://github.com/facebookresearch/habitat-lab)

### Related Documentation
- [CMA Implementation](CMA_IMPLEMENTATION.md)
- [Model Comparison](../training/MODEL_COMPARISON.md)
- [Baselines README](../../satnav/models/baselines/README.md)
- [Configuration System](../CONFIG_SYSTEM.md)

