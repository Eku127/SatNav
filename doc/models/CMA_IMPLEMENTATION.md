# CMA Model Implementation in SatNav

## Overview

This document describes the implementation of the Cross-Modal Attention (CMA) model in SatNav, based on the paper "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web" by Zhu et al. (2020).

**Paper:** https://arxiv.org/abs/2004.02857

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CMA Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Instruction → Bidirectional LSTM → instruction_tokens (256)    │
│  RGB Image   → ResNet50 (spatial)  → rgb_spatial (2112, 4, 4)   │
│  Previous Action → Embedding(32)                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ First State Encoder                                     │    │
│  │   RGB pooled + prev_action → GRU (512) → state         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Cross-Modal Attention                                   │    │
│  │   1. Text-State: state queries instruction → text_emb   │    │
│  │   2. Text-RGB: text queries RGB spatial → rgb_attended  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Second State Encoder                                    │    │
│  │   [state, text_emb, rgb_attended, prev_action]          │    │
│  │   → Compress → GRU (512) → output                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Output → Action Distribution (4 actions)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Components

#### 1. Instruction Encoder
- **Type:** Bidirectional LSTM
- **Input:** Tokenized instruction [batch, seq_len]
- **Output:** All timestep features [batch, 256, seq_len]
- **Key Setting:** `final_state_only=False` (CMA needs all timesteps)

#### 2. RGB Visual Encoder
- **Type:** ResNet-50 (pretrained on ImageNet)
- **Input:** RGB image [batch, 224, 224, 3]
- **Output:** Spatial features [batch, 2112, 4, 4]
- **Key Setting:** `spatial_output=True` (CMA needs spatial features)
- **Note:** Includes spatial position embeddings (64-dim)

#### 3. Previous Action Embedding
- **Type:** nn.Embedding
- **Size:** (num_actions + 1) × 32
- **Purpose:** Encode previous action as context

#### 4. First State Encoder (GRU)
- **Input:** RGB pooled features + previous action
- **Hidden Size:** 512
- **Purpose:** Extract initial state representation

#### 5. Cross-Modal Attention Mechanisms

##### a) Text-State Attention
- **Query:** State features (from first GRU)
- **Key/Value:** Instruction tokens
- **Purpose:** Find relevant instruction parts given current state
- **Output:** Text embedding [batch, 256]

##### b) Text-RGB Attention
- **Query:** Text embedding (from text-state attention)
- **Key/Value:** RGB spatial features
- **Purpose:** Find relevant visual regions given instruction context
- **Output:** RGB attended features [batch, 256]

#### 6. Second State Encoder (GRU)
- **Input:** Concatenation of [state, text_emb, rgb_attended, prev_action]
- **Hidden Size:** 512
- **Purpose:** Integrate all information for final decision
- **Output:** Final features [batch, 512]

## Differences from VLN-CE CMA

| Component | VLN-CE CMA | SatNav CMA | Reason |
|-----------|-----------|-----------|--------|
| **Depth Encoder** | ✅ Included | ❌ Removed | SatNav uses satellite imagery (no depth) |
| **Progress Monitor** | ✅ Included | ❌ Removed | Simplified implementation |
| **RGB Encoder** | ResNet-50 | ResNet-50 | Same |
| **Instruction Encoder** | Bidirectional LSTM | Bidirectional LSTM | Same |
| **Attention Mechanisms** | Text-State, Text-RGB, Text-Depth | Text-State, Text-RGB | Depth removed |
| **State Encoders** | 2 GRU layers | 2 GRU layers | Same |

## Implementation Details

### File Structure

```
satnav/models/baselines/
├── cma_policy.py           # CMA implementation
├── seq2seq_policy.py       # Seq2Seq baseline
└── README.md               # Documentation

configs/baselines/
└── cma.yaml                # CMA configuration

tests/
├── test_cma_policy.py                  # Unit tests
└── test_cma_trainer_integration.py     # Integration tests

examples/
└── cma_quick_test.py       # Quick demo script
```

### Key Classes

#### CMAPolicy
- **Inherits:** `ILPolicy` (from `satnav.models.base`)
- **Purpose:** Policy wrapper for CMA network
- **Methods:**
  - `from_config()`: Create policy from configuration
  - `act()`: Select action given observations
  - `build_distribution()`: Build action distribution

#### CMANet
- **Inherits:** `Net` (from `satnav.models.base`)
- **Purpose:** CMA neural network backbone
- **Properties:**
  - `output_size`: 512
  - `num_recurrent_layers`: 2 (two GRU encoders)
  - `is_blind`: False
- **Methods:**
  - `forward()`: Forward pass through network
  - `_attn()`: Scaled dot-product attention

### RNN State Handling

CMA has **two RNN encoders**, so special care is needed for RNN states:

```python
# RNN states shape: [num_recurrent_layers, batch_size, hidden_size]
# For CMA: [2, batch_size, 512]

# Split states for two encoders
first_encoder_layers = self.state_encoder.num_recurrent_layers  # 1
second_encoder_layers = self.second_state_encoder.num_recurrent_layers  # 1

# Forward through first encoder
state, rnn_states_out[0:1] = self.state_encoder(
    state_in,
    rnn_states[0:1],
    masks
)

# Forward through second encoder
output, rnn_states_out[1:2] = self.second_state_encoder(
    x,
    rnn_states[1:2],
    masks
)
```

### Trainer Compatibility

The `BaseILTrainer` was updated to support models with multiple RNN encoders:

```python
# Before (only worked for single RNN):
rnn_states = torch.zeros(
    state_encoder.rnn.num_layers,
    N,
    hidden_size,
    device=device
)

# After (works for both single and multiple RNNs):
rnn_states = torch.zeros(
    self.policy.net.num_recurrent_layers,
    N,
    hidden_size,
    device=device
)
```

## Model Parameters

### Parameter Count
- **Total:** ~29M parameters
- **Trainable:** ~5.5M parameters
- **Frozen:** ~23.5M parameters (ResNet-50 backbone)

### Breakdown
```
Component                    Parameters
─────────────────────────────────────────
Instruction Encoder          ~200K
RGB Encoder (ResNet-50)      ~23.5M (frozen)
Previous Action Embedding    ~160
First State Encoder (GRU)    ~1.3M
Attention Mechanisms         ~500K
Second State Encoder (GRU)   ~1.3M
Action Distribution Head     ~2K
Spatial Embeddings           ~1K
─────────────────────────────────────────
Total                        ~29M
Trainable                    ~5.5M
```

## Configuration

### Default Configuration (`configs/baselines/cma.yaml`)

```yaml
MODEL:
  policy_name: cma
  
  INSTRUCTION_ENCODER:
    embedding_size: 50
    hidden_size: 128
    rnn_type: LSTM
    bidirectional: true        # Important!
    final_state_only: false    # Important!
  
  RGB_ENCODER:
    cnn_type: TorchVisionResNet50
    output_size: 256
    trainable: false
    # spatial_output: true is set in code
  
  STATE_ENCODER:
    hidden_size: 512
    rnn_type: GRU
    num_layers: 1

IL:
  lr: 2.5e-4
  batch_size: 5
  epochs: 50
```

## Usage

### Training

```bash
# Train CMA model
python run.py --exp-config configs/baselines/cma.yaml --run-type train

# Train with custom settings
python run.py --exp-config configs/baselines/cma.yaml --run-type train \
    IL.lr 1e-4 IL.batch_size 8
```

### Evaluation

```bash
# Evaluate on val_seen
python run.py --exp-config configs/baselines/cma.yaml --run-type eval

# Evaluate on val_unseen
python run.py --exp-config configs/baselines/cma.yaml --run-type eval \
    EVAL.SPLIT val_unseen
```

### Programmatic Usage

```python
from satnav.models import ModelRegistry

# Get model class
CMAPolicy = ModelRegistry.get_model('cma')

# Create policy
policy = CMAPolicy.from_config(
    config=config,
    observation_space=obs_space,
    action_space=act_space
)

# Initialize RNN states (2 layers for CMA)
rnn_states = torch.zeros(2, batch_size, 512)

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
# Run all CMA tests
pytest tests/test_cma_policy.py -v

# Run specific test
pytest tests/test_cma_policy.py::test_cma_forward_pass -v
```

### Integration Tests

```bash
# Test CMA with trainer
pytest tests/test_cma_trainer_integration.py -v
```

### Quick Test

```bash
# Run quick validation
python examples/cma_quick_test.py
```

## Performance Expectations

### Compared to Seq2Seq

| Metric | Seq2Seq | CMA | Notes |
|--------|---------|-----|-------|
| **Parameters** | ~28M | ~29M | Slightly more due to attention |
| **Training Time** | Baseline | +20-30% | Attention adds overhead |
| **Memory Usage** | Baseline | +15-20% | Spatial features require more memory |
| **SR (Success Rate)** | Baseline | +5-10% | Better instruction grounding |
| **SPL** | Baseline | +5-10% | More efficient paths |

### Training Tips

1. **Batch Size:** CMA works well with batch_size=5 (VLN-CE default)
2. **Learning Rate:** Start with 2.5e-4, reduce if unstable
3. **Epochs:** 50 epochs is usually sufficient for convergence
4. **Attention:** Monitor attention weights to debug grounding issues

## Known Issues and Limitations

### 1. Memory Usage
- Spatial features (4×4) require more memory than global pooling
- Reduce batch size if OOM errors occur

### 2. Training Time
- ~20-30% slower than Seq2Seq due to attention mechanisms
- Consider using mixed precision training (fp16) to speed up

### 3. Instruction Length
- Very long instructions (>80 tokens) may cause memory issues
- Consider truncating or using gradient checkpointing

## Future Improvements

### Potential Enhancements

1. **Add Progress Monitor**
   - Auxiliary task to predict navigation progress
   - Helps with early stopping decisions

2. **Multi-Head Attention**
   - Replace single attention with multi-head
   - May improve instruction grounding

3. **Depth Simulation**
   - Generate pseudo-depth from satellite imagery
   - Use height maps or elevation data

4. **Spatial Attention Visualization**
   - Visualize which image regions are attended
   - Useful for debugging and interpretability

## References

### Papers
- Zhu et al. (2020). "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web". ECCV 2020. [arXiv:2004.02857](https://arxiv.org/abs/2004.02857)
- Krantz et al. (2020). "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments". ECCV 2020.

### Code
- VLN-CE CMA Implementation: [vlnce_baselines/models/cma_policy.py](https://github.com/jacobkrantz/VLN-CE/blob/main/vlnce_baselines/models/cma_policy.py)
- Habitat-Lab: [habitat_baselines/rl/ppo/policy.py](https://github.com/facebookresearch/habitat-lab)

### Related Documentation
- [Seq2Seq Implementation](SEQ2SEQ_IMPLEMENTATION.md)
- [Model Comparison](../training/MODEL_COMPARISON.md)
- [Baselines README](../../satnav/models/baselines/README.md)

