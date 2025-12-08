# Seq2Seq Baseline Model for SatNav

**Author**: SatNav Development Team  
**Date**: December 2025  
**Status**: Initial Implementation

## 1. Model Overview

The Seq2Seq (Sequence-to-Sequence) model is a baseline architecture for Vision-and-Language Navigation (VLN) in SatNav. It encodes natural language instructions and visual observations separately, then uses a recurrent network to produce action distributions.

**Key Characteristics:**
- **Task**: Vision-and-Language Navigation in continuous environments
- **Input**: Natural language instruction + RGB satellite imagery
- **Output**: Discrete action probabilities (STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT)
- **Architecture**: Encoder-Decoder with RNN state management

**Use Cases:**
- Baseline for comparing new VLN models
- Initial training and evaluation on SatNav datasets
- Foundation for developing more complex models (e.g., attention-based)

## 2. Architecture Design

### 2.1 Network Structure

```
┌─────────────────┐
│   Instruction   │ "Go forward, turn left..."
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LSTM Encoder    │ (embedding_size=50, hidden_size=128)
│ (GloVe 50d)     │
└────────┬────────┘
         │
         ▼
   instruction_embedding (128)
         │
         │         ┌─────────────────┐
         │         │   RGB Image     │ Satellite overhead view
         │         └────────┬────────┘
         │                  │
         │                  ▼
         │         ┌─────────────────┐
         │         │ ResNet-50       │ (ImageNet pretrained)
         │         │ Visual Encoder  │
         │         └────────┬────────┘
         │                  │
         │                  ▼
         │            rgb_embedding (256)
         │                  │
         └──────────┬───────┘
                    │
         ┌──────────▼──────────┐
         │  [prev_action_emb]  │ (Optional, 32)
         └──────────┬──────────┘
                    │
                    ▼
              Concatenate (128 + 256 [+ 32])
                    │
                    ▼
         ┌──────────────────────┐
         │   GRU State Encoder  │ (hidden_size=512)
         │  (Recurrent Memory)  │
         └──────────┬───────────┘
                    │
                    ▼
              hidden_state (512)
                    │
                    ▼
         ┌──────────────────────┐
         │   Action Head        │ Linear + Softmax
         │  (in ILPolicy)       │
         └──────────┬───────────┘
                    │
                    ▼
         action_logits (4 actions)
```

### 2.2 Components

#### Instruction Encoder
- **Type**: LSTM (uni-directional)
- **Input**: Tokenized instruction indices [batch, seq_len]
- **Embedding**: GloVe 50d (or random init)
  - Index 0: PAD (zeros)
  - Index 1: UNK (mean of all embeddings)
  - Index 2+: Word embeddings
- **Output**: instruction_embedding [batch, 128]
- **Reference**: VLN-CE `instruction_encoder.py`

#### Visual Encoder
- **Type**: ResNet-50 (torchvision pretrained on ImageNet)
- **Input**: RGB satellite image [batch, H, W, 3] in range [0, 255]
- **Preprocessing**: Scale to [0, 1], optional ImageNet normalization
- **Output**: rgb_embedding [batch, 256]
- **Trainable**: False (frozen by default)
- **Note**: SatNav uses satellite overhead imagery (NO depth encoder)
- **Reference**: VLN-CE `resnet_encoders.py` (TorchVisionResNet50)

#### State Encoder
- **Type**: GRU (uni-directional, 1 layer)
- **Input**: Concatenated embeddings [batch, 384 or 416]
- **Hidden State**: [1, batch, 512]
- **Output**: hidden_state [batch, 512]
- **Purpose**: Maintains temporal context across navigation steps
- **Reference**: Habitat-Lab `rnn_state_encoder.py`

#### Action Head
- **Type**: Linear layer + Categorical distribution
- **Input**: hidden_state [batch, 512]
- **Output**: action_logits [batch, 4]
- **Actions**: {STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT}
- **Sampling**: Categorical distribution (training: sample, eval: argmax)

## 3. Comparison with VLN-CE

| Component | VLN-CE (Indoor) | SatNav (Satellite) |
|-----------|-----------------|-------------------|
| **Instruction Encoder** | LSTM (128) | ✓ Same |
| **Embedding** | GloVe 50d | ✓ Same |
| **RGB Encoder** | ResNet-50 (256) | ✓ Same |
| **Depth Encoder** | ResNet-50 (128) | ✗ Removed (no depth) |
| **State Encoder** | GRU (512) | ✓ Same |
| **Prev Action** | Embedding (32) | ✓ Same (optional) |
| **Action Space** | 4 discrete | ✓ Same |

**Key Difference**: SatNav uses satellite overhead RGB imagery without depth information, so the depth encoder from VLN-CE is removed. This reduces the RNN input size from 512 (128+256+128) to 384 (128+256).

## 4. Configuration Parameters

### Default Configuration

See `configs/baselines/seq2seq.yaml`:

```yaml
MODEL:
  INSTRUCTION_ENCODER:
    embedding_size: 50           # GloVe dimension
    hidden_size: 128             # LSTM hidden size
    rnn_type: LSTM
    bidirectional: false
    use_pretrained_embeddings: true
    embedding_file: data/embeddings/glove_embeddings.json.gz
  
  RGB_ENCODER:
    cnn_type: TorchVisionResNet50
    output_size: 256
    trainable: false             # Freeze ResNet
  
  STATE_ENCODER:
    hidden_size: 512
    rnn_type: GRU
  
  SEQ2SEQ:
    use_prev_action: true        # Use action history
```

### Tunable Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `INSTRUCTION_ENCODER.hidden_size` | 128 | 64-256 | Instruction encoding capacity |
| `RGB_ENCODER.output_size` | 256 | 128-512 | Visual feature richness |
| `STATE_ENCODER.hidden_size` | 512 | 256-1024 | Temporal reasoning capacity |
| `embedding_size` | 50 | 50-300 | Semantic representation |
| `use_prev_action` | true | bool | Action history influence |

## 5. Usage Examples

### 5.1 Basic Usage

```python
from omegaconf import OmegaConf
from satnav.models import ModelRegistry

# Load configuration
config = OmegaConf.load("configs/baselines/seq2seq.yaml")

# Get model class from registry
model_class = ModelRegistry.get_model("seq2seq")

# Create model instance
model = model_class.from_config(config, observation_space, action_space)

# Initialize RNN states
batch_size = 4
num_recurrent_layers = model.net.num_recurrent_layers
hidden_size = config.MODEL.STATE_ENCODER.hidden_size
rnn_states = torch.zeros(num_recurrent_layers, batch_size, hidden_size)

# Forward pass
observations = {
    "instruction": instruction_tokens,  # [batch, seq_len]
    "rgb": rgb_images,                  # [batch, H, W, 3]
}
prev_actions = torch.zeros(batch_size, 1).long()
masks = torch.ones(batch_size, 1)

# Get action
action, rnn_states = model.act(
    observations, rnn_states, prev_actions, masks,
    deterministic=False  # Sample during training, True for evaluation
)
```

### 5.2 With Custom Configuration

```python
# Modify configuration
config.MODEL.INSTRUCTION_ENCODER.hidden_size = 256
config.MODEL.STATE_ENCODER.hidden_size = 1024
config.MODEL.RGB_ENCODER.trainable = True  # Fine-tune ResNet

# Create model with custom config
model = model_class.from_config(config, obs_space, act_space)
```

### 5.3 Loading Pretrained Weights

```python
from satnav.models.utils import load_checkpoint

# Load checkpoint
checkpoint = load_checkpoint(
    "checkpoints/seq2seq_best.pth",
    model=model,
    device="cuda"
)
epoch = checkpoint["epoch"]
print(f"Loaded model from epoch {epoch}")
```

## 6. Training Considerations

### 6.1 Data Requirements

- **Vocabulary**: Extract from dataset or use provided vocabulary
- **GloVe Embeddings**: Download glove.6B.50d.txt (171MB)
- **RGB Images**: Satellite overhead view, any resolution (will be resized by ResNet)

### 6.2 Typical Hyperparameters

```python
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

# Training
batch_size = 5
epochs = 15
gradient_clipping = 5.0

# DAgger (if using imitation learning)
dagger_iterations = 1
update_size = 10819
p = 1.0  # Teacher forcing probability
```

### 6.3 Evaluation Metrics

- **Success Rate (SR)**: % of episodes reaching goal
- **Oracle Success Rate (OSR)**: % episodes where agent passed near goal
- **Success weighted by Path Length (SPL)**: Efficiency metric
- **Navigation Error (NE)**: Distance to goal at episode end

## 7. Extension Guide

### 7.1 Adding Attention Mechanism

To create a CMA (Cross-Modal Attention) variant:

1. Set `INSTRUCTION_ENCODER.final_state_only = False` to get full sequence
2. Add attention module between instruction and visual features
3. Use spatial visual features (`RGB_ENCODER.spatial_output = True`)

Example structure:
```python
instruction_sequence = instruction_encoder(obs)  # [batch, 128, seq_len]
visual_features = rgb_encoder(obs)               # [batch, 256, 4, 4]

# Apply attention
attended_features = attention(instruction_sequence, visual_features)

# Continue with state encoder
x = torch.cat([attended_features, ...], dim=1)
```

### 7.2 Multi-Modal Fusion

Add additional modalities:
```python
# Add compass/GPS encoder
compass_embedding = compass_encoder(observations["compass"])

# Concatenate with other embeddings
x = torch.cat([
    instruction_embedding,
    rgb_embedding,
    compass_embedding,  # New modality
    prev_action_embedding
], dim=1)
```

### 7.3 Hierarchical Policy

Use Seq2Seq as low-level controller:
```python
# High-level planner
subgoal = high_level_planner(instruction, current_state)

# Low-level Seq2Seq policy
action = seq2seq_policy(subgoal_instruction, observations)
```

## 8. Reference Materials

### 8.1 Source Code References

| Component | VLN-CE Reference | SatNav Implementation |
|-----------|-----------------|----------------------|
| ILPolicy | `vlnce_baselines/models/policy.py` | `satnav/models/base.py` |
| Seq2SeqNet | `vlnce_baselines/models/seq2seq_policy.py` | `satnav/models/baselines/seq2seq_policy.py` |
| Instruction Encoder | `vlnce_baselines/models/encoders/instruction_encoder.py` | `satnav/models/encoders/instruction_encoder.py` |
| Visual Encoder | `vlnce_baselines/models/encoders/resnet_encoders.py` | `satnav/models/encoders/visual_encoder.py` |
| State Encoder | `habitat_baselines/rl/models/rnn_state_encoder.py` | `satnav/models/encoders/rnn_state_encoder.py` |

### 8.2 Papers

- **VLN-CE**: "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments" (ECCV 2020)
- **Room-to-Room**: "Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments" (CVPR 2018)
- **Habitat**: "Habitat: A Platform for Embodied AI Research" (ICCV 2019)

### 8.3 External Resources

- **VLN-CE GitHub**: https://github.com/jacobkrantz/VLN-CE
- **Habitat-Lab GitHub**: https://github.com/facebookresearch/habitat-lab  
- **GloVe Embeddings**: https://nlp.stanford.edu/projects/glove/
- **Embedding Guide**: `doc/EMBEDDING_GUIDE.md`

## 9. Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Install PyTorch: `conda install pytorch torchvision -c pytorch`

**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Reduce batch size or use CPU: `config.TORCH_GPU_ID = -1`

**Issue**: Low coverage when building embeddings
- **Solution**: Check vocabulary quality or use larger GloVe model (100d/300d)

**Issue**: Model overfits quickly
- **Solution**: Enable dropout, reduce model capacity, or increase dataset size

**Issue**: Actions are always the same
- **Solution**: Check RNN state initialization and mask handling

## 10. Future Improvements

Potential enhancements to the baseline:

1. **Attention Mechanisms**: Cross-modal attention between instruction and visual features
2. **Progress Monitoring**: Auxiliary task to predict navigation progress
3. **Auxiliary Losses**: Action prediction, distance estimation
4. **Recollection Training**: More efficient training with cached trajectories
5. **Waypoint Prediction**: Predict intermediate waypoints instead of low-level actions
6. **Transformer Encoders**: Replace LSTM with Transformer for instruction encoding
7. **Visual Grounding**: Explicit grounding of instruction phrases to image regions

## Appendix: Model Statistics

**Parameter Counts** (with default config):
- Instruction Encoder: ~50K parameters
- Visual Encoder (frozen): ~23M parameters (not trained)
- State Encoder: ~1.3M parameters
- Action Head: ~2K parameters
- **Total Trainable**: ~1.35M parameters

**Memory Usage**:
- Model: ~50 MB
- Batch (size=5): ~200 MB
- Peak Training (GPU): ~2 GB

**Inference Speed** (on Tesla V100):
- Single step: ~5 ms
- Batched (batch=32): ~20 ms (~0.6 ms per sample)

