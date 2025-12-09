# SatNav Baseline Models

This directory contains baseline model implementations for Vision-and-Language Navigation (VLN) in SatNav.

## Available Models

### Seq2Seq Policy

A sequence-to-sequence baseline model that encodes instructions and visual observations using separate encoders, then uses an RNN to produce action distributions.

**Architecture:**
```
Instruction → LSTM Encoder → instruction_embedding (128)
RGB Image   → ResNet50     → rgb_embedding (256)
[Previous Action → Embedding(32)]

Concatenate → GRU State Encoder (512) → action logits
```

**Key Features:**
- LSTM-based instruction encoder with GloVe embeddings (50d)
- ResNet-50 visual encoder (ImageNet pretrained)
- GRU-based state encoder
- Optional previous action embedding

**Configuration:**
See `configs/baselines/seq2seq.yaml` for default parameters.

**Usage:**
```python
from satnav.models import ModelRegistry

# Get model class
model_class = ModelRegistry.get_model("seq2seq")

# Create model instance
model = model_class.from_config(config, obs_space, act_space)

# Forward pass
action, rnn_states = model.act(observations, rnn_states, prev_actions, masks)
```

## Adding Custom Models

To add a custom baseline model:

1. Create a new file in this directory (e.g., `my_model.py`)
2. Implement your model class inheriting from `ILPolicy`
3. Register the model in `satnav/models/__init__.py`:
   ```python
   from satnav.models.baselines.my_model import MyModel
   ModelRegistry.register_baseline("my_model", MyModel)
   ```
4. Create a configuration file in `configs/baselines/my_model.yaml`

## Reference

- **VLN-CE:** [https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- **Habitat-Lab:** [https://github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)
- **Design Document:** `doc/models/SEQ2SEQ_BASELINE.md`

