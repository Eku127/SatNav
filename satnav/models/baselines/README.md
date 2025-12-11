# SatNav Baseline Models

This directory contains baseline model implementations for Vision-and-Language Navigation (VLN) in SatNav.

## Available Models

### 1. Seq2Seq Policy

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

**Parameters:** ~28M total, ~5M trainable (ResNet frozen)

**Configuration:** `configs/baselines/seq2seq.yaml`

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

---

### 2. CMA Policy (Cross-Modal Attention)

An attention-based model that uses cross-modal attention between instruction and visual features. Based on "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web" (Zhu et al., 2020).

**Paper:** https://arxiv.org/abs/2004.02857

**Architecture:**
```
Instruction → Bidirectional LSTM → instruction_tokens (256)
RGB Image   → ResNet50 (spatial)  → rgb_spatial_features (2112, 4, 4)
Previous Action → Embedding(32)

RGB pooled + prev_action → First GRU (512) → state

Cross-Modal Attention:
  1. Text-State Attention: state queries instruction → text_embedding
  2. Text-RGB Attention: text queries RGB spatial → rgb_attended

[state, text_embedding, rgb_attended, prev_action] 
  → Compress → Second GRU (512) → action logits
```

**Key Features:**
- Bidirectional LSTM instruction encoder (returns all timesteps)
- ResNet-50 with spatial output (4×4 feature maps)
- Dual GRU state encoders with cross-modal attention
- Attention mechanisms:
  - Text-State: State queries instruction for relevant text
  - Text-RGB: Text queries RGB spatial features for relevant visual regions

**Differences from VLN-CE CMA:**
- ❌ NO depth encoder (SatNav uses satellite imagery without depth)
- ❌ NO progress monitor (simplified implementation)
- ✅ Core CMA mechanism preserved (RGB-instruction cross-modal attention)

**Parameters:** ~29M total, ~5.5M trainable (ResNet frozen)

**Configuration:** `configs/baselines/cma.yaml`

**Usage:**
```python
from satnav.models import ModelRegistry

# Get model class
model_class = ModelRegistry.get_model("cma")

# Create model instance
model = model_class.from_config(config, obs_space, act_space)

# Forward pass
action, rnn_states = model.act(observations, rnn_states, prev_actions, masks)

# Note: CMA has 2 RNN encoders, so num_recurrent_layers = 2
```

**Training:**
```bash
# Train CMA model
python run.py --exp-config configs/baselines/cma.yaml --run-type train

# Evaluate CMA model
python run.py --exp-config configs/baselines/cma.yaml --run-type eval
```

---

## Model Comparison

| Model | Parameters (Total) | Parameters (Trainable) | Attention | Depth | Complexity |
|-------|-------------------|------------------------|-----------|-------|------------|
| Seq2Seq | ~28M | ~5M | ❌ | ❌ | Low |
| CMA | ~29M | ~5.5M | ✅ | ❌ | Medium |

**Notes:**
- Both models freeze ResNet-50 backbone by default
- CMA has slightly more parameters due to attention mechanisms
- CMA typically achieves better performance but requires more training time

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

## References

### Papers
- **CMA:** Zhu et al. (2020). "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web". ECCV 2020. [arXiv:2004.02857](https://arxiv.org/abs/2004.02857)
- **VLN-CE:** Krantz et al. (2020). "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments". ECCV 2020.

### Code
- **VLN-CE:** [https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- **Habitat-Lab:** [https://github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab)

### Documentation
- **Seq2Seq Implementation:** `doc/models/SEQ2SEQ_IMPLEMENTATION.md`
- **CMA Implementation:** `doc/models/CMA_IMPLEMENTATION.md`

---

## Non-learning Agents

### Random Agent

A random baseline agent that samples actions according to a probability distribution. This provides the simplest possible baseline - any learning-based agent should significantly outperform a random agent.

#### Features
- No training required
- Samples actions based on dataset statistics or manual probabilities
- Performance reflects dataset's average behavior

#### Configuration

```yaml
MODEL:
  policy_name: random
  RANDOM_AGENT:
    # Option 1: Compute from training data (recommended)
    stats_dataset: data/debug_data/train/train.json
    
    # Option 2: Manual probabilities (optional)
    # action_probs: [0.02, 0.68, 0.15, 0.15]  # STOP, FORWARD, LEFT, RIGHT

EVAL:
  SPLIT: val_seen
  CKPT_PATH: null  # No checkpoint needed
```

#### Usage

```bash
# Evaluate RandomAgent
python run.py --exp-config configs/baselines/random_agent.yaml --run-type eval

# With video generation
python run.py --exp-config configs/baselines/random_agent.yaml --run-type eval GENERATE_VIDEOS=true
```

#### Configuration

The configuration inherits from `configs/default.yaml` for common settings like `IL` and `STATE_ENCODER`. Only agent-specific parameters need to be specified:

```yaml
_base_: configs/default.yaml  # Inherit default configurations

MODEL:
  policy_name: random
  RANDOM_AGENT:
    action_probs: [0.02, 0.68, 0.15, 0.15]  # STOP, FORWARD, LEFT, RIGHT

EVAL:
  CKPT_PATH: null  # No checkpoint needed

GENERATE_VIDEOS: false  # Set to true for video generation
```

#### Expected Performance
- SPL: ~0.0-0.1 (very low, as expected for random behavior)
- Success: ~0% (random actions rarely reach the goal)

### Greedy Agent

A greedy baseline agent that follows the reference path waypoints sequentially. This provides an oracle-like upper bound for VLN performance.

#### Features
- No training required
- Uses ground-truth reference path waypoints
- Always selects the optimal action to reach current waypoint
- Reuses `SatNavPathFollower` for navigation logic
- Follows the same logic as `examples/satnav_path_follower_example.py`

#### Algorithm
1. Extract waypoints from episode's reference_path
2. Navigate to current waypoint using greedy strategy:
   - Calculate distance and bearing to current waypoint
   - If distance < goal_radius: Move to next waypoint
   - If heading is close to target bearing: MOVE_FORWARD
   - Otherwise: TURN_LEFT or TURN_RIGHT toward waypoint
3. When all waypoints reached: STOP

#### Configuration

```yaml
MODEL:
  policy_name: greedy
  GREEDY_AGENT:
    goal_radius: 3.0   # Distance threshold (meters)
    turn_angle: 15.0   # Turn angle (degrees)

EVAL:
  SPLIT: val_seen
  CKPT_PATH: null  # No checkpoint needed
```

#### Usage

```bash
# Evaluate GreedyAgent
python run.py --exp-config configs/baselines/greedy_agent.yaml --run-type eval

# With video generation
python run.py --exp-config configs/baselines/greedy_agent.yaml --run-type eval GENERATE_VIDEOS=true
```

#### Configuration

The configuration inherits from `configs/default.yaml` for common settings. Only agent-specific parameters need to be specified:

```yaml
_base_: configs/default.yaml  # Inherit default configurations

MODEL:
  policy_name: greedy
  GREEDY_AGENT:
    goal_radius: 3.0   # Distance threshold (meters)
    turn_angle: 15.0   # Turn angle (degrees)

EVAL:
  CKPT_PATH: null  # No checkpoint needed

GENERATE_VIDEOS: false  # Set to true for video generation
```

#### Expected Performance
- SPL: ~0.7-1.0 (high, as it has perfect goal knowledge)
- Success: ~80-100% (depends on environment complexity)

#### Note
GreedyAgent requires environment access to get current position and extract waypoints from the reference path. This is automatically handled by the evaluator via the `set_env()` method, which is called at the start of each episode.

