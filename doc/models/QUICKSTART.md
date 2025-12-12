# Seq2Seq Model Quick Start Guide

This guide helps you get started with the Seq2Seq baseline model in SatNav.

## Prerequisites

1. **Install Dependencies**

```bash
# Activate your conda environment
conda activate satnav

# Install PyTorch and torchvision
pip install torch torchvision

# Or using conda
conda install pytorch torchvision -c pytorch
```

2. **Download GloVe Embeddings** (if using pretrained embeddings)

```bash
# Download GloVe 6B
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove/
```

## Step-by-Step Usage

### Step 1: Prepare Vocabulary and Embeddings

```bash
# Create necessary directories
mkdir -p data/vocab data/embeddings

# Build vocabulary from your dataset
python -m satnav.utils.build_vocab \
    --dataset tests/test_data/satnav_dataset_complex.json \
    --output data/vocab/vocab.json

# Generate GloVe embeddings
python -m satnav.utils.build_glove_embeddings \
    --vocab data/vocab/vocab.json \
    --glove data/glove/glove.6B.50d.txt \
    --output data/embeddings/glove_embeddings.json.gz
```

### Step 2: Test the Model

```bash
# Run unit tests to verify installation
pytest tests/test_seq2seq_model.py -v

# If pytest not installed:
pip install pytest
```

### Step 3: Use the Model

```python
import torch
from omegaconf import OmegaConf
from satnav.models import ModelRegistry

# Load configuration
config = OmegaConf.load("configs/baselines/seq2seq.yaml")

# Update config with vocab info
vocab_data = json.load(open("data/vocab/vocab.json"))
config.MODEL.INSTRUCTION_ENCODER.vocab_size = vocab_data["vocab_size"]

# Get model from registry
model_class = ModelRegistry.get_model("seq2seq")

# Create mock spaces (replace with actual spaces from your environment)
class MockSpace:
    n = 4  # Number of actions

obs_space = MockSpace()
act_space = MockSpace()

# Instantiate model
model = model_class.from_config(config, obs_space, act_space)

# Initialize model states
batch_size = 1
device = torch.device('cpu')
rnn_states = model.net.get_initial_state(batch_size, device)

# Prepare observations
observations = {
    "instruction": torch.tensor([[2, 3, 4, 5, 0]]),  # Tokenized instruction
    "rgb": torch.randint(0, 256, (1, 224, 224, 3)).float(),  # RGB image
}
prev_actions = torch.zeros(1, 1).long()
masks = torch.ones(1, 1)

# Get action
with torch.no_grad():
    action, rnn_states = model.act(
        observations, rnn_states, prev_actions, masks,
        deterministic=True  # Use argmax for evaluation
    )

print(f"Predicted action: {action.item()}")
```

## Common Workflows

### Without GloVe (Random Initialization)

If you don't want to use GloVe embeddings:

```yaml
# In configs/baselines/seq2seq.yaml
MODEL:
  INSTRUCTION_ENCODER:
    use_pretrained_embeddings: false  # Disable GloVe
    vocab_size: 150  # Set your vocabulary size
```

### Fine-tuning Visual Encoder

To fine-tune the ResNet backbone:

```yaml
MODEL:
  RGB_ENCODER:
    trainable: true  # Enable gradients for ResNet
```

### Using ResNet-18 Instead of ResNet-50

For faster inference:

```yaml
MODEL:
  RGB_ENCODER:
    cnn_type: TorchVisionResNet18  # Lighter model
```

## Troubleshooting

**Issue**: `ImportError: No module named 'torch'`
```bash
pip install torch torchvision
```

**Issue**: `FileNotFoundError: embeddings.json.gz`
```bash
# Make sure you ran Step 1 to generate embeddings
# Or disable pretrained embeddings in config
```

**Issue**: Tests fail with CUDA errors
```bash
# Run tests on CPU
pytest tests/test_seq2seq_model.py -v --tb=short
```

## Next Steps

- **Training**: See [Training Guide](../training/TRAINING_GUIDE.md) for training the model
  ```bash
  python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train
  ```
- **Evaluation**: Run model on validation set
  ```bash
  python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval
  ```
- **Custom Models**: Extend Seq2Seq to create attention-based models
- **Documentation**: See `doc/models/SEQ2SEQ_IMPLEMENTATION.md` for architecture details

## Quick Reference

| Task | Command |
|------|---------|
| Build vocab | `python -m satnav.utils.build_vocab --dataset DATA --output OUT` |
| Build embeddings | `python -m satnav.utils.build_glove_embeddings --vocab VOCAB --glove GLOVE --output OUT` |
| Run tests | `pytest tests/test_seq2seq_model.py -v` |
| List models | `python -c "from satnav.models import ModelRegistry; print(ModelRegistry.list_models())"` |

## Resources

- **Seq2Seq Documentation**: `doc/models/SEQ2SEQ_IMPLEMENTATION.md`
- **CMA Documentation**: `doc/models/CMA_IMPLEMENTATION.md`
- **Embedding Guide**: `doc/EMBEDDING_GUIDE.md`
- **Baseline README**: `satnav/models/baselines/README.md`
- **VLN-CE Repository**: https://github.com/jacobkrantz/VLN-CE

