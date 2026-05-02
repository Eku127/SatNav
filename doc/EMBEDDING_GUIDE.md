# GloVe Embedding Generation Guide

This guide explains how to generate word embeddings for SatNav datasets using GloVe pretrained vectors.

## Prerequisites

1. **Download GloVe Vectors**
   
   Download the GloVe 6B (trained on Wikipedia + Gigaword) from Stanford NLP:
   
   ```bash
   # Option 1: Direct download
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   
   # Option 2: Using curl
   curl -O http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```
   
   This will extract several files:
   - `glove.6B.50d.txt` (171 MB) ← We use this one
   - `glove.6B.100d.txt`
   - `glove.6B.200d.txt`
   - `glove.6B.300d.txt`

2. **SatNav Dataset**
   
   You need a SatNav dataset JSON file with episodes containing instructions.

## Step-by-Step Process

### Step 1: Build Vocabulary

First, extract vocabulary from your dataset:

```bash
python -m satnav.utils.build_vocab \
    --dataset path/to/your/dataset.json \
    --output data/vocab/vocab.json \
    --min-count 1
```

**Options:**
- `--dataset`: Path to your SatNav dataset JSON file
- `--output`: Where to save the vocabulary file
- `--min-count`: Minimum word frequency (default: 1)
- `--no-dataset-vocab`: Ignore dataset vocabulary and extract from episodes

**Output:**
The script creates `vocab.json` containing:
```json
{
  "word2idx": {"<pad>": 0, "<unk>": 1, "go": 2, "to": 3, ...},
  "idx2word": ["<pad>", "<unk>", "go", "to", ...],
  "vocab_size": N
}
```

### Step 2: Generate Embeddings

Build GloVe embeddings for your vocabulary:

```bash
python -m satnav.utils.build_glove_embeddings \
    --vocab data/vocab/vocab.json \
    --glove path/to/glove.6B.50d.txt \
    --output data/embeddings/glove_embeddings.json.gz \
    --embedding-dim 50
```

**Options:**
- `--vocab`: Path to vocabulary JSON file (from Step 1)
- `--glove`: Path to GloVe text file
- `--output`: Where to save the embeddings file
- `--embedding-dim`: Embedding dimension (50 for glove.6B.50d)

**Output:**
The script creates `embeddings.json.gz` (gzipped JSON) with a 2D array of shape `[vocab_size, 50]`:
- Index 0: PAD token (all zeros)
- Index 1: UNK token (mean of all embeddings)
- Index 2+: Word embeddings

### Step 3: Configure Model

Update your model config to use the embeddings:

```yaml
# configs/baselines/seq2seq_offline_train.yaml
MODEL:
  INSTRUCTION_ENCODER:
    use_pretrained_embeddings: true
    embedding_file: data/embeddings/glove_embeddings.json.gz
    embedding_size: 50
    vocab_size: null  # Auto-detected from embeddings
```

## Example with Test Data

Quick example using the provided test data:

```bash
# Step 1: Build vocabulary
python -m satnav.utils.build_vocab \
    --dataset tests/test_data/satnav_dataset_complex.json \
    --output data/vocab/test_vocab.json

# Step 2: Generate embeddings (assuming you have GloVe downloaded)
python -m satnav.utils.build_glove_embeddings \
    --vocab data/vocab/test_vocab.json \
    --glove /path/to/glove.6B.50d.txt \
    --output data/embeddings/test_embeddings.json.gz
```

## Embedding Format (VLN-CE Compatible)

The generated embeddings are fully compatible with VLN-CE format:

```python
import gzip
import json

# Load embeddings
with gzip.open("data/embeddings/glove_embeddings.json.gz", "rt") as f:
    embeddings = json.load(f)

# embeddings is a list of lists: [[0, 0, ...], [mean, mean, ...], [word1_emb], ...]
print(f"Shape: [{len(embeddings)}, {len(embeddings[0])}]")
print(f"PAD embedding (index 0): {embeddings[0]}")  # All zeros
print(f"UNK embedding (index 1): {embeddings[1][:5]}...")  # Mean vector
```

## Troubleshooting

### Word Coverage

Check how many words from your vocabulary are found in GloVe:

```
Embedding statistics:
  Total vocabulary: 150
  Found in GloVe: 135 (90.0%)
  Not found (using UNK): 15
```

- Coverage > 80%: Good
- Coverage < 50%: Consider checking your vocabulary or using a larger GloVe model

### Memory Issues

If you encounter memory issues with large vocabularies:

1. Use `--min-count 2` or higher in Step 1 to reduce vocabulary size
2. Consider using a smaller GloVe model (50d instead of 300d)

### Missing GloVe Words

Words not found in GloVe will use the UNK embedding (mean of all found embeddings). This is standard practice and shouldn't significantly impact performance for small numbers of missing words.

## Advanced: Custom Embeddings

If you want to use custom pretrained embeddings instead of GloVe:

1. Convert your embeddings to the same format:
   ```python
   import gzip
   import json
   import numpy as np
   
   # Your embeddings: dict mapping word to vector
   custom_embeddings = {...}
   
   # Build matrix with PAD and UNK
   vocab_size = len(word2idx)
   embedding_dim = 50
   embeddings = np.zeros((vocab_size, embedding_dim))
   
   # ... (similar logic as build_glove_embeddings.py)
   
   # Save
   with gzip.open("output.json.gz", "wt") as f:
       json.dump(embeddings.tolist(), f)
   ```

2. Update the config to point to your custom embeddings file

## References

- **GloVe**: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **VLN-CE Embeddings**: See `vlnce_baselines/models/encoders/instruction_encoder.py`
- **Why average for UNK**: [https://bit.ly/3u3hkYg](https://bit.ly/3u3hkYg)

