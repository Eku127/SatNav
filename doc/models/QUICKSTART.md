# Baseline Model Quickstart

This guide runs the Seq2Seq and CMA baselines with the same default model
settings used for normal SatNav training: GloVe 50d instruction embeddings and
TorchVision ResNet50 ImageNet weights.

The bundled example data is intentionally tiny. It is only for checking that the
pipeline works end to end.

## Install

From the repository root:

```bash
conda activate satnav
pip install -r requirements.txt
```

`curl` or `wget` and `unzip` are needed to download and extract GloVe.
If you already have `glove.6B.50d.txt`, you can set `LOCAL_GLOVE_TXT` when
running the script to reuse that local file.

## One-Command Run

```bash
bash scripts/quickstart_models.sh
```

This script:

1. Builds a vocabulary from `applications/resources/satnav_example_episodes.json`.
2. Downloads GloVe 6B and builds `satnav_example_glove50d.json.gz`.
3. Downloads TorchVision ResNet50 weights if they are not cached.
4. Generates offline trajectory data from `applications/resources/map.tif`.
5. Trains and evaluates Seq2Seq and CMA.

All generated files are under:

```bash
output/quickstart_baselines/
```

Expected checkpoints:

```bash
output/quickstart_baselines/seq2seq/checkpoints/latest/best.pth
output/quickstart_baselines/cma/checkpoints/latest/best.pth
```

## Step-by-Step

Build the vocabulary:

```bash
python -m satnav.utils.build_vocab \
  --dataset applications/resources/satnav_example_episodes.json \
  --output output/quickstart_baselines/artifacts/vocab/satnav_example_vocab.json
```

Download GloVe and build embeddings:

```bash
mkdir -p output/quickstart_baselines/artifacts/glove
curl -L http://nlp.stanford.edu/data/glove.6B.zip \
  -o output/quickstart_baselines/artifacts/glove/glove.6B.zip
unzip -o output/quickstart_baselines/artifacts/glove/glove.6B.zip \
  glove.6B.50d.txt \
  -d output/quickstart_baselines/artifacts/glove

python -m satnav.utils.build_glove_embeddings \
  --vocab output/quickstart_baselines/artifacts/vocab/satnav_example_vocab.json \
  --glove output/quickstart_baselines/artifacts/glove/glove.6B.50d.txt \
  --output output/quickstart_baselines/artifacts/embeddings/satnav_example_glove50d.json.gz \
  --embedding-dim 50
```

Download ResNet50 weights:

```bash
python -c "from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"
```

Generate offline trajectory data:

```bash
python -m applications.trajectory_generation.generate \
  --config applications/resources/satnav_example_task.yaml \
  --output_dir output/quickstart_baselines/trajectory_data
```

See [trajectory_generation/README.md](../../applications/trajectory_generation/README.md)
for production trajectory generation.

Train and evaluate Seq2Seq:

```bash
python run.py \
  --exp-config configs/baselines/seq2seq_offline_train.yaml \
  --run-type train

python run.py \
  --exp-config configs/baselines/seq2seq_eval.yaml \
  --run-type eval
```

Train and evaluate CMA:

```bash
python run.py \
  --exp-config configs/baselines/cma_offline_train.yaml \
  --run-type train

python run.py \
  --exp-config configs/baselines/cma_eval.yaml \
  --run-type eval
```

## Default Paths

The default baseline YAMLs are runnable without path edits after the preparation
steps above:

| File | Purpose |
| --- | --- |
| `applications/resources/satnav_example_episodes.json` | Example episodes |
| `applications/resources/satnav_example_task.yaml` | Example task and scene config |
| `applications/resources/map.tif` | Example GeoTIFF scene |
| `output/quickstart_baselines/artifacts/vocab/satnav_example_vocab.json` | Generated vocab |
| `output/quickstart_baselines/artifacts/embeddings/satnav_example_glove50d.json.gz` | Generated GloVe embeddings |
| `output/quickstart_baselines/trajectory_data/` | Generated offline data |

For real experiments, keep the model settings and replace only the dataset,
scene, trajectory, vocab, checkpoint, and result paths.

For local development, you can skip the GloVe download if the file already
exists:

```bash
LOCAL_GLOVE_TXT=/path/to/glove.6B.50d.txt bash scripts/quickstart_models.sh
```
