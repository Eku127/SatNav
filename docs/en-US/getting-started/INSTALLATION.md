# SatNav Installation

This guide installs SatNav from source. SatNav Core, Classic baselines, and
each VLM baseline have different dependency boundaries; choose the environment
for the workflow you intend to run.

## 1. Get the source

```bash
git clone https://github.com/Eku127/SatNav.git
cd SatNav
```

## 2. Install Core

Core contains the SatNav dataset, SatSim, the environment API, and the shared
evaluation framework. It does not depend on PyTorch.

```bash
conda env create -f environments/satnav/conda.yml
conda activate satnav
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the installation:

```bash
python -c "from satnav.core.env import Env; print('SatNav import OK')"
```

## 3. Install optional components

### Data production and visualization

Install map download, trajectory generation, and video dependencies:

```bash
python -m pip install -e '.[applications]'
python -m applications.trajectory_generation.generate --help
```

Install this extra when downloading maps, generating training trajectories,
or exporting videos.

### Classic baselines

Classic baselines additionally require PyTorch. Random and ReferenceFollower
need no weights; Seq2Seq and CMA require trained checkpoints. The following
CUDA 12.1 combination has been validated:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[classic]'
python -m baselines.classic --help
```

For another CUDA version, select matching wheels from the official PyTorch
index. After installation, use [Model Training](../training/README.md) to run
the tiny Seq2Seq/CMA end-to-end example.

## 4. VLM baselines

The four VLM stacks require incompatible PyTorch, Transformers, and
FlashAttention versions. Give each VLM its own Conda environment; do not reuse
Core or another VLM environment.

| Baseline | Python | PyTorch | Guide |
| --- | --- | --- | --- |
| StreamVLN | 3.9 | 2.5.1 | [StreamVLN Baseline](../training/vlm/STREAMVLN.md) |
| NaVILA | 3.10 | 2.3.0 | [NaVILA Baseline](../training/vlm/NAVILA.md) |
| Uni-NaVid | 3.9 | 2.5.1 | [Uni-NaVid Baseline](../training/vlm/UNINAVID.md) |
| OpenFly | 3.10 | 2.3.0 | [OpenFly Baseline](../training/vlm/OPENFLY.md) |

Each VLM environment must install SatNav. Follow its guide in this order:

1. create the baseline-specific Conda environment;
2. install the PyTorch build matching CUDA;
3. install a FlashAttention wheel matching Python, PyTorch, CUDA, and the
   CXX11 ABI;
4. install that baseline's `requirements.txt`;
5. return to the SatNav repository root and install SatNav:

```bash
python -m pip install -e .
```

Here `-e .` refers to the SatNav root, not `baselines/vlm/<name>`. It installs
Core without adding dependencies from other VLMs.

OpenFly provides a bootstrap script:

```bash
bash baselines/vlm/openfly/scripts/bootstrap_env.sh satnav-openfly
```

It creates an isolated environment and installs SatNav automatically. Follow
the individual guides for the other three VLMs.

## 5. Local configuration

Do not write dataset, scene, model, or output paths into public configs. Copy
the local template and edit the ignored file:

```bash
mkdir -p .local
cp local.env.example .local/env.sh
```

Classic and every VLM directory also provide `local.env.example`. Create a
component overlay when using that baseline, for example:

```bash
mkdir -p baselines/vlm/streamvln/.local
cp baselines/vlm/streamvln/local.env.example \
  baselines/vlm/streamvln/.local/env.sh
```

All `.local/env.sh` files are ignored by Git and exist only for machine-local
paths.

## 6. Troubleshooting

### Why do I get `ModuleNotFoundError: satnav`?

From the repository root, install SatNav into the active environment:

```bash
python -m pip install -e .
python -c "import satnav; print(satnav.__file__)"
```

With Conda, verify that `which python` points to the intended environment.

### Why does CUDA or a model operator fail to load?

Python, PyTorch, CUDA, and the FlashAttention wheel must be one compatible
combination. Run `python -m pip check`, then reinstall the exact versions in
the corresponding baseline guide. Never mix multiple VLM stacks in one
environment.

### Why did installing one VLM break another?

The VLMs use different PyTorch, Transformers, and FlashAttention versions.
Create one Conda environment per baseline and remove any mixed environment.

### Why does `rasterio` or `pyproj` fail to install?

Prefer `environments/satnav/conda.yml` so Conda installs the native geospatial
libraries. Avoid mixing system Python, system GDAL, and binary wheels from
different package channels.
