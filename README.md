# Detection of Upper-Level Troughs and Ridges Using Deep Learning

## Application in the Mediterranean

[![Project Page](https://img.shields.io/badge/Project%20Page-2196F3?style=flat&logo=googlechrome&logoColor=white)](https://sela-omer.github.io/upper-level-trough-ridge-detection/)
[![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/Sela-Omer/upper-level-trough-ridge-detection)
[![Dataset](https://img.shields.io/badge/Dataset-FFD21E?style=flat&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/Omer-Sela/upper-level-trough-ridge-detection-data)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-FFD21E?style=flat&logo=huggingface&logoColor=000)](https://huggingface.co/Omer-Sela/upper-level-trough-ridge-detection-models)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-4051B5?style=flat)](LICENSE)

Official PyTorch implementation of *Detection of Upper-Level Troughs and Ridges Using Deep Learning – Application in the Mediterranean* by Ofir Ariel, Omer Sela, Hadas Saaroni, and Baruch Ziv.

The project provides a reproducible, physics-informed pipeline for training and applying neural detectors to upper-level trough and ridge axes in ERA5 500 hPa geopotential-height and horizontal-wind fields. Data and model artifacts are fetched directly from Hugging Face and cached locally; neither artifact repository needs to be cloned.

## Project resources

| Resource | Link |
| --- | --- |
| Interactive project page | [Explore benchmark scenes and climatology](https://sela-omer.github.io/upper-level-trough-ridge-detection/) |
| Source code | [Sela-Omer/upper-level-trough-ridge-detection](https://github.com/Sela-Omer/upper-level-trough-ridge-detection) |
| Dataset | [Omer-Sela/upper-level-trough-ridge-detection-data](https://huggingface.co/datasets/Omer-Sela/upper-level-trough-ridge-detection-data) |
| Model checkpoints | [Omer-Sela/upper-level-trough-ridge-detection-models](https://huggingface.co/Omer-Sela/upper-level-trough-ridge-detection-models) |

## Installation

Python 3.12 or newer is required. On Ubuntu 22.04, where the system Python is
older, [uv](https://docs.astral.sh/uv/getting-started/installation/) can install
Python 3.12 without modifying the system interpreter:

```bash
uv python install 3.12
uv venv --python 3.12 --seed
source .venv/bin/activate
python -m pip install -e .
```

If Python 3.12 is already installed:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

For an archival environment using the dependency versions validated by the
authors, add `-c constraints/reproducibility.txt` to the installation command.

## Inference

Run the production trough detector on a released scene:

```bash
ultr infer --task trough --sample-id 20180101T0000 --output prediction.json
```

Run ridge detection on an external NetCDF scene:

```bash
ultr infer --task ridge --input scene.nc --month 8 --output prediction.json
```

External scenes must contain two-dimensional `z500`, `u500`, and `v500` variables with `latitude` and `longitude` coordinates. Z500 is geopotential height in metres; wind components are in m s⁻¹. The released models expect the 41 × 71 grid spanning 20–60° N and 20° W–50° E. Output JSON contains each detected spline in pixel and geographic coordinates.

Use `--dataset-revision` and `--model-revision` to pin immutable Hub revisions. `--dataset` and `--models` accept either alternate Hub repository IDs or local directories.

## Training

Train one cross-validation fold with the paper configuration:

```bash
ultr train \
  --config configs/trough.yaml \
  --fold 0 \
  --output artifacts/trough/fold_0 \
  --accelerator gpu
```

The versioned recipe defines the detector architecture, wind normalization,
cross-validation design, and default optimization settings. Command-line
options such as `--batch-size`, `--learning-rate`, and `--max-epochs` can
override optimization defaults. Training does not download an existing model
or checkpoint.

Weights & Biases logging is optional. Enable it explicitly when wanted:

```bash
ultr train \
  --config configs/trough.yaml \
  --fold 0 \
  --output artifacts/trough/fold_0 \
  --accelerator gpu \
  --wandb-project upper-level-trough-ridge
```

Each output directory contains tensor-only `model.safetensors` weights, the
complete `config.json` needed to reload the model, and the resolved training
settings in `training.json`.

## Reproducing the reported evaluation

Each scene is evaluated with the checkpoint from the fold in which that scene was held out:

```bash
ultr benchmark --task trough
ultr benchmark --task ridge
```

| Task | Scenes | F1 | Completeness | Chamfer | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trough | 600 | 0.84 | 0.87 | 1.04 | 2171 / 404 / 408 |
| Ridge | 200 | 0.75 | 0.84 | 1.20 | 562 / 241 / 121 |

See [`docs/reproducibility.md`](docs/reproducibility.md) for metric definitions, revision-pinning guidance, and automated validation commands.

## Repository structure

```text
configs/                 versioned training recipes
src/ultr_detection/      data, model, training, inference, metrics, and CLI
tests/unit/              fast deterministic tests
tests/integration/       checkpoint, training, and Hub integration tests
tests/paper/             complete cross-validation regression tests
```

## Citation

If you use this work, please cite the accompanying paper. Machine-readable software citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

The software is released under the [BSD 3-Clause License](LICENSE). The dataset and model weights are distributed under the licenses stated in their respective Hugging Face repositories.
