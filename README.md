# Upper-level trough and ridge detection

PyTorch implementation of the local-attention detector described in
*Detection of upper-level troughs and ridges using deep learning – application
in the Mediterranean* by Ofir Ariel, Omer Sela, Hadas Saaroni, and Baruch Ziv
([paper](PAPER_URL)).

The package provides a reproducible command-line workflow for training,
inference, and evaluation. Data and model artifacts are downloaded directly
from Hugging Face and cached locally; neither repository needs to be cloned.

## Resources

| Resource | Repository |
| --- | --- |
| Interactive project website | [Upper-Level Atlas](https://sela-omer.github.io/upper-level-trough-ridge-detection/) |
| Source code | [Sela-Omer/upper-level-trough-ridge-detection](https://github.com/Sela-Omer/upper-level-trough-ridge-detection) |
| Dataset | [Omer-Sela/upper-level-trough-ridge-detection-data](https://huggingface.co/datasets/Omer-Sela/upper-level-trough-ridge-detection-data) |
| Model weights | [Omer-Sela/upper-level-trough-ridge-detection-models](https://huggingface.co/Omer-Sela/upper-level-trough-ridge-detection-models) |

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Inference

Run the all-data trough model on a released sample:

```bash
ultr infer --task trough --sample-id 20180101T0000 --output prediction.json
```

Run ridge detection on a new scene:

```bash
ultr infer --task ridge --input scene.nc --month 8 --output prediction.json
```

External NetCDF scenes must contain two-dimensional `z500`, `u500`, and
`v500` variables on `latitude` and `longitude` coordinates. Z500 is
geopotential height in metres; wind components are in m s⁻¹. The current
models expect the 41 × 71 grid spanning 20–60° N and 20° W–50° E. Output JSON
contains each detected spline in both pixel and geographic coordinates.

Use `--dataset-revision` and `--model-revision` to pin Hub commits or release
tags. `--dataset` and `--models` accept alternate Hub repository IDs or local
directories.

## Training

Train one cross-validation fold with the paper configuration:

```bash
ultr train \
  --task trough \
  --fold 0 \
  --output artifacts/trough/fold_0 \
  --accelerator gpu \
  --wandb-project upper-level-trough-ridge
```

W&B logging is optional. The output directory contains tensor-only
`model.safetensors` weights and the complete `config.json` needed for loading
the model. Task recipes are versioned in [`configs/`](configs/).

## Reproducing the reported evaluation

Each scene is evaluated with the model from the fold in which that scene was
held out:

```bash
ultr benchmark --task trough
ultr benchmark --task ridge
```

Expected results are:

| Task | Scenes | F1 | Completeness | Chamfer | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trough | 600 | 0.84 | 0.87 | 1.04 | 2171 / 404 / 408 |
| Ridge | 200 | 0.75 | 0.84 | 1.20 | 562 / 241 / 121 |

See [`docs/reproducibility.md`](docs/reproducibility.md) for the evaluation
definition, version-pinning guidance, and automated validation commands.

## Repository layout

```text
configs/                 training recipes for troughs and ridges
src/ultr_detection/      data, model, training, inference, metrics, and CLI
tests/unit/              fast deterministic tests
tests/integration/       checkpoint, training, and Hub integration tests
tests/paper/             complete cross-validation regression
```

## Citation

Please cite the accompanying [paper](PAPER_URL). Machine-readable software
citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

The software is released under the BSD 3-Clause License. The associated data
and model weights are licensed separately in their Hugging Face repositories.
