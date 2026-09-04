# Reproducibility

## Versioned inputs

The dataset and model repositories on Hugging Face are independent versioned
artifacts. For an archival run, pass an immutable commit hash or release tag to
both commands:

```bash
ultr benchmark \
  --dataset-revision <dataset-revision> \
  --model-revision <model-revision> \
  --task trough
```

The adjacent `config.json` is authoritative for each checkpoint's architecture,
preprocessing, task, and monthly wind normalization. Checkpoints are loaded
strictly: missing or unexpected tensors raise an error.

## Cross-validation evaluation

The benchmark consists of 600 trough scenes and 200 ridge scenes. Both tasks
use ten deterministic folds generated with seed 12345. For fold *k*, rows with
`cv_fold == k` are evaluated using `cross_validation/<task>/fold_<k>`; all
other rows of the same task form that fold's training set.

At evaluation time, heatmap values at grid cells with 500 hPa wind speed below
5 knots are set to zero. Task-specific M1 parameters then convert each heatmap
and side field into spline axes. Predicted and expert curves are matched with a
five-grid-cell Chamfer threshold, including the continuation-fragment rule
defined in the paper. F1, completeness, and Chamfer distance are averaged over
scenes; TP, FP, and FN are pooled counts.

The expected results are:

| Task | Scenes | F1 | Completeness | Chamfer | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trough | 600 | 0.84 | 0.87 | 1.04 | 2171 | 404 | 408 |
| Ridge | 200 | 0.75 | 0.84 | 1.20 | 562 | 241 | 121 |

## Training configurations

The tracked YAML files contain the configurations used for the released
training workflow. The trough configuration uses batch size 16; the ridge
configuration uses batch size 8. Random seeds and deterministic data folds are
fixed, although floating-point reductions and stochastic GPU kernels can vary
across hardware and library versions.

## Validation commands

Install development dependencies and run the portable test suite:

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy src/ultr_detection
pytest -m "not slow and not remote" -q
```

Exercise direct downloads from Hugging Face:

```bash
ULTR_RUN_HF_TESTS=1 pytest tests/integration/test_hf_release.py -q
```

Run the complete 800-scene regression:

```bash
ULTR_RUN_PAPER_TESTS=1 pytest tests/paper -q
```

The optional environment variables `ULTR_DATA_REVISION` and
`ULTR_MODEL_REVISION` pin the two Hub snapshots used by these tests.
