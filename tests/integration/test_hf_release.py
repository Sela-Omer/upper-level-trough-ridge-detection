from __future__ import annotations

import os

import pytest
import torch

from ultr_detection.assets import resolve_dataset, resolve_models
from ultr_detection.checkpoints import load_detector
from ultr_detection.config import Task
from ultr_detection.data import UpperLevelDataset


@pytest.mark.remote
@pytest.mark.parametrize("task", [Task.TROUGH, Task.RIDGE])
def test_hf_production_inference(task: Task) -> None:
    """Download the published artifacts and execute one real scene."""

    if os.environ.get("ULTR_RUN_HF_TESTS") != "1":
        pytest.skip("Set ULTR_RUN_HF_TESTS=1 to exercise Hugging Face downloads")

    dataset_root = resolve_dataset(
        revision=os.environ.get("ULTR_DATA_REVISION"),
        allow_patterns=("data/training_data.nc", "metadata/samples.parquet"),
    )
    checkpoint_relative = f"production/{task.value}"
    model_root = resolve_models(
        revision=os.environ.get("ULTR_MODEL_REVISION"),
        allow_patterns=(f"{checkpoint_relative}/*",),
        required=(checkpoint_relative,),
    )
    sample = UpperLevelDataset.from_local_release(dataset_root, task=task)[0]
    model = load_detector(model_root / checkpoint_relative)
    arguments = {
        "z500": sample.z500[None],
        "u500": sample.u500[None],
        "v500": sample.v500[None],
        "longitude": sample.longitude,
        "latitude": sample.latitude,
        "month_index": torch.tensor([sample.month_index]),
    }
    with torch.inference_mode():
        output = model(**arguments)

    assert output.axis_probability.shape == (1, 41, 71)
    assert output.side_logits.shape == (1, 1, 41, 71)
    assert torch.isfinite(output.axis_probability).all()
    assert 0.0 <= output.axis_probability.min() <= output.axis_probability.max() <= 1.0
