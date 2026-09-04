from __future__ import annotations

from pathlib import Path

import torch

from ultr_detection.checkpoints import load_detector, save_detector
from ultr_detection.config import ModelConfig, Task, WindNormalization
from ultr_detection.model import UpperLevelDetector


def _config() -> ModelConfig:
    zeros = (0.0,) * 12
    ones = (1.0,) * 12
    return ModelConfig(
        task=Task.TROUGH,
        threshold=0.1,
        minimum_gradient=0.0,
        curvature_distance_km=500.0,
        temperature=1.0,
        learnable_threshold=True,
        learnable_distance=True,
        predict_sides=True,
        hidden_dim=16,
        attention_heads=1,
        attention_blocks=1,
        attention_window=7,
        use_wind=True,
        use_cva=False,
        wind=WindNormalization(zeros, ones, zeros, ones),
    )


def test_detector_checkpoint_round_trips_without_trainer_state(tmp_path: Path) -> None:
    source = UpperLevelDetector(_config())
    save_detector(source, tmp_path)
    restored = load_detector(tmp_path)

    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "config.json").is_file()
    assert restored.config.task is Task.TROUGH
    assert restored.training is False
    assert source.state_dict().keys() == restored.state_dict().keys()
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
