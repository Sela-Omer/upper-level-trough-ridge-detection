"""Strict loading of versioned detector artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from ultr_detection.config import ModelConfig
from ultr_detection.model import UpperLevelDetector


def load_detector(directory: str | Path, *, device: str = "cpu") -> UpperLevelDetector:
    """Load a detector and fail on every missing or unexpected tensor."""

    directory = Path(directory)
    config = ModelConfig.from_checkpoint_json(directory / "config.json")
    detector = UpperLevelDetector(config)
    detector.load_state_dict(load_file(directory / "model.safetensors", device=device), strict=True)
    detector.to(device)
    detector.eval()
    return detector


def save_detector(detector: UpperLevelDetector, directory: str | Path) -> None:
    """Save weights without executable pickle or optimizer/trainer state."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in detector.state_dict().items()
    }
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(
        json.dumps(detector.config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
