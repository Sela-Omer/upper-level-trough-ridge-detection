from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultr_detection.config import ModelConfig


def test_config_rejects_an_unsupported_architecture(
    tmp_path: Path,
) -> None:
    source = {
        "axis_attention_mode": "cross",
        "feature_mode": "trough",
        "use_wind_channel": True,
        "use_cva_channel": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="local self-attention"):
        ModelConfig.from_checkpoint_json(path)
