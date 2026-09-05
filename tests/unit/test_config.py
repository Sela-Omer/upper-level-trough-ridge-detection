from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultr_detection.config import ModelConfig, Task, TrainingRecipe

REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(("task", "batch_size"), [(Task.TROUGH, 16), (Task.RIDGE, 8)])
def test_public_training_recipes_are_complete(task: Task, batch_size: int) -> None:
    recipe = TrainingRecipe.from_yaml(REPOSITORY_ROOT / "configs" / f"{task.value}.yaml")

    assert recipe.schema_version == 1
    assert recipe.task is task
    assert recipe.model.task is task
    assert recipe.cross_validation.folds == 10
    assert recipe.cross_validation.fold_seed == 12345
    assert recipe.training.seed == 42
    assert recipe.training.batch_size == batch_size
    assert recipe.training.learning_rate == 3e-4
    assert recipe.training.weight_decay == 1e-5
    assert len(recipe.model.wind.u_monthly_mean) == 12


def test_training_recipe_rejects_task_mismatch(tmp_path: Path) -> None:
    source = (REPOSITORY_ROOT / "configs/trough.yaml").read_text(encoding="utf-8")
    path = tmp_path / "mismatch.yaml"
    path.write_text(source.replace("task: trough", "task: ridge", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Recipe task must match"):
        TrainingRecipe.from_yaml(path)


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
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="local self-attention"):
        ModelConfig.from_checkpoint_json(path)
