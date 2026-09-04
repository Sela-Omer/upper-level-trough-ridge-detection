from __future__ import annotations

import torch

from ultr_detection.assets import AssetRoots
from ultr_detection.config import ModelConfig, Task
from ultr_detection.data import UpperLevelDataset, collate_samples
from ultr_detection.losses import DetectionLoss
from ultr_detection.model import UpperLevelDetector
from ultr_detection.training import DetectionLightningModule


def test_published_checkpoint_config_is_compact_and_complete(
    release_roots: AssetRoots,
) -> None:
    path = release_roots.models / "production/trough/config.json"
    config = ModelConfig.from_checkpoint_json(path)

    assert config.task is Task.TROUGH
    assert config.hidden_dim == 64
    assert config.attention_window == 7
    assert config.attention_heads == 1
    assert config.attention_blocks == 1
    assert config.use_wind is True
    assert config.use_cva is False
    assert len(config.wind.u_monthly_mean) == 12
    assert set(config.to_dict()) == {
        "task",
        "threshold",
        "minimum_gradient",
        "curvature_distance_km",
        "temperature",
        "learnable_threshold",
        "learnable_distance",
        "predict_sides",
        "hidden_dim",
        "attention_heads",
        "attention_blocks",
        "attention_window",
        "use_wind",
        "use_cva",
        "wind",
    }


def test_training_module_uses_configured_objective_and_optimizer(
    release_roots: AssetRoots,
) -> None:
    config = ModelConfig.from_checkpoint_json(
        release_roots.models / "cross_validation/trough/fold_0/config.json"
    )
    model = UpperLevelDetector(config)
    module = DetectionLightningModule(model, learning_rate=3e-4, weight_decay=1e-5, max_epochs=100)
    dataset = UpperLevelDataset.from_local_release(release_roots.dataset, task=config.task)
    batch = collate_samples([dataset[0], dataset[1]])

    actual, terms = module.compute_loss(batch)
    output = model(
        z500=batch.z500,
        u500=batch.u500,
        v500=batch.v500,
        longitude=batch.longitude,
        latitude=batch.latitude,
        month_index=batch.month_index,
    )
    expected, expected_terms = DetectionLoss()(output, batch.targets)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert terms == expected_terms
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-5
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 100
    assert scheduler.eta_min == 1e-6
