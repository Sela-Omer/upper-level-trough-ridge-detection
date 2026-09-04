from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from ultr_detection.assets import AssetRoots
from ultr_detection.checkpoints import load_detector, save_detector
from ultr_detection.config import ModelConfig, Task
from ultr_detection.data import UpperLevelDataset, collate_samples
from ultr_detection.model import UpperLevelDetector
from ultr_detection.training import DetectionLightningModule


@pytest.mark.parametrize("task", [Task.TROUGH, Task.RIDGE])
def test_one_training_and_validation_batch_runs_end_to_end(
    release_roots: AssetRoots, tmp_path: Path, task: Task
) -> None:
    config = ModelConfig.from_checkpoint_json(
        release_roots.models / f"cross_validation/{task.value}/fold_0/config.json"
    )
    train = UpperLevelDataset.from_local_release(
        release_roots.dataset, task=task, fold=0, split="train"
    )
    validation = UpperLevelDataset.from_local_release(
        release_roots.dataset, task=task, fold=0, split="validation"
    )
    train_loader = DataLoader(Subset(train, [0]), batch_size=1, collate_fn=collate_samples)
    validation_loader = DataLoader(
        Subset(validation, [0]), batch_size=1, collate_fn=collate_samples
    )
    module = DetectionLightningModule(UpperLevelDetector(config), max_epochs=1)
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / ".training",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )

    trainer.fit(module, train_loader, validation_loader)
    saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
    module.load_state_dict(saved["state_dict"], strict=True)
    save_detector(module.model, tmp_path / "published")
    restored = load_detector(tmp_path / "published")

    assert trainer.global_step == 1
    assert len(restored.state_dict()) == 45
