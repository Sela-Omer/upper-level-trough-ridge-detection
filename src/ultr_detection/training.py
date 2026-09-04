"""PyTorch Lightning training interface for the released detector."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.optim.adamw import AdamW

from ultr_detection.data import UpperLevelBatch
from ultr_detection.losses import DetectionLoss
from ultr_detection.model import DetectorOutput, UpperLevelDetector


class DetectionLightningModule(L.LightningModule):
    """Minimal, reproducible training wrapper used for both feature types."""

    def __init__(
        self,
        model: UpperLevelDetector,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = DetectionLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, batch: UpperLevelBatch) -> DetectorOutput:
        return self.model.forward(
            z500=batch.z500,
            u500=batch.u500,
            v500=batch.v500,
            longitude=batch.longitude,
            latitude=batch.latitude,
            month_index=batch.month_index,
        )

    def compute_loss(self, batch: UpperLevelBatch) -> tuple[torch.Tensor, dict[str, float]]:
        return self.criterion.forward(self(batch), batch.targets)

    def _step(self, batch: UpperLevelBatch, stage: str) -> torch.Tensor:
        loss, terms = self.compute_loss(batch)
        batch_size = len(batch.sample_ids)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            sync_dist=stage != "train",
            batch_size=batch_size,
        )
        for name, value in terms.items():
            self.log(
                f"{stage}/{name}",
                value,
                sync_dist=stage != "train",
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: UpperLevelBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: UpperLevelBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "validation")

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
