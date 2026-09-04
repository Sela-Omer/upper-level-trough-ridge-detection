"""Training objective used by the selected paper models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ultr_detection.model import DetectorOutput


class DetectionLoss(nn.Module):
    """Axis, side-classification, and curvature-mask sparsity objective.

    The public training dataset contains axis and side targets. Consequently,
    the selected paper recipe activates axis BCE, masked side BCE, and the
    weak sparsity regularizer. Optional auxiliary objectives are outside this
    interface.
    """

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self.weights = {
            "axis": 2.0,
            "side": 2.0,
            "sparsity": 0.01,
            **(weights or {}),
        }

    def forward(
        self, output: DetectorOutput, target: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        probability = output.axis_probability
        losses: dict[str, torch.Tensor] = {}
        if probability.abs().sum() > 1e-6:
            losses["trough_axis"] = F.binary_cross_entropy(
                probability.clamp(1e-7, 1.0 - 1e-7),
                target["axis_target"].float(),
                reduction="mean",
            )

        logits = output.side_logits.reshape(-1)
        side_target = target["side_label"].float().reshape(-1)
        side_mask = target["supervision_mask"].float().reshape(-1)
        raw_side = F.binary_cross_entropy_with_logits(logits, side_target, reduction="none")
        losses["trough_side"] = (raw_side * side_mask).sum() / (side_mask.sum() + 1e-7)
        losses["sparsity"] = output.feature_mask.mean()

        total = self.weights["sparsity"] * losses["sparsity"]
        if "trough_axis" in losses:
            total = total + self.weights["axis"] * losses["trough_axis"]
        total = total + self.weights["side"] * losses["trough_side"]
        return total, {name: value.item() for name, value in losses.items()}
