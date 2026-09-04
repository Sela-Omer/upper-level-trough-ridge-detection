from __future__ import annotations

import torch
from torch.nn import functional as F

from ultr_detection.losses import DetectionLoss
from ultr_detection.model import DetectorOutput


def test_detection_loss_combines_axis_side_and_sparsity_terms() -> None:
    axis = torch.tensor([[[0.25, 0.75], [0.60, 0.40]]])
    feature_mask = torch.tensor([[[0.10, 0.20], [0.30, 0.40]]])
    side_logits = torch.tensor([[[[0.0, 1.0], [-1.0, 2.0]]]])
    output = DetectorOutput(
        feature_mask=feature_mask,
        curvature=torch.zeros_like(axis),
        gradient_magnitude=torch.zeros_like(axis),
        axis_probability=axis,
        side_logits=side_logits,
    )
    target = {
        "axis_target": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        "side_label": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        "supervision_mask": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
    }

    actual, terms = DetectionLoss()(output, target)
    axis_loss = F.binary_cross_entropy(axis, target["axis_target"])
    raw_side = F.binary_cross_entropy_with_logits(
        side_logits.reshape(-1), target["side_label"].reshape(-1), reduction="none"
    )
    side_loss = (raw_side * target["supervision_mask"].reshape(-1)).sum() / 2.0
    expected = 2.0 * axis_loss + 2.0 * side_loss + 0.01 * feature_mask.mean()

    torch.testing.assert_close(actual, expected)
    assert terms["trough_axis"] == axis_loss.item()
    assert terms["trough_side"] == side_loss.item()
    assert terms["sparsity"] == feature_mask.mean().item()
