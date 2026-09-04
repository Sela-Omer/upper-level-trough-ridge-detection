from __future__ import annotations

from ultr_detection.cli import build_parser
from ultr_detection.config import Task


def test_train_parser_exposes_reproducibility_controls() -> None:
    arguments = build_parser().parse_args(
        [
            "train",
            "--dataset",
            "owner/data",
            "--models",
            "owner/models",
            "--task",
            "trough",
            "--fold",
            "3",
            "--output",
            "result",
        ]
    )

    assert arguments.task is Task.TROUGH
    assert arguments.dataset == "owner/data"
    assert arguments.models == "owner/models"
    assert arguments.fold == 3
    assert arguments.seed == 42
    assert arguments.max_epochs == 100
    assert arguments.patience == 15
