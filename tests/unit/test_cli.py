from __future__ import annotations

from ultr_detection.cli import build_parser
from ultr_detection.config import Task


def test_train_parser_exposes_reproducibility_controls() -> None:
    arguments = build_parser().parse_args(
        [
            "train",
            "--dataset",
            "owner/data",
            "--config",
            "configs/trough.yaml",
            "--fold",
            "3",
            "--output",
            "result",
        ]
    )

    assert arguments.dataset == "owner/data"
    assert not hasattr(arguments, "models")
    assert not hasattr(arguments, "task")
    assert arguments.fold == 3
    assert arguments.seed is None
    assert arguments.max_epochs is None
    assert arguments.patience is None


def test_inference_parser_retains_model_selection() -> None:
    arguments = build_parser().parse_args(
        ["infer", "--task", "trough", "--sample-id", "20180101T0000"]
    )

    assert arguments.task is Task.TROUGH
    assert arguments.models
