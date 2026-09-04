from __future__ import annotations

import numpy as np
import pytest

from ultr_detection.metrics import score_scene


def test_identical_axis_is_a_perfect_match() -> None:
    axis = np.column_stack([np.linspace(0.0, 20.0, 100), np.zeros(100)])
    result = score_scene([axis], [axis.copy()])

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert result.f1 == 1.0
    assert result.completeness == 1.0
    assert result.correctness == 1.0
    assert result.quality == 1.0
    assert result.chamfer == 0.0


def test_close_continuation_is_not_counted_as_a_false_positive() -> None:
    x = np.linspace(0.0, 20.0, 300)
    truth = np.column_stack([x, np.zeros_like(x)])
    predictions = [
        np.column_stack([x[:170], np.full(170, 0.3)]),
        np.column_stack([x[160:], np.full(140, -0.2)]),
        np.column_stack([x, np.full(300, 12.0)]),
    ]
    result = score_scene([truth], predictions, threshold_px=5.0)

    assert (result.tp, result.fp, result.fn, result.n_fragments) == (1, 1, 0, 1)
    assert result.f1 == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("ground_truth", "predictions", "counts"),
    [
        ([], [], (0, 0, 0)),
        ([np.asarray([[0.0, 0.0], [1.0, 0.0]])], [], (0, 0, 1)),
        ([], [np.asarray([[0.0, 0.0], [1.0, 0.0]])], (0, 1, 0)),
    ],
)
def test_empty_scene_counts(
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
    counts: tuple[int, int, int],
) -> None:
    result = score_scene(ground_truth, predictions)
    assert (result.tp, result.fp, result.fn) == counts
