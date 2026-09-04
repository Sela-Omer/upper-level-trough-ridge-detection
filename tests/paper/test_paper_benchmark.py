from __future__ import annotations

import pytest

from ultr_detection.assets import AssetRoots
from ultr_detection.config import Task
from ultr_detection.evaluation import evaluate_out_of_fold


@pytest.mark.paper
@pytest.mark.slow
@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (Task.TROUGH, (600, 2171, 404, 408, 0.84, 0.87, 1.04)),
        (Task.RIDGE, (200, 562, 241, 121, 0.75, 0.84, 1.20)),
    ],
)
def test_released_models_reproduce_paper_headline(
    release_roots: AssetRoots,
    task: Task,
    expected: tuple[int, int, int, int, float, float, float],
) -> None:
    result, _ = evaluate_out_of_fold(release_roots.dataset, task, model_root=release_roots.models)
    scenes, tp, fp, fn, f1, completeness, chamfer = expected

    assert (result.n_scenes, result.tp, result.fp, result.fn) == (scenes, tp, fp, fn)
    assert round(result.f1, 2) == f1
    assert round(result.completeness, 2) == completeness
    assert round(result.chamfer, 2) == chamfer
