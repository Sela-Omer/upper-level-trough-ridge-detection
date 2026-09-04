"""End-to-end out-of-fold evaluation against the released expert axes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ultr_detection.checkpoints import load_detector
from ultr_detection.config import Task
from ultr_detection.data import AxisCatalog, UpperLevelDataset
from ultr_detection.metrics import SceneMetrics, score_scene
from ultr_detection.postprocess import PostprocessConfig, extract_axes

CALM_WIND_MS = 5.0 * 1852.0 / 3600.0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    task: Task
    n_scenes: int
    f1: float
    completeness: float
    chamfer: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["task"] = self.task.value
        return result


def _aggregate(task: Task, rows: list[SceneMetrics]) -> BenchmarkResult:
    tp = sum(row.tp for row in rows)
    fp = sum(row.fp for row in rows)
    fn = sum(row.fn for row in rows)
    # Headline F1 is the mean scene-level score; TP/FP/FN are also reported
    # separately and intentionally do not define a pooled F1.
    f1 = float(np.nanmean([row.f1 for row in rows]))
    completeness = float(np.nanmean([row.completeness for row in rows]))
    chamfer = float(np.nanmean([row.chamfer for row in rows]))
    return BenchmarkResult(task, len(rows), f1, completeness, chamfer, tp, fp, fn)


def evaluate_out_of_fold(
    dataset_root: str | Path,
    task: Task,
    *,
    model_root: str | Path | None = None,
    device: str = "cpu",
    progress: Callable[[int, int], None] | None = None,
) -> tuple[BenchmarkResult, list[SceneMetrics]]:
    """Run every scene through the fold model that did not train on it."""

    dataset_root = Path(dataset_root)
    model_root = Path(model_root) if model_root is not None else dataset_root / "checkpoints"
    catalog = AxisCatalog.from_local_release(dataset_root)
    postprocess = PostprocessConfig.for_task(task)
    rows: list[SceneMetrics] = []
    total = 600 if task is Task.TROUGH else 200
    completed = 0
    for fold in range(10):
        model = load_detector(
            model_root / f"cross_validation/{task.value}/fold_{fold}", device=device
        )
        dataset = UpperLevelDataset.from_local_release(
            dataset_root, task=task, fold=fold, split="validation"
        )
        for position in range(len(dataset)):
            sample = dataset[position]
            with torch.inference_mode():
                output = model(
                    z500=sample.z500[None].to(device),
                    u500=sample.u500[None].to(device),
                    v500=sample.v500[None].to(device),
                    longitude=sample.longitude.to(device),
                    latitude=sample.latitude.to(device),
                    month_index=torch.tensor([sample.month_index], device=device),
                )
            heatmap = output.axis_probability[0].cpu().numpy().copy()
            wind_speed = torch.sqrt(sample.u500**2 + sample.v500**2).numpy()
            heatmap[wind_speed < CALM_WIND_MS] = 0.0
            predictions = extract_axes(heatmap, output.side_logits[0, 0].cpu().numpy(), postprocess)
            rows.append(score_scene(catalog.for_sample(task, sample.sample_id), predictions))
            completed += 1
            if progress is not None:
                progress(completed, total)
    return _aggregate(task, rows), rows
