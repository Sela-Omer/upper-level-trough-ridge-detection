"""Curve-level metrics used in the paper's M1 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Curve = np.ndarray


@dataclass(frozen=True, slots=True)
class SceneMetrics:
    f1: float
    precision: float
    recall: float
    completeness: float
    correctness: float
    quality: float
    chamfer: float
    chamfer_pairs: list[float]
    tp: int
    fp: int
    fn: int
    n_fragments: int
    n_gt: int
    n_pred: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid(curves: Sequence[Curve]) -> list[Curve]:
    return [
        np.asarray(curve, dtype=np.float64)
        for curve in curves
        if curve is not None and len(curve) >= 2
    ]


def _chamfer(first: Curve, second: Curve) -> float:
    distances = cdist(first, second, metric="euclidean")
    return 0.5 * (float(distances.min(axis=1).mean()) + float(distances.min(axis=0).mean()))


def _distance_matrix(ground_truth: list[Curve], predictions: list[Curve]) -> np.ndarray:
    if not ground_truth or not predictions:
        return np.full((len(ground_truth), len(predictions)), np.nan)
    return np.asarray(
        [[_chamfer(truth, prediction) for prediction in predictions] for truth in ground_truth]
    )


def _match(distances: np.ndarray, threshold: float) -> tuple[int, int, int, list[tuple[int, int]]]:
    n_gt, n_pred = distances.shape
    if not n_gt or not n_pred:
        return 0, n_pred, n_gt, []
    valid = ~np.isnan(distances) & (distances <= threshold)
    if not valid.any():
        return 0, n_pred, n_gt, []
    cost = np.full((n_gt, n_pred), 1e10, dtype=np.float64)
    cost[valid] = distances[valid] - (threshold + 1.0)
    padded = np.zeros((n_gt + n_pred, n_gt + n_pred), dtype=np.float64)
    padded[:n_gt, :n_pred] = cost
    rows, columns = linear_sum_assignment(padded)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if row < n_gt and column < n_pred and valid[row, column]
    ]
    return (
        len(pairs),
        n_pred - len({column for _, column in pairs}),
        n_gt - len({row for row, _ in pairs}),
        pairs,
    )


def _adjusted_counts(
    distances: np.ndarray, threshold: float
) -> tuple[int, int, int, list[tuple[int, int]], list[int]]:
    tp, _, fn, pairs = _match(distances, threshold)
    matched_gt = {row for row, _ in pairs}
    matched_predictions = {column for _, column in pairs}
    fragments: list[int] = []
    false_positives = 0
    for column in range(distances.shape[1]):
        if column in matched_predictions:
            continue
        close = [
            row
            for row in range(distances.shape[0])
            if not np.isnan(distances[row, column]) and distances[row, column] <= threshold
        ]
        close_matched = [row for row in close if row in matched_gt]
        close_unmatched = [row for row in close if row not in matched_gt]
        if close_matched and not close_unmatched:
            fragments.append(column)
        else:
            false_positives += 1
    return tp, false_positives, fn, pairs, fragments


def _arc_samples(curve: Curve, count: int = 300) -> Curve:
    segments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 0:
        return curve[:1]
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    targets = np.linspace(0.0, total, count)
    output = np.empty((count, 2), dtype=np.float64)
    segment = 0
    for index, target in enumerate(targets):
        while segment < len(segments) - 1 and cumulative[segment + 1] < target:
            segment += 1
        start, end = cumulative[segment], cumulative[segment + 1]
        fraction = 0.0 if end <= start else float((target - start) / (end - start))
        output[index] = (1.0 - fraction) * curve[segment] + fraction * curve[segment + 1]
    return output


def _coverage(curve: Curve, others: list[Curve], threshold: float) -> float:
    if not others:
        return 0.0
    points = _arc_samples(curve)
    nearest = np.full(len(points), np.inf)
    for other in others:
        nearest = np.minimum(nearest, cdist(points, other).min(axis=1))
    return float((nearest <= threshold).mean())


def _mean_gt_coverage(gt: list[Curve], predictions: list[Curve], threshold: float) -> float:
    if not gt and not predictions:
        return 1.0
    if not gt:
        return float("nan")
    return float(np.mean([_coverage(curve, predictions, threshold) for curve in gt]))


def _buffer_lengths(
    gt: list[Curve], predictions: list[Curve], threshold: float
) -> tuple[float, float, float, float]:
    matched_gt = total_gt = matched_prediction = total_prediction = 0.0
    for curve in gt:
        length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        total_gt += length
        matched_gt += _coverage(curve, predictions, threshold) * length
    for curve in predictions:
        length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        total_prediction += length
        matched_prediction += _coverage(curve, gt, threshold) * length
    return matched_gt, total_gt, matched_prediction, total_prediction


def score_scene(
    ground_truth: Sequence[Curve], predictions: Sequence[Curve], *, threshold_px: float = 5.0
) -> SceneMetrics:
    """Score one scene with continuation-aware object and arc metrics."""

    gt, predicted = _valid(ground_truth), _valid(predictions)
    distances = _distance_matrix(gt, predicted)
    tp, fp, fn, pairs, fragments = _adjusted_counts(distances, threshold_px)
    precision = float(tp / (tp + fp)) if tp + fp else float("nan")
    recall = float(tp / (tp + fn)) if tp + fn else float("nan")
    if not gt and not predicted:
        f1 = 1.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    gt_to_predictions = {row: [column] for row, column in pairs}
    matched_gt = set(gt_to_predictions)
    for column in fragments:
        candidates = [
            (float(distances[row, column]), row)
            for row in matched_gt
            if distances[row, column] <= threshold_px
        ]
        if candidates:
            gt_to_predictions[min(candidates)[1]].append(column)
    chamfers = [
        _chamfer(gt[row], np.concatenate([predicted[column] for column in columns]))
        for row, columns in gt_to_predictions.items()
    ]
    chamfer = float(np.mean(chamfers)) if chamfers else float("nan")
    matched_gt_length, total_gt, matched_pred_length, total_pred = _buffer_lengths(
        gt, predicted, threshold_px
    )
    correctness = (
        matched_pred_length / total_pred
        if total_pred > 0
        else (1.0 if total_gt <= 0 else float("nan"))
    )
    false_length = max(0.0, total_pred - matched_pred_length)
    missed_length = max(0.0, total_gt - matched_gt_length)
    denominator = matched_gt_length + false_length + missed_length
    quality = (
        matched_gt_length / denominator
        if denominator > 0
        else (1.0 if total_gt <= 0 and total_pred <= 0 else 0.0)
    )
    return SceneMetrics(
        f1=float(f1),
        precision=precision,
        recall=recall,
        completeness=_mean_gt_coverage(gt, predicted, threshold_px),
        correctness=float(correctness),
        quality=float(quality),
        chamfer=chamfer,
        chamfer_pairs=chamfers,
        tp=tp,
        fp=fp,
        fn=fn,
        n_fragments=len(fragments),
        n_gt=len(gt),
        n_pred=len(predicted),
    )
