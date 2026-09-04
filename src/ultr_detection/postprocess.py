"""Locked M1 conversion from dense predictions to spline axes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from skimage.morphology import dilation, disk, h_maxima
from skimage.segmentation import watershed

from ultr_detection.config import Task


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Frozen paper settings for the M1 axis extractor."""

    h_watershed: float
    mask_min: float
    min_blob_pixels: int
    blob_zc_dilate_radius: int = 1
    extra_peak_h: float | None = None
    extra_used_px: float = 0.0
    merge_endpoints_px: float = 2.0
    extra_max_curves: int = 3

    @classmethod
    def for_task(cls, task: Task) -> PostprocessConfig:
        if task is Task.TROUGH:
            return cls(0.4, 0.3, 10, extra_peak_h=0.3)
        return cls(0.45, 0.25, 15, extra_peak_h=0.25)


def _norm01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    maximum = float(array.max()) if array.size else 0.0
    return array / (maximum + 1e-8) if maximum > 0 else np.zeros_like(array)


def _watershed_blobs(heatmap: np.ndarray, config: PostprocessConfig) -> list[np.ndarray]:
    if heatmap.size == 0 or heatmap.max() <= 0:
        return []
    normalized = _norm01(heatmap)
    coordinates = peak_local_max(
        h_maxima(normalized, config.h_watershed), min_distance=1, exclude_border=False
    )
    if len(coordinates) == 0:
        return []
    markers = np.zeros_like(normalized, dtype=np.int32)
    for marker, (row, column) in enumerate(coordinates, 1):
        markers[row, column] = marker
    labels = watershed(-normalized, markers=markers, mask=normalized > config.mask_min)
    return [
        labels == region
        for region in range(1, int(labels.max()) + 1)
        if int((labels == region).sum()) >= config.min_blob_pixels
    ]


def _true_runs(mask: np.ndarray, *, closed: bool = False) -> list[slice | np.ndarray]:
    if mask.size == 0 or not mask.any():
        return []
    runs: list[slice] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append(slice(start, index))
            start = None
    if start is not None:
        runs.append(slice(start, len(mask)))
    if closed and len(runs) >= 2 and mask[0] and mask[-1]:
        wrapped = np.concatenate([np.arange(runs[-1].start, len(mask)), np.arange(0, runs[0].stop)])
        return [wrapped, *runs[1:-1]]
    return list(runs)


def _is_owned(
    x: float,
    y: float,
    side: np.ndarray,
    owner: np.ndarray,
    search: np.ndarray,
) -> bool:
    height, width = side.shape
    column, row = round(x), round(y)
    if not (0 <= column < width and 0 <= row < height):
        return False
    if not owner[row, column]:
        y0, y1 = max(0, row - 1), min(height, row + 2)
        x0, x1 = max(0, column - 1), min(width, column + 2)
        if not owner[y0:y1, x0:x1].any():
            return False
    signs = [
        float(side[row + dr, column + dc])
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
        if 0 <= row + dr < height and 0 <= column + dc < width and search[row + dr, column + dc]
    ]
    return any(value > 1e-6 for value in signs) and any(value < -1e-6 for value in signs)


def _owned_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    side: np.ndarray,
    owner: np.ndarray,
    search: np.ndarray,
    other: np.ndarray | None,
) -> np.ndarray:
    height, width = side.shape
    columns = np.clip(np.round(xs).astype(int), 0, width - 1)
    rows = np.clip(np.round(ys).astype(int), 0, height - 1)
    owned = np.asarray(
        [
            search[row, column]
            and (other is None or not other[row, column])
            and _is_owned(x, y, side, owner, search)
            for x, y, column, row in zip(xs, ys, columns, rows, strict=True)
        ],
        dtype=bool,
    )
    # Bridge search-halo gaps bounded by owned points.
    index = 0
    while index < len(owned):
        if owned[index]:
            index += 1
            continue
        start = index
        while index < len(owned) and not owned[index]:
            index += 1
        if start == 0 or index >= len(owned) or not (owned[start - 1] and owned[index]):
            continue
        okay = True
        for point in range(start, index):
            column, row = round(xs[point]), round(ys[point])
            if (
                not (0 <= row < height and 0 <= column < width)
                or not search[row, column]
                or owner[row, column]
                or (other is not None and other[row, column])
            ):
                okay = False
                break
        if okay:
            owned[start:index] = True
    # Preserve up to three terminal contour points caused by rounding jitter.
    if owned.any():
        true_indices = np.flatnonzero(owned)
        for start, stop in ((0, int(true_indices[0])), (int(true_indices[-1]) + 1, len(owned))):
            if 0 < stop - start <= 3:
                okay = True
                for point in range(start, stop):
                    column, row = round(xs[point]), round(ys[point])
                    if (
                        not (0 <= row < height and 0 <= column < width)
                        or not search[row, column]
                        or (other is not None and other[row, column])
                    ):
                        okay = False
                        break
                if okay:
                    owned[start:stop] = True
    return owned


def _zero_crossing_segments(
    side: np.ndarray,
    search: np.ndarray,
    owner: np.ndarray,
    other: np.ndarray | None,
    minimum: int,
    *,
    all_segments: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    field = np.asarray(side, dtype=np.float64).copy()
    field[~search] = np.nan
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for contour in find_contours(field, level=0.0):
        ys, xs = contour[:, 0], contour[:, 1]
        closed = bool(len(xs) > 2 and np.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) < 1.5)
        if all_segments and closed:
            xs, ys = xs[:-1], ys[:-1]
        owned = _owned_mask(xs, ys, side, owner, search, other)
        if all_segments and closed and owned.mean() >= 0.8:
            continue
        runs = _true_runs(owned, closed=closed if all_segments else False)
        if not all_segments and runs:
            runs = [max(runs, key=lambda run: len(np.arange(len(xs))[run]))]
        for run in runs:
            x_run, y_run = xs[run], ys[run]
            if len(x_run) >= minimum:
                output.append((x_run.astype(np.float64), y_run.astype(np.float64)))
    output.sort(key=lambda pair: len(pair[0]), reverse=True)
    return output


def _clip(
    xs: np.ndarray, ys: np.ndarray, mask: np.ndarray, minimum: int
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = mask.shape
    columns = np.clip(np.round(xs).astype(int), 0, width - 1)
    rows = np.clip(np.round(ys).astype(int), 0, height - 1)
    inside = mask[rows, columns].copy()
    index = 0
    while index < len(inside):
        if inside[index]:
            index += 1
            continue
        start = index
        while index < len(inside) and not inside[index]:
            index += 1
        if index - start <= 2 and start > 0 and index < len(inside):
            inside[start:index] = True
    runs = _true_runs(inside)
    if not runs:
        return None
    run = max(runs, key=lambda value: value.stop - value.start)  # type: ignore[union-attr]
    if run.stop - run.start < minimum:  # type: ignore[union-attr]
        return None
    return xs[run].astype(np.float64), ys[run].astype(np.float64)


def _fit(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray, minimum: int) -> np.ndarray | None:
    clipped = _clip(xs, ys, mask, minimum)
    if clipped is None:
        return None
    points = np.column_stack(clipped)
    points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
    degree = min(3, max(1, len(points) - 1))
    if len(points) < degree + 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline, _ = splprep(
                [points[:, 0], points[:, 1]], s=0.1 * len(points), k=degree, per=False
            )
            evaluated = splev(np.linspace(0.0, 1.0, 300), spline)
    except (ValueError, RuntimeError, TypeError):
        return None
    if not (np.isfinite(evaluated[0]).all() and np.isfinite(evaluated[1]).all()):
        return None
    final = _clip(np.asarray(evaluated[0]), np.asarray(evaluated[1]), mask, minimum)
    return None if final is None else np.column_stack(final)


def _too_close(first: np.ndarray, second: np.ndarray) -> bool:
    distances = cdist(first.astype(float), second.astype(float))
    nearest = distances.min(axis=1) if len(first) <= len(second) else distances.min(axis=0)
    return float((nearest < 2.0).mean()) >= 0.5


def _blob_curves(
    heatmap: np.ndarray,
    side: np.ndarray,
    mask: np.ndarray,
    other: np.ndarray | None,
    config: PostprocessConfig,
) -> list[np.ndarray | None]:
    minimum = 4
    search = dilation(mask.astype(bool), disk(config.blob_zc_dilate_radius))
    primary_segments = _zero_crossing_segments(
        side, search, mask, other, minimum, all_segments=False
    )
    primary = _fit(*primary_segments[0], mask, minimum) if primary_segments else None
    if config.extra_peak_h is None or primary is None:
        return [primary]
    height, width = mask.shape
    normalized = _norm01(heatmap)
    extras: list[np.ndarray] = []
    for xs, ys in _zero_crossing_segments(side, search, mask, other, minimum, all_segments=True):
        unused = (
            cdist(np.column_stack([xs, ys]), primary).min(axis=1) > config.extra_used_px
            if config.extra_used_px > 0
            else np.ones(len(xs), dtype=bool)
        )
        for run in _true_runs(unused):
            x_run, y_run = xs[run], ys[run]
            columns = np.clip(np.round(x_run).astype(int), 0, width - 1)
            rows = np.clip(np.round(y_run).astype(int), 0, height - 1)
            for high_run in _true_runs(normalized[rows, columns] >= config.mask_min):
                if len(x_run[high_run]) < minimum:
                    continue
                curve = _fit(x_run[high_run], y_run[high_run], mask, minimum)
                if curve is not None and not _too_close(curve, primary):
                    extras.append(curve)
    kept: list[np.ndarray] = []
    for curve in sorted(extras, key=len, reverse=True):
        if not any(_too_close(curve, prior) for prior in kept):
            kept.append(curve)
    return [primary, *kept[: max(0, config.extra_max_curves - 1)]]


def _merge(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    best_gap = float("inf")
    best: np.ndarray | None = None
    for reverse_first in (False, True):
        one = first[::-1] if reverse_first else first
        for reverse_second in (False, True):
            two = second[::-1] if reverse_second else second
            for flip in (False, True):
                start, end = (one, two) if not flip else (two, one)
                gap = float(cdist(start[[-1]], end[[0]])[0, 0])
                if gap < best_gap:
                    best_gap, best = gap, np.vstack([start, end])
    assert best is not None
    fitted = _fit(best[:, 0], best[:, 1], mask, 4)
    return fitted if fitted is not None and len(fitted) >= 2 else best


def extract_axes(
    heatmap: np.ndarray, side_logits: np.ndarray, config: PostprocessConfig
) -> list[np.ndarray]:
    """Extract ordered ``(x, y)`` pixel-coordinate splines from one scene."""

    side = np.asarray(side_logits, dtype=np.float64).squeeze()
    blobs = _watershed_blobs(np.asarray(heatmap), config)
    output: list[np.ndarray] = []
    for index, mask in enumerate(blobs):
        other = np.zeros_like(mask, dtype=bool)
        for other_index, other_mask in enumerate(blobs):
            if other_index != index:
                other |= other_mask
        curves = _blob_curves(heatmap, side, mask, other if len(blobs) > 1 else None, config)
        blob_output: list[np.ndarray] = []
        for curve in curves:
            if curve is None or len(curve) < 2:
                continue
            merged = False
            if config.extra_peak_h is not None and config.merge_endpoints_px > 0:
                for existing_index, existing in enumerate(blob_output):
                    endpoint_gap = float(cdist(curve[[0, -1]], existing[[0, -1]]).min())
                    if endpoint_gap <= config.merge_endpoints_px:
                        blob_output[existing_index] = _merge(existing, curve, mask)  # type: ignore[assignment]
                        merged = True
                        break
            if not merged:
                blob_output.append(np.asarray(curve, dtype=float))
        output.extend(blob_output)
    return output
