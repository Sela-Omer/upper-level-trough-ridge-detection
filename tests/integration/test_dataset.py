from __future__ import annotations

import numpy as np
import torch

from ultr_detection.assets import AssetRoots
from ultr_detection.config import Task
from ultr_detection.data import AxisCatalog, UpperLevelDataset


def test_fold_selection_matches_published_contract(release_roots: AssetRoots) -> None:
    train = UpperLevelDataset.from_local_release(
        release_roots.dataset, task=Task.TROUGH, fold=3, split="train"
    )
    validation = UpperLevelDataset.from_local_release(
        release_roots.dataset, task=Task.TROUGH, fold=3, split="validation"
    )

    assert len(train) == 540
    assert len(validation) == 60
    assert set(train.sample_ids).isdisjoint(validation.sample_ids)
    assert len(set(train.sample_ids) | set(validation.sample_ids)) == 600


def test_sample_schema_and_coordinates(release_roots: AssetRoots) -> None:
    dataset = UpperLevelDataset.from_local_release(
        release_roots.dataset, task=Task.RIDGE, fold=0, split="validation"
    )
    sample = dataset[0]

    assert sample.z500.shape == (41, 71)
    assert sample.u500.shape == (41, 71)
    assert sample.v500.shape == (41, 71)
    assert sample.axis_target.shape == (41, 71)
    assert sample.latitude.tolist() == list(np.arange(20.0, 61.0))
    assert sample.longitude.tolist() == list(np.arange(-20.0, 51.0))
    assert sample.month_index in range(12)
    assert all(torch.isfinite(value).all() for value in (sample.z500, sample.u500, sample.v500))


def test_axis_catalog_matches_reported_counts(release_roots: AssetRoots) -> None:
    catalog = AxisCatalog.from_local_release(release_roots.dataset)

    assert catalog.count(Task.TROUGH) == 2579
    assert catalog.count(Task.RIDGE) == 683
    first = catalog.for_sample(Task.TROUGH, "20180101T0000")
    assert first
    assert first[0].shape == (300, 2)
