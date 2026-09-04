from __future__ import annotations

from dataclasses import replace

import torch

from ultr_detection.config import Task
from ultr_detection.data import UpperLevelSample, collate_samples


def _sample(sample_id: str, value: float) -> UpperLevelSample:
    grid = torch.full((2, 3), value)
    return UpperLevelSample(
        sample_id=sample_id,
        task=Task.TROUGH,
        month_index=int(value),
        latitude=torch.tensor([40.0, 39.0]),
        longitude=torch.tensor([10.0, 11.0, 12.0]),
        z500=grid,
        u500=grid + 1,
        v500=grid + 2,
        axis_target=grid + 3,
        supervision_mask=grid + 4,
        side_label=grid + 5,
        signed_split=grid + 6,
    )


def test_collate_samples_preserves_shared_coordinates_and_stacks_fields() -> None:
    batch = collate_samples([_sample("a", 0), _sample("b", 1)])

    assert batch.sample_ids == ("a", "b")
    assert batch.task is Task.TROUGH
    assert batch.z500.shape == (2, 2, 3)
    assert batch.month_index.tolist() == [0, 1]
    torch.testing.assert_close(batch.latitude, torch.tensor([40.0, 39.0]))
    assert set(batch.targets) == {"axis_target", "supervision_mask", "side_label"}


def test_collate_samples_rejects_mixed_tasks() -> None:
    samples = [_sample("a", 0), _sample("b", 1)]
    samples[1] = replace(samples[1], task=Task.RIDGE)
    try:
        collate_samples(samples)
    except ValueError as error:
        assert "same task" in str(error)
    else:
        raise AssertionError("mixed-task batch should fail")
