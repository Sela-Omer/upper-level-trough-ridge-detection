"""Dataset access for the versioned Hugging Face release."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ultr_detection.config import Task

Split = Literal["train", "validation", "all"]
TASK_CODE = {Task.TROUGH: 0, Task.RIDGE: 1}


class AxisCatalog:
    """Ground-truth expert spline axes indexed by task and sample identifier."""

    def __init__(self, path: str | Path) -> None:
        rows = pq.read_table(path).to_pylist()
        self._axes: dict[tuple[Task, str], list[np.ndarray]] = {}
        for row in rows:
            key = (Task(row["task"]), str(row["sample_id"]))
            curve = np.column_stack(
                [
                    np.asarray(row["curve_x"], dtype=np.float64),
                    np.asarray(row["curve_y"], dtype=np.float64),
                ]
            )
            self._axes.setdefault(key, []).append(curve)

    @classmethod
    def from_local_release(cls, root: str | Path) -> AxisCatalog:
        return cls(Path(root) / "metadata/axes.parquet")

    def for_sample(self, task: Task, sample_id: str) -> tuple[np.ndarray, ...]:
        return tuple(self._axes.get((task, sample_id), ()))

    def count(self, task: Task) -> int:
        return sum(len(axes) for (axis_task, _), axes in self._axes.items() if axis_task is task)


@dataclass(frozen=True, slots=True)
class UpperLevelSample:
    """One model-ready labelled atmospheric scene."""

    sample_id: str
    task: Task
    month_index: int
    latitude: torch.Tensor
    longitude: torch.Tensor
    z500: torch.Tensor
    u500: torch.Tensor
    v500: torch.Tensor
    axis_target: torch.Tensor
    supervision_mask: torch.Tensor
    side_label: torch.Tensor
    signed_split: torch.Tensor


@dataclass(slots=True)
class UpperLevelBatch:
    """A homogeneous task batch accepted by the detector and loss."""

    sample_ids: tuple[str, ...]
    task: Task
    month_index: torch.Tensor
    latitude: torch.Tensor
    longitude: torch.Tensor
    z500: torch.Tensor
    u500: torch.Tensor
    v500: torch.Tensor
    targets: dict[str, torch.Tensor]


def collate_samples(samples: Sequence[UpperLevelSample]) -> UpperLevelBatch:
    """Stack samples while retaining coordinates as a single shared grid."""

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    task = samples[0].task
    if any(sample.task is not task for sample in samples):
        raise ValueError("All samples in a batch must have the same task")
    latitude = samples[0].latitude
    longitude = samples[0].longitude
    if any(
        not torch.equal(sample.latitude, latitude) or not torch.equal(sample.longitude, longitude)
        for sample in samples[1:]
    ):
        raise ValueError("All samples in a batch must use the same coordinate grid")

    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(sample, name) for sample in samples])

    return UpperLevelBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        task=task,
        month_index=torch.tensor([sample.month_index for sample in samples], dtype=torch.long),
        latitude=latitude,
        longitude=longitude,
        z500=stack("z500"),
        u500=stack("u500"),
        v500=stack("v500"),
        targets={
            "axis_target": stack("axis_target"),
            "supervision_mask": stack("supervision_mask"),
            "side_label": stack("side_label"),
        },
    )


class UpperLevelDataset(Dataset[UpperLevelSample]):
    """In-memory view of a task and cross-validation split.

    The complete release is small enough to load once, which avoids sharing an
    open HDF5 handle between PyTorch DataLoader workers.
    """

    def __init__(
        self,
        data_path: Path,
        metadata_path: Path,
        *,
        task: Task,
        fold: int | None = None,
        split: Split = "all",
    ) -> None:
        if split != "all" and fold not in range(10):
            raise ValueError("fold must be in [0, 9] for train or validation splits")
        metadata = pq.read_table(metadata_path).to_pylist()
        selected = [row for row in metadata if row["task"] == task.value]
        if split == "train":
            selected = [row for row in selected if row["cv_fold"] != fold]
        elif split == "validation":
            selected = [row for row in selected if row["cv_fold"] == fold]
        elif split != "all":
            raise ValueError(f"Unknown split: {split}")
        self.task = task
        self._indices = np.asarray([row["sample_index"] for row in selected], dtype=np.int64)
        self.sample_ids = tuple(str(row["sample_id"]) for row in selected)

        with netCDF4.Dataset(data_path) as source:
            task_values = np.asarray(source["task"][:], dtype=np.int8)
            if not np.all(task_values[self._indices] == TASK_CODE[task]):
                raise ValueError("Metadata and NetCDF task coordinates disagree")
            self._valid_time = np.asarray(source["valid_time"][:], dtype=np.int64)
            self._latitude = np.asarray(source["latitude"][:], dtype=np.float32)
            self._longitude = np.asarray(source["longitude"][:], dtype=np.float32)
            names = (
                "z500",
                "u500",
                "v500",
                "axis_target",
                "supervision_mask_r5",
                "side_label_r5",
                "signed_split_r5",
            )
            self._arrays = {name: np.asarray(source[name][:], dtype=np.float32) for name in names}

    @classmethod
    def from_local_release(
        cls,
        root: str | Path,
        *,
        task: Task,
        fold: int | None = None,
        split: Split = "all",
    ) -> UpperLevelDataset:
        root = Path(root)
        return cls(
            root / "data/training_data.nc",
            root / "metadata/samples.parquet",
            task=task,
            fold=fold,
            split=split,
        )

    def __len__(self) -> int:
        return len(self._indices)

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    def __getitem__(self, position: int) -> UpperLevelSample:
        index = int(self._indices[position])
        timestamp = np.datetime64(int(self._valid_time[index]), "s")
        month_index = int(str(timestamp)[5:7]) - 1
        return UpperLevelSample(
            sample_id=self.sample_ids[position],
            task=self.task,
            month_index=month_index,
            latitude=self._tensor(self._latitude),
            longitude=self._tensor(self._longitude),
            z500=self._tensor(self._arrays["z500"][index]),
            u500=self._tensor(self._arrays["u500"][index]),
            v500=self._tensor(self._arrays["v500"][index]),
            axis_target=self._tensor(self._arrays["axis_target"][index]),
            supervision_mask=self._tensor(self._arrays["supervision_mask_r5"][index]),
            side_label=self._tensor(self._arrays["side_label_r5"][index]),
            signed_split=self._tensor(self._arrays["signed_split_r5"][index]),
        )
