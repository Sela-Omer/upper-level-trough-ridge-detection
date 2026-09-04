"""High-level inference utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import netCDF4
import numpy as np
import torch

from ultr_detection.config import Task
from ultr_detection.data import UpperLevelSample
from ultr_detection.evaluation import CALM_WIND_MS
from ultr_detection.model import UpperLevelDetector
from ultr_detection.postprocess import PostprocessConfig, extract_axes


@dataclass(frozen=True, slots=True)
class AxisPrediction:
    pixel_x: list[float]
    pixel_y: list[float]
    longitude: list[float]
    latitude: list[float]


@dataclass(frozen=True, slots=True)
class SamplePrediction:
    sample_id: str
    task: str
    axes: list[AxisPrediction]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AtmosphericScene:
    """Unlabelled model input loaded from the documented NetCDF schema."""

    sample_id: str
    task: Task
    month_index: int
    latitude: torch.Tensor
    longitude: torch.Tensor
    z500: torch.Tensor
    u500: torch.Tensor
    v500: torch.Tensor


def load_scene(
    path: str | Path, *, task: Task, month_index: int, sample_id: str | None = None
) -> AtmosphericScene:
    """Load one scene containing z500/u500/v500 and latitude/longitude."""

    if month_index not in range(12):
        raise ValueError("month_index must be in [0, 11]")
    path = Path(path)
    with netCDF4.Dataset(path) as source:
        missing = [
            name
            for name in ("z500", "u500", "v500", "latitude", "longitude")
            if name not in source.variables
        ]
        if missing:
            raise ValueError(f"Input NetCDF is missing: {', '.join(missing)}")

        def tensor(name: str) -> torch.Tensor:
            return torch.from_numpy(
                np.array(np.asarray(source[name][:]).squeeze(), dtype=np.float32, copy=True)
            )

        scene = AtmosphericScene(
            sample_id=sample_id or path.stem,
            task=task,
            month_index=month_index,
            latitude=tensor("latitude"),
            longitude=tensor("longitude"),
            z500=tensor("z500"),
            u500=tensor("u500"),
            v500=tensor("v500"),
        )
    expected = (len(scene.latitude), len(scene.longitude))
    if any(field.shape != expected for field in (scene.z500, scene.u500, scene.v500)):
        raise ValueError(f"Atmospheric fields must all have shape {expected}")
    return scene


def predict_sample(
    detector: UpperLevelDetector,
    sample: UpperLevelSample | AtmosphericScene,
    *,
    device: str = "cpu",
) -> SamplePrediction:
    """Predict georeferenced axes for one release-format sample."""

    detector.to(device).eval()
    with torch.inference_mode():
        output = detector(
            z500=sample.z500[None].to(device),
            u500=sample.u500[None].to(device),
            v500=sample.v500[None].to(device),
            longitude=sample.longitude.to(device),
            latitude=sample.latitude.to(device),
            month_index=torch.tensor([sample.month_index], device=device),
        )
    heatmap = output.axis_probability[0].cpu().numpy().copy()
    speed = torch.sqrt(sample.u500**2 + sample.v500**2).numpy()
    heatmap[speed < CALM_WIND_MS] = 0.0
    curves = extract_axes(
        heatmap,
        output.side_logits[0, 0].cpu().numpy(),
        PostprocessConfig.for_task(sample.task),
    )
    longitude = sample.longitude.numpy()
    latitude = sample.latitude.numpy()
    axes = [
        AxisPrediction(
            pixel_x=curve[:, 0].tolist(),
            pixel_y=curve[:, 1].tolist(),
            longitude=np.interp(curve[:, 0], np.arange(len(longitude)), longitude).tolist(),
            latitude=np.interp(curve[:, 1], np.arange(len(latitude)), latitude).tolist(),
        )
        for curve in curves
    ]
    return SamplePrediction(sample.sample_id, sample.task.value, axes)
