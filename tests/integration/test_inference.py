from __future__ import annotations

from ultr_detection.assets import AssetRoots
from ultr_detection.checkpoints import load_detector
from ultr_detection.config import Task
from ultr_detection.data import UpperLevelDataset
from ultr_detection.inference import predict_sample


def test_production_inference_returns_georeferenced_axes(
    release_roots: AssetRoots,
) -> None:
    dataset = UpperLevelDataset.from_local_release(release_roots.dataset, task=Task.TROUGH)
    sample = dataset[0]
    detector = load_detector(release_roots.models / "production/trough")

    result = predict_sample(detector, sample)

    assert result.sample_id == sample.sample_id
    assert result.task == "trough"
    for axis in result.axes:
        assert len(axis.pixel_x) == len(axis.longitude)
        assert len(axis.pixel_y) == len(axis.latitude)
        assert min(axis.longitude) >= float(sample.longitude.min())
        assert max(axis.longitude) <= float(sample.longitude.max())
        assert min(axis.latitude) >= float(sample.latitude.min())
        assert max(axis.latitude) <= float(sample.latitude.max())
