from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ultr_detection.assets import AssetRoots, resolve_assets, resolve_models


def test_resolve_assets_accepts_valid_local_release(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "checkpoints/production").mkdir(parents=True)
    (tmp_path / "checkpoints/cross_validation").mkdir()
    (tmp_path / "data/training_data.nc").touch()
    (tmp_path / "metadata/samples.parquet").touch()
    (tmp_path / "checkpoints/manifest.json").touch()

    assert resolve_assets(str(tmp_path), str(tmp_path)) == AssetRoots(
        dataset=tmp_path.resolve(), models=(tmp_path / "checkpoints").resolve()
    )


def test_resolve_assets_downloads_independent_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    models = tmp_path / "models"
    (dataset / "data").mkdir(parents=True)
    (dataset / "metadata").mkdir()
    (models / "production").mkdir(parents=True)
    (models / "cross_validation").mkdir()
    (dataset / "data/training_data.nc").touch()
    (dataset / "metadata/samples.parquet").touch()
    (models / "manifest.json").touch()
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(dataset if kwargs["repo_type"] == "dataset" else models)

    monkeypatch.setattr("ultr_detection.assets.snapshot_download", fake_download)
    assert resolve_assets(
        "owner/data",
        "owner/models",
        dataset_revision="data123",
        model_revision="model456",
    ) == AssetRoots(dataset=dataset.resolve(), models=models.resolve())
    assert calls == [
        {"repo_id": "owner/data", "repo_type": "dataset", "revision": "data123"},
        {"repo_id": "owner/models", "repo_type": "model", "revision": "model456"},
    ]


def test_resolve_models_filters_remote_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    checkpoint = models / "production/trough"
    checkpoint.mkdir(parents=True)
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(models)

    monkeypatch.setattr("ultr_detection.assets.snapshot_download", fake_download)
    assert (
        resolve_models(
            "owner/models",
            revision="model456",
            allow_patterns=("production/trough/*",),
            required=("production/trough",),
        )
        == models.resolve()
    )
    assert calls == [
        {
            "repo_id": "owner/models",
            "repo_type": "model",
            "revision": "model456",
            "allow_patterns": ["production/trough/*"],
        }
    ]
