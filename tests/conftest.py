from __future__ import annotations

import os
from pathlib import Path

import pytest

from ultr_detection.assets import AssetRoots, resolve_assets


@pytest.fixture(scope="session")
def release_roots() -> AssetRoots:
    """Resolve optional binary assets for integration and paper tests."""

    local = os.environ.get("ULTR_ASSET_ROOT")
    if local:
        root = Path(local)
        dataset = root
        models = root / "checkpoints" if (root / "checkpoints").is_dir() else root
        if not (dataset / "data/training_data.nc").is_file():
            pytest.fail(f"ULTR_ASSET_ROOT does not contain the dataset release: {root}")
        return AssetRoots(dataset=dataset, models=models)

    if (
        os.environ.get("ULTR_RUN_ASSET_TESTS") != "1"
        and os.environ.get("ULTR_RUN_PAPER_TESTS") != "1"
    ):
        pytest.skip("Set ULTR_ASSET_ROOT or enable the asset-backed test suite")

    return resolve_assets(
        dataset_revision=os.environ.get("ULTR_DATA_REVISION"),
        model_revision=os.environ.get("ULTR_MODEL_REVISION"),
    )
