"""Resolve versioned dataset and model assets from local paths or the HF Hub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_DATASET_REPO = "Omer-Sela/upper-level-trough-ridge-detection-data"
DEFAULT_MODEL_REPO = "Omer-Sela/upper-level-trough-ridge-detection-models"


@dataclass(frozen=True, slots=True)
class AssetRoots:
    """Resolved immutable snapshots for dataset files and model artifacts."""

    dataset: Path
    models: Path


def _validate(root: Path, required: tuple[str, ...], kind: str) -> Path:
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Invalid {kind} release at {root}; missing: {', '.join(missing)}")
    return root.resolve()


def _resolve(
    source: str,
    *,
    repo_type: str,
    revision: str | None,
    required: tuple[str, ...],
    allow_patterns: tuple[str, ...] | None = None,
) -> Path:
    local = Path(source)
    if local.exists():
        return _validate(local, required, repo_type)
    if allow_patterns is None:
        snapshot = snapshot_download(repo_id=source, repo_type=repo_type, revision=revision)
    else:
        snapshot = snapshot_download(
            repo_id=source,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=list(allow_patterns),
        )
    root = Path(snapshot)
    return _validate(root, required, repo_type)


def resolve_dataset(
    source: str = DEFAULT_DATASET_REPO,
    *,
    revision: str | None = None,
    allow_patterns: tuple[str, ...] | None = None,
    required: tuple[str, ...] = ("data/training_data.nc", "metadata/samples.parquet"),
) -> Path:
    """Resolve a Dataset repository snapshot, optionally filtering its files."""

    return _resolve(
        source,
        repo_type="dataset",
        revision=revision,
        required=required,
        allow_patterns=allow_patterns,
    )


def resolve_models(
    source: str = DEFAULT_MODEL_REPO,
    *,
    revision: str | None = None,
    allow_patterns: tuple[str, ...] | None = None,
    required: tuple[str, ...] = ("production", "cross_validation", "manifest.json"),
) -> Path:
    """Resolve a Model repository snapshot, optionally filtering its files."""

    local = Path(source)
    if local.exists() and (local / "checkpoints").is_dir():
        source = str(local / "checkpoints")
    return _resolve(
        source,
        repo_type="model",
        revision=revision,
        required=required,
        allow_patterns=allow_patterns,
    )


def resolve_assets(
    dataset_source: str = DEFAULT_DATASET_REPO,
    model_source: str = DEFAULT_MODEL_REPO,
    *,
    dataset_revision: str | None = None,
    model_revision: str | None = None,
) -> AssetRoots:
    """Resolve the independently versioned HF Dataset and Model repositories.

    Sources may also be local directories. For compatibility with the audited
    assembly tree, a local model source containing a ``checkpoints`` directory
    is resolved to that child automatically.
    """

    return AssetRoots(
        dataset=resolve_dataset(
            dataset_source,
            revision=dataset_revision,
        ),
        models=resolve_models(
            model_source,
            revision=model_revision,
        ),
    )
