"""Command-line entry points for training, inference, and paper verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultr_detection.assets import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    resolve_dataset,
    resolve_models,
)
from ultr_detection.checkpoints import load_detector, save_detector
from ultr_detection.config import Task, TrainingRecipe
from ultr_detection.data import UpperLevelDataset, collate_samples
from ultr_detection.evaluation import evaluate_out_of_fold
from ultr_detection.inference import load_scene, predict_sample
from ultr_detection.model import UpperLevelDetector


def _task(value: str) -> Task:
    return Task(value.lower())


def _train(arguments: argparse.Namespace) -> None:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    from ultr_detection.training import DetectionLightningModule

    recipe = TrainingRecipe.from_yaml(arguments.config)
    if not 0 <= arguments.fold < recipe.cross_validation.folds:
        raise SystemExit(
            f"--fold must be between 0 and {recipe.cross_validation.folds - 1} for this recipe"
        )
    dataset_root = resolve_dataset(
        arguments.dataset,
        revision=arguments.dataset_revision,
        allow_patterns=("data/training_data.nc", "metadata/samples.parquet"),
    )
    settings = recipe.training
    seed = settings.seed if arguments.seed is None else arguments.seed
    batch_size = settings.batch_size if arguments.batch_size is None else arguments.batch_size
    learning_rate = (
        settings.learning_rate if arguments.learning_rate is None else arguments.learning_rate
    )
    weight_decay = (
        settings.weight_decay if arguments.weight_decay is None else arguments.weight_decay
    )
    max_epochs = settings.max_epochs if arguments.max_epochs is None else arguments.max_epochs
    patience = (
        settings.early_stopping_patience if arguments.patience is None else arguments.patience
    )
    config = recipe.model
    L.seed_everything(seed, workers=True)
    train = UpperLevelDataset.from_local_release(
        dataset_root, task=recipe.task, fold=arguments.fold, split="train"
    )
    validation = UpperLevelDataset.from_local_release(
        dataset_root, task=recipe.task, fold=arguments.fold, split="validation"
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": arguments.workers,
        "collate_fn": collate_samples,
        "persistent_workers": arguments.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    module = DetectionLightningModule(
        UpperLevelDetector(config),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    )
    logger: Any = False
    if arguments.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=arguments.wandb_project, name=arguments.run_name)
    checkpoint = ModelCheckpoint(
        dirpath=Path(arguments.output) / ".training",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=arguments.accelerator,
        devices=1,
        deterministic=True,
        logger=logger,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="validation/loss", mode="min", patience=patience),
        ],
    )
    trainer.fit(module, train_loader, validation_loader)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
    module.load_state_dict(best["state_dict"], strict=True)
    save_detector(module.model, arguments.output)
    if checkpoint.best_model_score is None:
        raise RuntimeError("Training completed without a validation score")
    Path(arguments.output, "training.json").write_text(
        json.dumps(
            {
                "recipe_schema_version": recipe.schema_version,
                "recipe": Path(arguments.config).name,
                "task": recipe.task.value,
                "fold": arguments.fold,
                "seed": seed,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
                "early_stopping_patience": patience,
                "best_validation_loss": float(checkpoint.best_model_score.item()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _infer(arguments: argparse.Namespace) -> None:
    sample: Any
    if arguments.input:
        if arguments.month is None:
            raise SystemExit("--month is required with --input")
        sample = load_scene(
            arguments.input,
            task=arguments.task,
            month_index=arguments.month - 1,
        )
    else:
        dataset_root = resolve_dataset(
            arguments.dataset,
            revision=arguments.dataset_revision,
            allow_patterns=("data/training_data.nc", "metadata/samples.parquet"),
        )
        dataset = UpperLevelDataset.from_local_release(dataset_root, task=arguments.task)
        try:
            position = dataset.sample_ids.index(arguments.sample_id)
        except ValueError as error:
            raise SystemExit(
                f"Unknown {arguments.task.value} sample: {arguments.sample_id}"
            ) from error
        sample = dataset[position]
    if arguments.checkpoint:
        checkpoint = Path(arguments.checkpoint)
    else:
        checkpoint_relative = f"production/{arguments.task.value}"
        model_root = resolve_models(
            arguments.models,
            revision=arguments.model_revision,
            allow_patterns=(f"{checkpoint_relative}/*",),
            required=(checkpoint_relative,),
        )
        checkpoint = model_root / checkpoint_relative
    result = predict_sample(
        load_detector(checkpoint, device=arguments.device), sample, device=arguments.device
    )
    rendered = json.dumps(result.to_dict(), indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _benchmark(arguments: argparse.Namespace) -> None:
    dataset_root = resolve_dataset(
        arguments.dataset,
        revision=arguments.dataset_revision,
        allow_patterns=(
            "data/training_data.nc",
            "metadata/samples.parquet",
            "metadata/axes.parquet",
        ),
        required=(
            "data/training_data.nc",
            "metadata/samples.parquet",
            "metadata/axes.parquet",
        ),
    )
    checkpoint_relative = f"cross_validation/{arguments.task.value}"
    model_root = resolve_models(
        arguments.models,
        revision=arguments.model_revision,
        allow_patterns=(f"{checkpoint_relative}/**",),
        required=(checkpoint_relative,),
    )

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"{done}/{total}", flush=True)

    result, _ = evaluate_out_of_fold(
        dataset_root,
        arguments.task,
        model_root=model_root,
        device=arguments.device,
        progress=progress,
    )
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ultr")
    subparsers = parser.add_subparsers(required=True)
    data_common = argparse.ArgumentParser(add_help=False)
    data_common.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_REPO,
        help="Local dataset path or Hugging Face Dataset repo ID",
    )
    data_common.add_argument("--dataset-revision", help="Immutable Dataset commit or tag")

    inference_common = argparse.ArgumentParser(add_help=False, parents=[data_common])
    inference_common.add_argument(
        "--models",
        default=DEFAULT_MODEL_REPO,
        help="Local model path or Hugging Face Model repo ID",
    )
    inference_common.add_argument("--model-revision", help="Immutable Model commit or tag")
    inference_common.add_argument("--task", required=True, type=_task, choices=list(Task))

    train = subparsers.add_parser("train", parents=[data_common])
    train.set_defaults(handler=_train)
    train.add_argument("--config", type=Path, required=True, help="Local YAML training recipe")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--seed", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--patience", type=int)
    train.add_argument("--accelerator", default="auto")
    train.add_argument("--wandb-project")
    train.add_argument("--run-name")

    infer = subparsers.add_parser("infer", parents=[inference_common])
    infer.set_defaults(handler=_infer)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-id", help="Sample from the released benchmark")
    source.add_argument("--input", help="One external z500/u500/v500 NetCDF scene")
    infer.add_argument("--month", type=int, choices=range(1, 13), help="Month of external scene")
    infer.add_argument("--checkpoint")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--output")

    benchmark = subparsers.add_parser("benchmark", parents=[inference_common])
    benchmark.set_defaults(handler=_benchmark)
    benchmark.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
