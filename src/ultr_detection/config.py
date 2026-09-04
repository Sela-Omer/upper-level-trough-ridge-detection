"""Typed, validated configuration for the released detector family."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Task(StrEnum):
    """Synoptic feature detected by a model."""

    TROUGH = "trough"
    RIDGE = "ridge"


def _twelve(values: Any, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 12:
        raise ValueError(f"{name} must contain 12 monthly values")
    return result


@dataclass(frozen=True, slots=True)
class WindNormalization:
    """Monthly ERA5 wind statistics stored with every released checkpoint."""

    u_monthly_mean: tuple[float, ...]
    u_monthly_std: tuple[float, ...]
    v_monthly_mean: tuple[float, ...]
    v_monthly_std: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("u_monthly_mean", "u_monthly_std", "v_monthly_mean", "v_monthly_std"):
            if len(getattr(self, name)) != 12:
                raise ValueError(f"{name} must contain 12 values")
        if any(value <= 0 for value in (*self.u_monthly_std, *self.v_monthly_std)):
            raise ValueError("Wind standard deviations must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration of the architecture selected in the paper."""

    task: Task
    threshold: float
    minimum_gradient: float
    curvature_distance_km: float
    temperature: float
    learnable_threshold: bool
    learnable_distance: bool
    predict_sides: bool
    hidden_dim: int
    attention_heads: int
    attention_blocks: int
    attention_window: int
    use_wind: bool
    use_cva: bool
    wind: WindNormalization

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.attention_blocks <= 0:
            raise ValueError("attention_blocks must be positive")
        if self.attention_window <= 0:
            raise ValueError("The released architecture requires local self-attention")
        if not self.use_wind:
            raise ValueError("The released architecture requires U500/V500 inputs")
        if self.use_cva:
            raise ValueError("The selected paper model does not use CVA as an input channel")

    @classmethod
    def from_checkpoint_json(cls, path: str | Path) -> ModelConfig:
        """Read a checkpoint configuration and validate the supported architecture."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "task" in raw:
            wind = raw["wind"]
            return cls(
                task=Task(raw["task"]),
                threshold=float(raw["threshold"]),
                minimum_gradient=float(raw["minimum_gradient"]),
                curvature_distance_km=float(raw["curvature_distance_km"]),
                temperature=float(raw["temperature"]),
                learnable_threshold=bool(raw["learnable_threshold"]),
                learnable_distance=bool(raw["learnable_distance"]),
                predict_sides=bool(raw["predict_sides"]),
                hidden_dim=int(raw["hidden_dim"]),
                attention_heads=int(raw["attention_heads"]),
                attention_blocks=int(raw["attention_blocks"]),
                attention_window=int(raw["attention_window"]),
                use_wind=bool(raw["use_wind"]),
                use_cva=bool(raw["use_cva"]),
                wind=WindNormalization(
                    u_monthly_mean=_twelve(wind["u_monthly_mean"], "u_monthly_mean"),
                    u_monthly_std=_twelve(wind["u_monthly_std"], "u_monthly_std"),
                    v_monthly_mean=_twelve(wind["v_monthly_mean"], "v_monthly_mean"),
                    v_monthly_std=_twelve(wind["v_monthly_std"], "v_monthly_std"),
                ),
            )
        if raw.get("axis_attention_mode") != "self":
            raise ValueError(
                "This package supports the selected local self-attention architecture only"
            )
        if raw.get("wind_representation", "uv") != "uv":
            raise ValueError("This package supports U/V wind representation only")
        unsupported = {
            "learnable_smoothing": False,
            "learnable_cva_smoothing": False,
            "use_mae_features": False,
            "use_cra5_features": False,
        }
        for key, expected in unsupported.items():
            if bool(raw.get(key, expected)) is not expected:
                raise ValueError(f"Unsupported experimental option in checkpoint: {key}")

        return cls(
            task=Task(raw.get("feature_mode", "trough")),
            threshold=float(raw["threshold"]),
            minimum_gradient=float(raw["absgradmin"]),
            curvature_distance_km=float(raw["dist"]),
            temperature=float(raw["temperature"]),
            learnable_threshold=bool(raw["learnable_threshold"]),
            learnable_distance=bool(raw["learnable_dist"]),
            predict_sides=bool(raw["predict_sides"]),
            hidden_dim=int(raw["hidden_dim"]),
            attention_heads=int(raw["num_heads"]),
            attention_blocks=int(raw["num_attn_blocks"]),
            attention_window=int(raw["window_size"]),
            use_wind=bool(raw["use_wind_channel"]),
            use_cva=bool(raw["use_cva_channel"]),
            wind=WindNormalization(
                u_monthly_mean=_twelve(raw["wind_u_monthly_mean"], "wind_u_monthly_mean"),
                u_monthly_std=_twelve(raw["wind_u_monthly_std"], "wind_u_monthly_std"),
                v_monthly_mean=_twelve(raw["wind_v_monthly_mean"], "wind_v_monthly_mean"),
                v_monthly_std=_twelve(raw["wind_v_monthly_std"], "wind_v_monthly_std"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable public configuration schema."""

        value = asdict(self)
        value["task"] = self.task.value
        return value
