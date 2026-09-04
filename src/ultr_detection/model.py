"""Physics-informed local-attention model selected in the paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ultr_detection.config import ModelConfig, Task


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    """Dense intermediate and final outputs for audit-friendly inference."""

    feature_mask: torch.Tensor
    curvature: torch.Tensor
    gradient_magnitude: torch.Tensor
    axis_probability: torch.Tensor
    side_logits: torch.Tensor


def _pad_for_unfold(value: torch.Tensor, window_size: int) -> torch.Tensor:
    left = window_size // 2
    right = window_size - 1 - left
    return F.pad(value, (left, right, left, right), mode="replicate")


class LocalSelfAttentionAxisDetector(nn.Module):
    """Convolutional encoder followed by local self-attention."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        predict_sides: bool,
        window_size: int,
        num_heads: int,
        num_attn_blocks: int,
    ) -> None:
        super().__init__()
        if num_heads != 1 or num_attn_blocks != 1:
            raise ValueError("Released checkpoints use one attention head and one attention block")
        self.hidden_dim = hidden_dim
        self.predict_sides = predict_sides
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.qkv = nn.Conv2d(hidden_dim, hidden_dim * 3, 1)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        out_channels = 2 if predict_sides else 1
        if predict_sides:
            self.head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 2, out_channels, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 2, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        query, key, value = self.qkv(encoded).chunk(3, dim=1)
        attended = self._local_attention(query, key, value)
        output = self.head(encoded + self.out_proj(attended))
        if self.predict_sides:
            return torch.sigmoid(output[:, 0]), output[:, 1:2]
        return output.squeeze(1), torch.zeros_like(output)

    def _local_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = query.shape
        window = self.window_size
        key_windows = F.unfold(_pad_for_unfold(key, window), window).view(
            batch, channels, window * window, height, width
        )
        value_windows = F.unfold(_pad_for_unfold(value, window), window).view(
            batch, channels, window * window, height, width
        )
        scores = (query.unsqueeze(2) * key_windows).sum(dim=1, keepdim=True) / channels**0.5
        attention = F.softmax(scores, dim=2)
        return (attention * value_windows).sum(dim=2)


class UpperLevelDetector(nn.Module):
    """Differentiable curvature front end and learned axis detector."""

    wind_u_monthly_mean: torch.Tensor
    wind_u_monthly_std: torch.Tensor
    wind_v_monthly_mean: torch.Tensor
    wind_v_monthly_std: torch.Tensor

    latitude_degree_km = 111.0

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(config.threshold))
            self.absgradmin = nn.Parameter(torch.tensor(config.minimum_gradient))
        else:
            self.register_buffer("threshold", torch.tensor(config.threshold))
            self.register_buffer("absgradmin", torch.tensor(config.minimum_gradient))
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        if config.learnable_distance:
            self.dist = nn.Parameter(torch.tensor(config.curvature_distance_km))
        else:
            self.register_buffer("dist", torch.tensor(config.curvature_distance_km))

        # These names intentionally preserve the stable released checkpoint schema.
        self.register_buffer("wind_u_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("wind_u_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("wind_v_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("wind_v_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("wind_u_monthly_mean", torch.tensor(config.wind.u_monthly_mean))
        self.register_buffer("wind_u_monthly_std", torch.tensor(config.wind.u_monthly_std))
        self.register_buffer("wind_v_monthly_mean", torch.tensor(config.wind.v_monthly_mean))
        self.register_buffer("wind_v_monthly_std", torch.tensor(config.wind.v_monthly_std))
        # Compatibility buffers remain part of the stable checkpoint schema.
        for name, default in (
            ("wind_speed_monthly_mean", 0.0),
            ("wind_speed_monthly_std", 1.0),
            ("wind_cos_monthly_mean", 0.0),
            ("wind_cos_monthly_std", 1.0),
            ("wind_sin_monthly_mean", 0.0),
            ("wind_sin_monthly_std", 1.0),
        ):
            self.register_buffer(name, torch.full((12,), default, dtype=torch.float32))

        self.axis_detector = LocalSelfAttentionAxisDetector(
            in_channels=7,
            hidden_dim=config.hidden_dim,
            predict_sides=config.predict_sides,
            window_size=config.attention_window,
            num_heads=config.attention_heads,
            num_attn_blocks=config.attention_blocks,
        )

    def forward(
        self,
        *,
        z500: torch.Tensor,
        u500: torch.Tensor,
        v500: torch.Tensor,
        longitude: torch.Tensor,
        latitude: torch.Tensor,
        month_index: torch.Tensor | None = None,
    ) -> DetectorOutput:
        gradient_x, gradient_y = self._gradient(z500, longitude, latitude)
        gradient_x, gradient_y, magnitude = self._normalize_gradient(gradient_x, gradient_y)
        tangent_x = -gradient_y
        tangent_y = gradient_x
        curvature = self._turning_angle(tangent_x, tangent_y, longitude, latitude) / self.dist
        valid_gradient = torch.sigmoid(self.temperature * (magnitude - self.absgradmin))
        signed_curvature = -curvature if self.config.task is Task.RIDGE else curvature
        feature_mask = valid_gradient * torch.sigmoid(
            self.temperature * (signed_curvature - self.threshold)
        )
        feature_mask = torch.clamp(self._fill_holes(feature_mask), 0.0, 1.0)

        z_mean = z500.mean(dim=(1, 2), keepdim=True)
        z_std = z500.std(dim=(1, 2), keepdim=True)
        z_std = torch.where(z_std < 1e-6, torch.ones_like(z_std), z_std)
        normalized_z = (z500 - z_mean) / z_std
        direction_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2 + 1e-8)
        normalized_u, normalized_v = self._normalize_wind(u500, v500, month_index)
        features = torch.stack(
            [
                feature_mask,
                gradient_x,
                gradient_y,
                normalized_z,
                direction_magnitude,
                normalized_u,
                normalized_v,
            ],
            dim=1,
        )
        if feature_mask.sum() < 1e-6:
            axis_probability = torch.zeros_like(feature_mask)
            side_logits = torch.zeros(
                z500.shape[0], 1, z500.shape[1], z500.shape[2], device=z500.device
            )
        else:
            axis_probability, side_logits = self.axis_detector(features)
            axis_probability = torch.nan_to_num(axis_probability)
            side_logits = torch.nan_to_num(side_logits)
        return DetectorOutput(
            feature_mask=feature_mask,
            curvature=curvature,
            gradient_magnitude=magnitude,
            axis_probability=axis_probability,
            side_logits=side_logits,
        )

    def _gradient(
        self, field: torch.Tensor, longitude: torch.Tensor, latitude: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = field.shape
        kernel_y = torch.tensor([[-1.0], [0.0], [1.0]], device=field.device).view(1, 1, 3, 1) / 2.0
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0]], device=field.device).view(1, 1, 1, 3) / 2.0
        field_4d = field.unsqueeze(1)
        gradient_y_raw = F.conv2d(
            F.pad(field_4d, (0, 0, 1, 1), mode="replicate"), kernel_y
        ).squeeze(1)
        gradient_x_raw = F.conv2d(
            F.pad(field_4d, (1, 1, 0, 0), mode="replicate"), kernel_x
        ).squeeze(1)
        latitude_step = torch.abs(latitude[1] - latitude[0]).item()
        longitude_step = torch.abs(longitude[1] - longitude[0]).item()
        gradient_y = gradient_y_raw / (2.0 * latitude_step * self.latitude_degree_km)
        latitude_grid = latitude.view(1, -1, 1).expand(batch, height, width)
        scale_x = 1.0 / (
            2.0 * longitude_step * self.latitude_degree_km * torch.cos(torch.deg2rad(latitude_grid))
            + 1e-8
        )
        return gradient_x_raw * scale_x, gradient_y

    @staticmethod
    def _normalize_gradient(
        gradient_x: torch.Tensor, gradient_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epsilon = 1e-10
        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2 + epsilon)
        return gradient_x / (magnitude + epsilon), gradient_y / (magnitude + epsilon), magnitude

    def _turning_angle(
        self,
        tangent_x: torch.Tensor,
        tangent_y: torch.Tensor,
        longitude: torch.Tensor,
        latitude: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width = tangent_x.shape
        longitude_grid = longitude.view(1, 1, -1).expand(batch, height, width)
        latitude_grid = latitude.view(1, -1, 1).expand(batch, height, width)
        cosine = torch.cos(torch.deg2rad(latitude_grid)) + 1e-8
        longitude_offset = tangent_x * self.dist / (self.latitude_degree_km * cosine)
        latitude_offset = tangent_y * self.dist / self.latitude_degree_km
        longitude_1 = torch.clamp(
            longitude_grid + longitude_offset, longitude.min(), longitude.max()
        )
        latitude_1 = torch.clamp(latitude_grid + latitude_offset, latitude.min(), latitude.max())
        normalized_longitude = (
            2.0 * (longitude_1 - longitude.min()) / (longitude.max() - longitude.min() + 1e-8) - 1.0
        )
        normalized_latitude = (
            2.0 * (latitude_1 - latitude.min()) / (latitude.max() - latitude.min() + 1e-8) - 1.0
        )
        grid = torch.stack([normalized_longitude, normalized_latitude], dim=-1)
        sampled_x = F.grid_sample(
            tangent_x.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)
        sampled_y = F.grid_sample(
            tangent_y.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)
        cross = tangent_x * sampled_y - tangent_y * sampled_x
        dot = tangent_x * sampled_x + tangent_y * sampled_y
        return torch.rad2deg(torch.atan2(cross, dot))

    @staticmethod
    def _fill_holes(mask: torch.Tensor, kernel_size: int = 5, iterations: int = 5) -> torch.Tensor:
        current = mask.unsqueeze(1) if mask.ndim == 3 else mask
        for _ in range(iterations):
            dilated = F.max_pool2d(current, kernel_size, stride=1, padding=kernel_size // 2)
            current = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=kernel_size // 2)
        return current.squeeze(1)

    def _normalize_wind(
        self, u500: torch.Tensor, v500: torch.Tensor, month_index: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = u500.shape[0]
        if month_index is None:
            month_index = torch.zeros(batch, dtype=torch.long, device=u500.device)
        month_index = month_index.to(device=u500.device, dtype=torch.long).clamp(0, 11)
        u_mean = self.wind_u_monthly_mean[month_index].view(batch, 1, 1)
        u_std = self.wind_u_monthly_std[month_index].view(batch, 1, 1).clamp(min=1e-6)
        v_mean = self.wind_v_monthly_mean[month_index].view(batch, 1, 1)
        v_std = self.wind_v_monthly_std[month_index].view(batch, 1, 1).clamp(min=1e-6)
        return (u500 - u_mean) / u_std, (v500 - v_mean) / v_std
