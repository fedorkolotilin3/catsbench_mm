import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        embedding_dim: int,
        *,
        time_scale: float = 64.0,
        max_period: float = 100.0,
    ) -> None:
        super().__init__()
        if num_timesteps < 0:
            raise ValueError("num_timesteps must be non-negative")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if max_period <= 1:
            raise ValueError("max_period must be greater than 1")

        self.num_transitions = num_timesteps + 1
        self.embedding_dim = embedding_dim
        self.time_scale = float(time_scale)

        half_dim = embedding_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / denominator
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def fourier_features(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(
                f"Expected one-dimensional timesteps, got {tuple(timestep.shape)}"
            )
        physical_time = timestep.to(dtype=self.frequencies.dtype)
        physical_time = physical_time / self.num_transitions
        phases = physical_time[:, None] * self.time_scale * self.frequencies[None]
        embedding = torch.cat((phases.sin(), phases.cos()), dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(
                embedding, (0, self.embedding_dim - embedding.shape[-1])
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.fourier_features(timestep)
