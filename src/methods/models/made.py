from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .time_embedding import TimeEmbedding


class MaskedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask.to(dtype=torch.bool))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight * self.mask, self.bias)


class DenoiserMADE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
        timestep_dim: int = 16,
        timestep_embedding_scale: float = 64.0,
        timestep_embedding_max_period: float = 100.0,
        layers: List[int] | None = None,
        num_layers: int = 3,
        min_hidden_dim: int = 128,
        hidden_units_per_degree: int = 8,
    ) -> None:
        super().__init__()
        if layers is None:
            hidden_dim = max(
                min_hidden_dim, hidden_units_per_degree * input_dim
            )
            layers = (hidden_dim,) * num_layers
        else:
            layers = tuple(layers)

        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.mask_token_id = num_categories
        net = []
        ch_prev = 2 * input_dim + timestep_dim
        degrees_prev = torch.cat(
            (
                torch.zeros(input_dim + timestep_dim, dtype=torch.long),
                torch.arange(1, input_dim + 1, dtype=torch.long),
            )
        )
        for ch_next in layers:
            degrees_next = torch.arange(ch_next).remainder(input_dim)
            mask = degrees_prev[None, :] <= degrees_next[:, None]
            net.extend([MaskedLinear(ch_prev, ch_next, mask), nn.ReLU()])
            ch_prev = ch_next
            degrees_prev = degrees_next

        output_degrees = torch.arange(1, input_dim + 1).repeat_interleave(
            num_categories
        )
        mask = degrees_prev[None, :] < output_degrees[:, None]
        net.append(MaskedLinear(ch_prev, num_categories * input_dim, mask))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = TimeEmbedding(
            num_timesteps=num_timesteps,
            embedding_dim=timestep_dim,
            time_scale=timestep_embedding_scale,
            max_period=timestep_embedding_max_period,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        model_input = torch.cat(
            [x_t.float(), self.timestep_embedding(t), x_prev.float()], dim=1
        )
        logits = self.net(model_input)
        return logits.view(-1, self.input_dim, self.num_categories)
