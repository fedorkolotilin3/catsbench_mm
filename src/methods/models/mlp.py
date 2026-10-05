from typing import List

import torch
from torch import nn

from .time_embedding import TimeEmbedding


class DenoiserMLP(nn.Module):
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
        hidden_units_per_dimension: int = 8,
     ) -> None: 
        super().__init__()
        if layers is None:
            hidden_dim = max(
                min_hidden_dim, hidden_units_per_dimension * input_dim
            )
            layers = (hidden_dim,) * num_layers
        else:
            layers = tuple(layers)

        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        net = []
        ch_prev = input_dim + timestep_dim
        for ch_next in layers:
            net.extend([nn.Linear(ch_prev, ch_next), nn.ReLU()])
            ch_prev = ch_next
        net.append(nn.Linear(ch_prev, num_categories * input_dim))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = TimeEmbedding(
            num_timesteps=num_timesteps,
            embedding_dim=timestep_dim,
            time_scale=timestep_embedding_scale,
            max_period=timestep_embedding_max_period,
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_start_logits = self.net(torch.cat([x.float(), self.timestep_embedding(t)], dim=1))
        x_start_logits = x_start_logits.view(-1, self.input_dim, self.num_categories)
        return x_start_logits