import torch
from torch import nn


class CausalDenoiserTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
        hidden_dim: int = 56,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 112,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if num_categories < 2:
            raise ValueError("num_categories must be at least 2")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.mask_token_id = num_categories
        self.bos_token_id = num_categories + 1

        self.state_embedding = nn.Embedding(num_categories, hidden_dim)
        self.prefix_embedding = nn.Embedding(num_categories + 2, hidden_dim)
        self.position_embedding = nn.Embedding(input_dim, hidden_dim)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, hidden_dim)

        self.state_projection = nn.Linear(input_dim * hidden_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.output_weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim, num_categories)
        )
        self.output_bias = nn.Parameter(torch.zeros(input_dim, num_categories))
        for weight in self.output_weight:
            nn.init.xavier_uniform_(weight)

        causal_mask = torch.triu(
            torch.ones(input_dim, input_dim, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _validate_inputs(
        self, x_t: torch.Tensor, t: torch.Tensor, x_prev: torch.Tensor
    ) -> None:
        if x_t.shape != x_prev.shape:
            raise ValueError(
                f"x_t and x_prev must have equal shapes, got "
                f"{tuple(x_t.shape)} and {tuple(x_prev.shape)}"
            )
        if x_t.ndim < 2 or x_t[0].numel() != self.input_dim:
            raise ValueError(
                f"Expected each state to contain {self.input_dim} coordinates, "
                f"got shape {tuple(x_t.shape)}"
            )
        if t.shape != (x_t.shape[0],):
            raise ValueError(
                f"Expected t shape ({x_t.shape[0]},), got {tuple(t.shape)}"
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Return all conditional logits using a shifted causal prefix."""
        self._validate_inputs(x_t, t, x_prev)
        batch_size = x_t.shape[0]
        event_shape = x_t.shape[1:]
        x_flat = x_t.reshape(batch_size, self.input_dim).long()
        prefix_flat = x_prev.reshape(batch_size, self.input_dim).long()

        shifted = torch.full_like(prefix_flat, self.bos_token_id)
        if self.input_dim > 1:
            shifted[:, 1:] = prefix_flat[:, :-1]

        state = self.state_embedding(x_flat).flatten(start_dim=1)
        state_context = self.state_projection(state)[:, None, :]
        positions = torch.arange(self.input_dim, device=x_t.device)
        tokens = (
            self.prefix_embedding(shifted)
            + self.position_embedding(positions)[None]
            + self.timestep_embedding(t)[:, None, :]
            + state_context
        )
        hidden = self.transformer(tokens, mask=self.causal_mask)
        hidden = self.output_norm(hidden)
        logits = torch.einsum("bdh,dhs->bds", hidden, self.output_weight)
        logits = logits + self.output_bias[None]
        return logits.reshape(
            batch_size, *event_shape, self.num_categories
        )
