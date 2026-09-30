import torch
from torch import nn

from ...vq_diffusion.modeling.transformers.transformer_utils import UnCondition2ImageTransformer


class LatentTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
        hidden_dim: int = 256,
        num_channels: int = 4,
        num_layers: int = 18,
        num_att_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.content_seq_len = input_dim * input_dim
        self.mask_token_id = num_categories
        content_emb_config = {
            "num_embed": num_categories,
            "spatial_size": input_dim,
            "embed_dim": hidden_dim,
            "trainable": True,
            "pos_emb_type": "embedding",
        }
        self.model = UnCondition2ImageTransformer(
            n_layer=num_layers,
            n_embd=hidden_dim,
            n_head=num_att_heads,
            content_seq_len=input_dim * input_dim,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            mlp_hidden_times=num_channels,
            block_activate="GELU2",
            attn_type="self",
            content_spatial_size=[input_dim, input_dim],
            diffusion_step=num_timesteps + 2,
            timestep_type="adalayernorm",
            content_emb_config=content_emb_config,
            mlp_type="conv_mlp",
        )
        prefix_mask = torch.tril(
            torch.ones(self.content_seq_len, self.content_seq_len, dtype=torch.bool)
        )
        state_mask = torch.ones_like(prefix_mask)
        causal_mask = torch.cat((state_mask, prefix_mask), dim=-1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_prev is None:
            return self.model(x, t)
        if x.shape != x_prev.shape:
            raise ValueError(
                f"x and x_prev must have equal shapes, got "
                f"{tuple(x.shape)} and {tuple(x_prev.shape)}"
            )

        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.content_seq_len)
        x_prev = x_prev.reshape(batch_size, self.content_seq_len)

        shifted = torch.full_like(x_prev, self.mask_token_id)
        shifted[:, 1:] = x_prev[:, :-1]

        state_type = self.model.content_emb.emb.weight[self.mask_token_id]
        context = self.model.content_emb(x.clone()) + state_type[None, None]
        return self.model(
            shifted,
            t,
            context=context,
            mask=self.causal_mask,
        )
