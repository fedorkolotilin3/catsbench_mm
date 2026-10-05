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
        self._decode_cache = None

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

        use_cache = (
            not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
        )
        if not use_cache:
            self._decode_cache = None
        elif self._decode_cache is None:
            is_first_step = torch.all(x_prev == self.mask_token_id).item()
            if is_first_step:
                self._decode_cache = self.init_decode_cache(x, t)

        if self._decode_cache is not None:
            index = self._decode_cache["index"]
            input_token = (
                x_prev[:, 0] if index == 0 else x_prev[:, index - 1]
            )
            logits = self.decode_step(input_token, self._decode_cache)
            if index == self.content_seq_len - 1:
                self._decode_cache = None
            return logits[:, None].expand(-1, self.content_seq_len, -1)

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

    def init_decode_cache(self, x: torch.Tensor, t: torch.Tensor):
        if self.training:
            raise RuntimeError("Decode cache is only available in eval mode.")
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.content_seq_len)
        state_type = self.model.content_emb.emb.weight[self.mask_token_id]
        context = self.model.content_emb(x.clone()) + state_type[None, None]
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0))
        neighbor_indices = []
        for index in range(self.content_seq_len):
            row, column = divmod(index, self.input_dim)
            neighbors = []
            for row_offset, column_offset in offsets:
                source_row = row + row_offset
                source_column = column + column_offset
                if (
                    0 <= source_row < self.input_dim
                    and 0 <= source_column < self.input_dim
                ):
                    neighbors.append(source_row * self.input_dim + source_column)
                else:
                    neighbors.append(self.content_seq_len)
            neighbor_indices.append(neighbors)
        neighbor_indices = torch.tensor(
            neighbor_indices, dtype=torch.long, device=x.device
        )
        blocks = []
        for block in self.model.blocks:
            scale, shift = block.ln1.get_scale_shift(t)
            blocks.append(
                {
                    "attention": block.attn.init_cache(
                        context, self.content_seq_len
                    ),
                    "mlp": block.mlp.init_cache(
                        context,
                        self.content_seq_len,
                        neighbor_indices,
                    ),
                    "scale": scale,
                    "shift": shift,
                }
            )
        return {"blocks": blocks, "timestep": t, "index": 0}

    def decode_step(self, token: torch.Tensor, cache) -> torch.Tensor:
        index = cache["index"]
        if index >= self.content_seq_len:
            raise RuntimeError("Decode cache is full.")
        if token.dim() == 1:
            token = token[:, None]

        embedding = self.model.content_emb
        row, column = divmod(index, self.input_dim)
        x = embedding.emb(token)
        x = x + embedding.height_emb.weight[row]
        x = x + embedding.width_emb.weight[column]

        for block, block_cache in zip(self.model.blocks, cache["blocks"]):
            attention_input = block.ln1.layernorm(x)
            attention_input = attention_input * (1 + block_cache["scale"])
            attention_input = attention_input + block_cache["shift"]
            x = x + block.attn.decode_step(
                attention_input, block_cache["attention"]
            )
            mlp_input = block.ln2(x)
            x = x + block.mlp.decode_step(
                mlp_input,
                index,
                block_cache["mlp"],
            )

        cache["index"] += 1
        return self.model.to_logits(x)[:, 0]
