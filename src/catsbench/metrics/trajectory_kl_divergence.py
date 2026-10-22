import torch
from torchmetrics.regression import KLDivergence

from catsbench.utils import stable_clamp


class TrajectoryKLDivergence(KLDivergence):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self, 
        dim: int,
        num_timesteps: int,
        logits: bool = True,
    ):
        super().__init__(log_prob=logits, reduction='mean')
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.logits = logits

    def update(self, p: torch.Tensor, q: torch.Tensor) -> None:
        if len(p.shape) < 3 or p.shape != q.shape:
            raise ValueError(
                'Expected `p` and `q` to have the same shape with at least 3 dimensions, '
                f'but got p: {p.shape}, q: {q.shape}.'
            )
        dtype = torch.promote_types(p.dtype, q.dtype)
        p, q = p.to(dtype=dtype), q.to(dtype=dtype)
        if self.logits:
            p = p.log_softmax(dim=-1)
            q = q.log_softmax(dim=-1)

        type = 'logs' if self.logits else 'probs'
        p = stable_clamp(p, type=type)
        q = stable_clamp(q, type=type)
        
        super().update(
            p=p.flatten(end_dim=-2), 
            q=q.flatten(end_dim=-2)
        )

    def compute(self) -> torch.Tensor:
        kl_div = super().compute()
        return self.dim * self.num_timesteps * kl_div
