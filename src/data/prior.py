from typing import Literal, Optional, Union

import torch
from torch import nn

from catsbench.lse_matmul import LSEImplementation
from catsbench.utils import lse_matmul, logits_prod 
from ..utils import broadcast, gumbel_sample


def get_cum_matrices(
    prior_type: Literal['uniform', 'gaussian'],
    alpha: float,
    num_timesteps: int, 
    num_skip_steps: int,
    log_onestep_matrix: torch.Tensor
) -> torch.Tensor:
    num_categories = log_onestep_matrix.shape[0]
    dtype, device = log_onestep_matrix.dtype, log_onestep_matrix.device
    log_cum_matrices = torch.empty(
        size=(num_timesteps + 2, num_categories, num_categories), 
        dtype=dtype, device=device
    )
    
    # Add identity matrix for the zero timestep
    log_cum_matrices[0] = torch.full(
        (num_categories, num_categories), float('-inf'), 
        dtype=dtype, device=device
    )
    diag_value = torch.as_tensor(
        0.0, dtype=dtype, device=device
    )
    log_cum_matrices[0].diagonal().fill_(diag_value)

    # compute cumulative matrices
    for timestep in range(1, num_timesteps + 2):
        if prior_type == 'uniform': # use closed form
            log_cum_matrices[timestep] = uniform_onestep(
                alpha, num_categories, timestep*num_skip_steps, 
                dtype=dtype, device=device
            )
        else: # use log space matrix multiplication
            log_cum_matrices[timestep] = lse_matmul(
                log_cum_matrices[timestep-1], 
                log_onestep_matrix,
                implementation="triton",
                eps=0.0,
            )
    return log_cum_matrices
    
def uniform_onestep(
    alpha: float, 
    num_categories: int, 
    step: int,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    alpha = 1 - (alpha * num_categories) / (num_categories - 1)
    alpha_n = alpha ** step
    
    off_diag_value = torch.as_tensor(
        (1.0 - alpha_n) / num_categories, dtype=dtype, device=device
    )
    diag_value = torch.as_tensor(
        alpha_n + off_diag_value, dtype=dtype, device=device
    )

    log_p_onestep_mat = torch.empty(
        (num_categories, num_categories), dtype=dtype, device=device
    )
    log_p_onestep_mat.fill_(off_diag_value.log())
    log_p_onestep_mat.diagonal().fill_(diag_value.log())

    return log_p_onestep_mat

def gaussian_onestep(
    alpha: float,
    num_categories: int, 
    num_skip_steps: int,
    use_doubly_stochastic: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    max_distance = num_categories - 1
    if not use_doubly_stochastic:
        indices = torch.arange(
            num_categories, dtype=dtype, device=device
        )[None, ...]
        values = (-4 * (indices - indices.T)**2) / ((alpha *max_distance)**2)
        log_p_onestep_mat = values.log_softmax(dim=1)

    else: 
        sum_aux = -4 * torch.arange(
            -num_categories + 1, num_categories+1, 
            dtype=dtype, device=device
        )**2 / (alpha * max_distance)**2
        log_Z = torch.logsumexp(sum_aux, dim=0)

        indices = torch.arange(num_categories, dtype=dtype, device=device)
        diff = indices[:, None] - indices[None, :]
        exp_argument = -4*(diff)**2/(alpha * max_distance)**2
        log_pi_vals = exp_argument - log_Z

        off_diag_mask = (diff != 0)

        off_diag_probs = torch.exp(log_pi_vals) * off_diag_mask.float()
        diag_log = torch.log1p(-off_diag_probs.sum(dim=1, keepdim=True))
        
        log_p_onestep_mat = torch.full(
            (num_categories, num_categories), float('-inf'), 
            dtype=dtype, device=device
        )
        
        log_p_onestep_mat[off_diag_mask] = log_pi_vals[off_diag_mask]
        
        rows = torch.arange(num_categories)
        log_p_onestep_mat[rows, rows] = diag_log.squeeze(1)

    if num_skip_steps == 1:
        return log_p_onestep_mat
    
    log_p_onestep_mat_orig = log_p_onestep_mat.clone()
    for _ in range(num_skip_steps - 1):
        log_p_onestep_mat = lse_matmul(
            log_p_onestep_mat,
            log_p_onestep_mat_orig,
            implementation="triton",
            eps=0.0,
        )
    return log_p_onestep_mat

# Cumulative returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        0->2        ...         0->N        0->N+1       

# Inherit from nn.Module to do device casting automatically
class Prior(nn.Module):
    log_p_onestep: torch.Tensor
    log_p_cum: torch.Tensor
    
    def __init__(
        self, 
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        tau: float = 1.0,
        eps: float = 1e-20,
        prior_type: Literal[
            'uniform', 
            'gaussian',
        ] = 'uniform',
        implementation: LSEImplementation = "normalized",
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.num_skip_steps = num_skip_steps
        self.tau = tau
        self.eps = eps
        self.prior_type = prior_type
        self.implementation = implementation
        if eps < 0:
            raise ValueError(f'eps must be non-negative, got {eps}.')
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)

        if prior_type == 'gaussian':
            log_p_onestep = gaussian_onestep(
                alpha, num_categories, num_skip_steps, 
                dtype=dtype, device=device
            )
        elif prior_type == 'uniform':
            if eps > 0:
                retention = 1 - (alpha * num_categories) / (num_categories - 1)
                retention /= (1 + num_categories * eps) ** (1 / num_skip_steps)
                alpha = (1 - retention) * (num_categories - 1) / num_categories
            log_p_onestep = uniform_onestep(
                alpha, num_categories, num_skip_steps, 
                dtype=dtype, device=device
            )
        else:
            raise NotImplementedError(f'Got unknown prior: {prior_type}!')
        if prior_type == 'gaussian' and eps > 0:
            log_p_onestep = self.add_eps(log_p_onestep)
            log_p_onestep -= log_p_onestep.logsumexp(dim=-1, keepdim=True)
        log_p_cum = get_cum_matrices(
            prior_type, alpha, num_timesteps, 
            num_skip_steps, log_p_onestep
        )
        log_p_onestep = log_p_onestep.transpose(0, 1).contiguous()
        # register as non-persistent buffer to avoid saving in checkpoints
        self.register_buffer('log_p_onestep', log_p_onestep, persistent=False)
        self.register_buffer('log_p_cum', log_p_cum, persistent=False)

    def _reduction_eps(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ) -> float:
        if self.implementation != "normalized" or self.eps != 0:
            return 0.0
        dtype = torch.promote_types(tensor1.dtype, tensor2.dtype)
        return torch.finfo(dtype).tiny

    @property
    def dtype(self) -> torch.dtype:
        return self.log_p_onestep.dtype

    def add_eps(self, log_probs: torch.Tensor) -> torch.Tensor:
        r'Returns $\log(p + \epsilon)$ from $\log(p)$.'
        if self.eps == 0:
            return log_probs
        log_eps = log_probs.new_tensor(self.eps).log()
        return torch.logaddexp(log_probs, log_eps)

    def extract(
        self, 
        mat_type: Literal['onestep', 'cumulative'], 
        t: torch.Tensor, 
        *,
        row_id: Optional[torch.Tensor] = None,
        column_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''Extracts row/column/element from transition matrix.''' 
        if mat_type not in ("onestep", "cumulative"):
            raise ValueError("mat_type must be 'onestep' or 'cumulative'")

        if row_id is not None and column_id is not None:
            if row_id.shape != column_id.shape:
                raise ValueError('Number of dims for row_id and column_id must be equal!')
            if mat_type  == 'onestep':
                return self.log_p_onestep[row_id, column_id]
            t = broadcast(t, row_id.dim() - 1)
            return self.log_p_cum[t, row_id, column_id]
        
        elif row_id is not None and column_id is None:
            if mat_type == 'onestep':
                return self.log_p_onestep[row_id]
            t = broadcast(t, row_id.dim() - 1)
            return self.log_p_cum[t, row_id, :]

        elif row_id is None and column_id is not None:
            if mat_type == 'onestep':
                return self.log_p_onestep[:, column_id].movedim(0, -1).contiguous()
            t = broadcast(t, column_id.dim() - 1)
            return self.log_p_cum[t, :, column_id]
        
        else:   
            raise ValueError('row_id and column_id cannot be None both!')
        
    def extract_last_cum_matrix(self, x: torch.Tensor) -> torch.Tensor:
        last_timestep = torch.full(
            size=(x.shape[0],), 
            fill_value=self.num_timesteps + 1, 
            device=x.device 
        )
        return self.extract('cumulative', last_timestep, row_id=x)
    
    def sample_bridge(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r'Samples from bridge $p(x_{t} | x_{0}, x_{1})$.'
        log_probs = self.bridge_logits(x_start, x_end, t)
        x_t = gumbel_sample(log_probs, dim=-1, tau=self.tau)

        is_final_step = broadcast(t, x_start.dim() - 1) == self.num_timesteps + 1
        x_t = torch.where(is_final_step, x_end, x_t)

        is_first_step = broadcast(t, x_start.dim() - 1) == 0
        x_t = torch.where(is_first_step, x_start, x_t)

        return x_t

    def bridge_logits(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r'Calculates log probability of $p(x_{t} | x_{0}, x_{1})$.'
        log_p_start_t = self.extract('cumulative', t, row_id=x_start)
        log_p_t_end = self.extract('cumulative', self.num_timesteps + 1 - t, column_id=x_end)
        log_probs = log_p_start_t + log_p_t_end
        return log_probs - log_probs.logsumexp(dim=-1, keepdim=True)

    def posterior_logits(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        logits: bool = False,
    ) -> torch.Tensor:
        r'''Calculates logits of $p(x_{t-1} | x_{t}, x_{0})$.
        If logits is True, the output is summed over x_0 and transition matrix returned.''' 
        if not logits:
            x_start_logits = torch.log(
                torch.nn.functional.one_hot(x_start, self.num_categories)
                + torch.finfo(self.dtype).tiny
            )
        else:
            x_start_logits = x_start.clone()
        assert x_start_logits.shape == x_t.shape + (self.num_categories,), \
            f'x_start_logits.shape: {x_start_logits.shape}, x_t.shape: {x_t.shape}'
        x_start_logits = x_start_logits.to(self.dtype)
        # fact1 is 'guess of x_{t}' from x_{t-1}
        log_fact1 = self.extract('onestep', t, row_id=x_t)

        # fact2 is 'guess of x_{t-1}' from x_{0}
        x_start_logits = x_start_logits.log_softmax(dim=-1)  # bs, ..., num_categories
        is_first_step = broadcast(t, x_t.dim()) == 1
        cumulative_t = torch.where(t == 1, torch.ones_like(t), t - 1)
        log_p_cum = self.log_p_cum[cumulative_t]
        log_fact2 = logits_prod(
            x_start_logits,
            log_p_cum,
            implementation=self.implementation,
            eps=self._reduction_eps(x_start_logits, log_p_cum),
        )

        p_posterior_logits = log_fact1 + log_fact2

        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        p_posterior_logits = torch.where(is_first_step, x_start_logits, p_posterior_logits)
        return p_posterior_logits
    
    def posterior_logits_reverse(
        self,
        x_end: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        logits: bool = False,
    ) -> torch.Tensor:
        r'''Calculates logits of $p(x_{t+1} | x_{t}, x_{1})$.
        If logits is True, the output is summed over x_1 and transition matrix returned.''' 
        if not logits:
            x_end_logits = torch.log(
                torch.nn.functional.one_hot(x_end, self.num_categories)
                + torch.finfo(self.dtype).tiny
            )
        else:
            x_end_logits = x_end.clone()
        assert x_end_logits.shape == x_t.shape + (self.num_categories,), \
            f'x_end_logits.shape: {x_end_logits.shape}, x_t.shape: {x_t.shape}'

        log_fact1 = self.extract('onestep', t+1, row_id=x_t)

        x_end_logits = x_end_logits.log_softmax(dim=-1)
        is_last_step = broadcast(t, x_t.dim()) == self.num_timesteps
        cumulative_t = torch.where(
            t == self.num_timesteps,
            torch.ones_like(t),
            self.num_timesteps - t,
        )
        log_p_cum = self.log_p_cum[cumulative_t]
        log_fact2 = logits_prod(
            x_end_logits,
            log_p_cum,
            implementation=self.implementation,
            eps=self._reduction_eps(x_end_logits, log_p_cum),
        )

        p_posterior_logits = log_fact1 + log_fact2

        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        p_posterior_logits = torch.where(is_last_step, x_end_logits, p_posterior_logits)
        return p_posterior_logits
