from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from tqdm.auto import tqdm

from .hub_mixin import BenchmarkModelHubMixin
from ..prior import Prior
from ..utils import (
    continuous_to_discrete, 
    gumbel_sample, 
    Logger,
    lse_matmul
)

log = Logger('catsbench', rank_zero_only=True)
STDS = {2:2, 16:2, 64:2, 3072:15, 4096:15}

@dataclass(kw_only=True)
class BenchmarkBaseConfig:
    dim: int
    input_shape: Tuple[int, ...]
    num_potentials: int
    num_categories: int
    radius: float
    alpha: float
    num_timesteps: int
    num_skip_steps: int
    prior_type: Literal['gaussian', 'uniform']
    benchmark_type: Literal['gaussian', 'uniform']
    num_val_samples: int
    init_batch_size: int
    reverse: bool
    tau: float
    params_dtype: str
    eps: float = 1e-20

class BenchmarkBase(nn.Module, BenchmarkModelHubMixin):

    def __init__(
        self, 
        config: BenchmarkBaseConfig,
        num_timesteps: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.input_shape = config.input_shape
        self.num_potentials = config.num_potentials
        self.num_categories = config.num_categories
        self.radius = config.radius
        self.alpha = config.alpha
        self.prior_type = config.prior_type
        self.benchmark_type = config.benchmark_type
        self.num_val_samples = config.num_val_samples
        self.init_batch_size = config.init_batch_size
        self.reverse = config.reverse
        self.tau = config.tau
        self.params_dtype = getattr(torch, config.params_dtype)
        self.eps = config.eps

        if num_timesteps is not None:
            # check that num_timesteps is compatible config's num_timesteps and num_skip_steps
            total_timesteps = (config.num_timesteps + 1) * config.num_skip_steps
            # the num_timesteps must be devisible by new num_timesteps
            if total_timesteps % (num_timesteps + 1) != 0:
                raise ValueError(
                    f'num_timesteps {num_timesteps} is not compatible with config.num_timesteps '
                    f'{config.num_timesteps} and config.num_skip_steps {config.num_skip_steps}'
                )
            self.num_timesteps = num_timesteps
            self.num_skip_steps = total_timesteps // (num_timesteps + 1)
        else:
            self.num_timesteps = config.num_timesteps
            self.num_skip_steps = config.num_skip_steps

    def register_buffers(
        self, init_benchmark: bool = True, 
        device: Union[str, torch.device] = 'cpu'
    ):
        if init_benchmark:
            log.info('Initializing parameters...')
            log_alpha, log_cp_cores = self._init_parameters(self.radius, device=device)
        else:
            log.info('Skipping parameters initialization!')
            log_alpha = torch.empty(
                (self.num_potentials,), dtype=self.params_dtype, device=device
            )
            log_cp_cores = torch.empty(
                (self.dim, self.num_categories, self.num_potentials), 
                dtype=self.params_dtype, 
                device=device
            )
        self.register_buffer('log_alpha', log_alpha)
        self.register_buffer('log_cp_cores', log_cp_cores)

        log.info('Initializing prior...')
        self.prior = Prior(
            alpha=self.alpha,
            num_categories=self.num_categories,
            num_timesteps=self.num_timesteps,
            num_skip_steps=self.num_skip_steps,
            tau=self.tau,
            eps=self.eps,
            prior_type=self.prior_type,
            implementation="triton",
            dtype=self.params_dtype,
            device=device
        )

        if init_benchmark:
            log.info('Initializing validation dataset...')
            input_dataset, target_dataset = self._init_dataset(
                num_samples=self.num_val_samples,
                batch_size=self.init_batch_size,
            )
        else:
            log.info('Skipping dataset initialization!')
            input_dataset = torch.empty(
                (self.num_val_samples, *self.input_shape), dtype=torch.long, device=device
            )
            target_dataset = torch.empty(
                (self.num_val_samples, *self.input_shape), dtype=torch.long, device=device
            )
        self.register_buffer('input_dataset', input_dataset)
        self.register_buffer('target_dataset', target_dataset)
        
    def _init_parameters(
        self, radius: float, device: Union[str, torch.device] = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize weights uniformly
        log_alpha = torch.log(
            torch.ones(
                self.num_potentials, 
                dtype=self.params_dtype, 
                device=device
            ) / self.num_potentials
        )

        # initialize CP cores
        if self.benchmark_type == 'gaussian':
            # means are sampled from uniformly from sphere of radius 5
            # use rejection sampling to ensure means are well separated
            means_list = []
            min_dist = 2.5 * STDS[self.dim]
            for _ in range(self.num_potentials):
                for _ in range(100):
                    candidate = torch.randn(self.dim, 1, dtype=self.params_dtype, device=device)
                    candidate = radius * candidate / candidate.norm()
                    
                    if not means_list:
                        means_list.append(candidate)
                        break
                        
                    dists = torch.norm(torch.cat(means_list, dim=1) - candidate, dim=0)
                    if (dists > min_dist).all():
                        means_list.append(candidate)
                        break
                else:
                    means_list.append(candidate)
            
            means = torch.cat(means_list, dim=1)
            stds = torch.full(
                (self.dim, self.num_potentials), 
                fill_value=STDS[self.dim], 
                dtype=self.params_dtype,
                device=device
            ).unsqueeze(-1)

            # autmatically scale to the num_categories
            means = continuous_to_discrete(
                means, self.num_categories, quantize_range=(-5,5)
            ).unsqueeze(-1).to(dtype=self.params_dtype)
            log.info(f'Sampled means (indices):\n{means.squeeze(-1).T.cpu().numpy()}')
            
            # compute log probs
            distribution = Normal(means, stds)
            values = torch.arange(
                self.num_categories, device=device
            ).view(1, 1, self.num_categories)
            log_cp_cores: torch.Tensor = distribution.log_prob(values) # (D, K, S)
            log_cp_cores = log_cp_cores.permute(0, 2, 1).contiguous() # (D, S, K)

        elif self.benchmark_type == 'uniform':
            log_cp_cores = torch.rand(
                (self.dim, self.num_categories, self.num_potentials), dtype=self.params_dtype, device=device
            ) # (D, S, K)

        else:
            raise ValueError(f'Unknown benchmark type: {self.benchmark_type}')
        
        return log_alpha, log_cp_cores

    def _init_dataset(
        self,
        num_samples: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dataset = torch.empty((num_samples, *self.input_shape), dtype=torch.long, device=self.device)
        target_dataset = torch.empty((num_samples, *self.input_shape), dtype=torch.long, device=self.device)

        batch_sizes_list = [batch_size] * (num_samples // batch_size)
        if num_samples % batch_size:
            batch_sizes_list.append(num_samples % batch_size)
        batch_sizes_list = tqdm(batch_sizes_list, desc='Initializing dataset')
        for i, bs in enumerate(batch_sizes_list):
            start = i * batch_size
            input_batch = self._sample_input(bs)
            target_batch = self.sample(input_batch)
            input_dataset[start:start+bs] = input_batch
            target_dataset[start:start+bs] = target_batch

        random_indices = torch.randperm(len(target_dataset))
        input_dataset  = input_dataset[random_indices]
        target_dataset = target_dataset[random_indices]
        if self.reverse:
            input_dataset, target_dataset = target_dataset, input_dataset

        return input_dataset, target_dataset

    @property
    def name(self) -> str:
        return f'd{self.dim}_s{self.num_categories}_prior_{self.prior.prior_type}_a{self.prior.alpha}'

    @property
    def device(self) -> str:
        return self.log_cp_cores.device
    
    def dtype(self) -> torch.dtype:
        return self.log_cp_cores.dtype

    def _log_u_t(
        self, 
        x_t: torch.Tensor, # [B, D]
        t: torch.Tensor
    ) -> torch.Tensor:
        log_pi_ref_t = self.prior.extract("cumulative", t, row_id=x_t) # [B, D, S]
        log_u_t = lse_matmul( 
            log_pi_ref_t.unsqueeze(-2), # [B, D, 1, S]
            self.log_cp_cores.unsqueeze(0), # [1, D, S, K]
        ).squeeze(-2) # [B, D, K]
        return log_u_t # [B, D, K]

    def _log_phi_tp1_all(
        self, 
        log_u_t: torch.Tensor,
        tp1: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = log_u_t.shape[0]
        if tp1.numel() == 1 or bool((tp1 == tp1.flatten()[0]).all()):
            # all values are the save
            tp1_val = int(tp1.flatten()[0].item())
            log_pi_ref_tp1 = self.prior.log_p_cum[tp1_val].unsqueeze(0).expand(
                batch_size, self.num_categories, self.num_categories
            ) # [B, S_next, S_end]
        else:
            # values are the different
            log_pi_ref_tp1 = self.prior.log_p_cum.index_select(0, tp1)  # [B, S_next, S_end]
        
        log_z_t = log_u_t.sum(dim=1) # [B, K]
        term = log_z_t[:, None, :] - log_u_t
        w = term + self.log_alpha.view(1, 1, -1)
        v = lse_matmul(
            self.log_cp_cores.unsqueeze(0), # [1, D, S_end, K]
            w.unsqueeze(-1), # [B, D, K, 1]
        ).squeeze(-1) # [B, D, S_end]
        log_phi_tp1 = lse_matmul(
            log_pi_ref_tp1.unsqueeze(1), # [B, 1, S_next, S_end]
            v.unsqueeze(-1), # [B, D, S_end, 1]
        ).squeeze(-1) # [B, D, S_next]
        return log_phi_tp1

    def _log_u_tp1_star_all(
        self, 
        log_u_t: torch.Tensor,
        tp1: torch.Tensor,
        k_star: torch.Tensor
    ) -> torch.Tensor:
        batch_size = log_u_t.shape[0]
        if tp1.numel() == 1 or bool((tp1 == tp1.flatten()[0]).all()):
            # all values are the save
            tp1_val = int(tp1.flatten()[0].item())
            log_pi_ref_tp1 = self.prior.log_p_cum[tp1_val].unsqueeze(0).expand(
                batch_size, self.num_categories, self.num_categories
            ) # [B, S_next, S_end]
        else:
            # values are the different
            log_pi_ref_tp1 = self.prior.log_p_cum.index_select(0, tp1)  # [B, S_next, S_end]

        log_cp_cores = self.log_cp_cores.movedim(-1, 0).contiguous() # [K, D, S]
        log_cp_cores_star = log_cp_cores.index_select(0, k_star)

        log_u_tp1_star = lse_matmul(
            log_pi_ref_tp1.unsqueeze(1), # [B, 1, S_next, S_end]
            log_cp_cores_star.unsqueeze(-1), # [B, D, S_end, 1]
        ).squeeze(-1) # [B, D, S_next]
        return log_u_tp1_star

    @torch.no_grad()
    def get_transition_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.num_timesteps + 1 - t_orig
        tp1 = self.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t) # [B, D, K]
        log_phi_tp1 = self._log_phi_tp1_all(log_u_t, tp1) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        transition_logits = log_phi_tp1 + onestep  
        return transition_logits.reshape(*input_shape, self.num_categories) # [B, ..., S]
 
    @torch.no_grad()
    def get_cum_transition_logits(self, x_start: torch.Tensor) -> torch.Tensor:
        input_shape = x_start.shape
        x_start = x_start.flatten(start_dim=1) # [B, D]

        last_timestep = torch.full(
            size=(x_start.shape[0],),
            fill_value=self.num_timesteps + 1,
            device=self.device,
            dtype=torch.long,
        )
        log_u = self._log_u_t(x_start, last_timestep) # [B, D, K]
        log_z = log_u.sum(dim=1) # [B, K]
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=-1) # [B]

        log_pi_ref = self.prior.extract_last_cum_matrix(x_start) # [B, D, S]
        log_p_k = self.log_alpha[None, :] + log_z - log_c[:, None]
        log_mix = lse_matmul(
            self.log_cp_cores.unsqueeze(0), # [1, D, S, K], 
            (log_p_k[:, None, :] - log_u).unsqueeze(-1), # [B, D, K, 1], 
        ).squeeze(-1)
        logits = log_pi_ref + log_mix # [B, ..., S]
        return logits.reshape(*input_shape, self.num_categories) # [B, ..., S]

    @torch.no_grad()
    def markov_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        return_transitions: bool = False
    ) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)

        t_orig = t  # keep original for onestep
        t = self.num_timesteps + 1 - t_orig
        tp1 = self.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t) # [B, D, K]
        log_z_t = log_u_t.sum(dim=1) # [B, K]
        log_w_k = self.log_alpha[None, :] + log_z_t # [B, K]
        k_star = gumbel_sample(log_w_k, tau=self.tau, dim=-1) # [B]

        log_u_tp1_star = self._log_u_tp1_star_all(log_u_t, tp1, k_star) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        logits = onestep + log_u_tp1_star # [B, D, S]
        
        x_tp1 = gumbel_sample(logits, tau=self.tau, dim=-1).reshape(input_shape) # [B, ...]
        if return_transitions:
            log_phi_tp1 = self._log_phi_tp1_all(log_u_t, tp1) # [B, D, S]
            transition_logits = log_phi_tp1 + onestep # [B, D, S]
            return x_tp1, transition_logits
        return x_tp1

    @torch.no_grad()
    def sample(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = True
    ) -> torch.Tensor:
        '''Sample from the model starting from `x` returning the final sample.'''
        if use_onestep_sampling:
            input_shape = x_start.shape
            x_start = x_start.flatten(start_dim=1) # (B, D)

            last_timestep = torch.full(
                size=(x_start.shape[0],), 
                fill_value=self.num_timesteps + 1, 
                device=self.device 
            )
            log_u = self._log_u_t(x_start, last_timestep) # [B, D, K]
            log_z = log_u.sum(dim=1) # [B, K]

            log_w_k = self.log_alpha[None, :] + log_z # [B, K]
            k_star = gumbel_sample(log_w_k, dim=-1, tau=self.tau) # [B]
            k_idx = k_star.view(
                x_start.shape[0], 1, 1, 1 # [B, 1, 1, 1]
            ).expand(-1, self.dim, self.num_categories, 1) # [B, D, S, 1]
            log_cp_cores_star = torch.gather(
                self.log_cp_cores.unsqueeze(0).expand(
                    x_start.shape[0], -1, -1, -1 # [B, D, S, K]
                ), dim=-1, index=k_idx
            ).squeeze(-1) # [B, D, S]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_start) # [B, D, S]
            logits = log_pi_ref + log_cp_cores_star # [B, D, S]
            x_end = gumbel_sample(logits, dim=-1, tau=self.tau).reshape(input_shape) # [B, ...]
        else:
            x_t = x_start
            for t in range(0, self.num_timesteps + 1):
                t = torch.full([x_start.shape[0]], t, device=self.device)
                x_t = self.markov_sample(x_t, t, return_transitions=False)
            x_end = x_t
        return x_end
    
    @torch.no_grad()
    def sample_trajectory(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = False,
        return_transitions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_onestep_sampling:
            assert not return_transitions, \
                'Returning transitions is not supported when using bridge samples!'
        
        trajectory, transitions = [x_start], []
        if use_onestep_sampling:
            x_end = self.sample(x_start, use_onestep_sampling=True)
            
            for t in range(1, self.num_timesteps + 1):
                t = torch.full((x_start.shape[0],), t, device=x_start.device)
                x_t = self.prior.sample_bridge(x_start, x_end, t)
                trajectory.append(x_t)
            trajectory.append(x_end)
            
        else:
            x_t = x_start
            for t in range(0, self.num_timesteps + 1):
                t = torch.full([x_t.shape[0]], t, device=self.device)
                out = self.markov_sample(x_t, t, return_transitions=return_transitions)
                if return_transitions:
                    x_t, logits = out
                    transitions.append(logits)
                else:
                    x_t = out
                trajectory.append(x_t)

        trajectory = torch.stack(trajectory, dim=0)
        if return_transitions:
            transitions = torch.stack(transitions, dim=0)
            return trajectory, transitions
        return trajectory

    def _sample_input(self, num_samples: int) -> torch.LongTensor:
        raise NotImplementedError

    @torch.no_grad()
    def _sample_target(self, num_samples: int) -> torch.LongTensor:
        input_samples = self._sample_input(num_samples)
        target_samples = self.sample(input_samples)
        return target_samples

    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.LongTensor:
        if self.reverse:
            return self._sample_target(num_samples)
        return self._sample_input(num_samples)
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.LongTensor:
        if self.reverse:
            return self._sample_input(num_samples)
        return self._sample_target(num_samples)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        '''Sample paired input and target data'''
        input_samples = self._sample_input(num_samples)
        target_samples = self.sample(input_samples)
        if self.reverse:
            return target_samples, input_samples
        return input_samples, target_samples

    def _plot_samples(self, num_samples: int):
        raise NotImplementedError
    
    def _plot_trajectories(
        self, 
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int
    ):
        raise NotImplementedError

    def plot(
        self, 
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int
    ):
        self._plot_samples(num_samples)
        if not self.reverse:
            self._plot_trajectories(num_samples, num_trajectories, num_translations)
