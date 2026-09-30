from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from catsbench.utils import lse_matmul

from .base import BaseMethod
from ..data.batch import Batch
from ..data.prior import Prior
from ..utils import optimize_coupling, gumbel_sample
from ..utils.ranked_logger import RankedLogger


HPARAMS = (
    'dim', 'num_categories', 'num_potentials', 'num_timesteps',
    'sample_prob', 'use_mini_batch', 'distr_init', 'tau',
    'entropy_lambda', 'entropy_warmup_steps', 'kl_loss_coeff', 'mse_loss_coeff', 
    'optimizer', 'scheduler'
)
log = RankedLogger(__name__, rank_zero_only=True)


class DLightSB_M(BaseMethod):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_categories: int,
        num_potentials: int,
        num_timesteps: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        kl_loss_coeff: float = 1.0,
        mse_loss_coeff: float = 0.0,
        use_mini_batch: bool = False,
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian', 
        entropy_lambda: float = 1e-3,
        entropy_warmup_steps: int = 20_000,
        sample_prob: float = 0.9,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.Parameter(torch.empty(dim, num_categories, num_potentials))

        self.bidirectional = False  
        self.iteration = 1
        self._did_weight_init = False
        self._loaded_from_ckpt = False

    @torch.no_grad()
    def init_weights(self, init_samples: Optional[torch.Tensor] = None):
        log.info('Initializing model weights')

        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        if self.hparams.distr_init == 'gaussian':
            cur = (-1.0 + (0.5**2) * torch.randn(
                self.hparams.dim, self.hparams.num_categories, self.hparams.num_potentials, 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype)
            ) / (self.hparams.num_categories * self.hparams.num_potentials)
            cur = torch.log((cur ** 2).clamp_min(1e-12)) # [D, S, K]
            self.log_cp_cores.data.copy_(cur)

        elif self.hparams.distr_init == 'uniform':
            val = math.log(1.0 / (self.hparams.num_categories * self.hparams.num_potentials))
            self.log_cp_cores.data.fill_(val)

        elif self.hparams.distr_init == 'samples':
            assert init_samples is not None, "init_samples should not be None when using benchmark samples"
            init_samples = torch.as_tensor(init_samples, device=self.log_alpha.device)
            assert init_samples.dim() == 2 and init_samples.shape == (self.hparams.num_potentials, self.hparams.dim), \
                f"init_samples must be (num_potentials, dim), got {tuple(init_samples.shape)}"

            base_val = torch.tensor(
                (1 - self.hparams.sample_prob) / (self.hparams.num_categories - 1),
                device=self.log_alpha.device, 
                dtype=self.log_alpha.dtype
            ).log()
            hot_val = torch.tensor(
                math.log(self.hparams.sample_prob), 
                device=self.log_alpha.device, 
                dtype=self.log_alpha.dtype
            )

            cores = torch.full((
                self.hparams.dim, self.hparams.num_categories, self.hparams.num_potentials), base_val, 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            )
            idx = init_samples.t().contiguous().long().unsqueeze(-1) # [D, K, 1]
            idx = idx.permute(0, 2, 1) # [D, 1, K]
            src = torch.full(
                (self.hparams.dim, 1, self.hparams.num_potentials), 
                hot_val, 
                device=self.log_alpha.device, 
                dtype=self.log_alpha.dtype
            )
            cores.scatter_(dim=1, index=idx, src=src)
            self.log_cp_cores.data.copy_(cores)

        else:
            raise ValueError(f"Invalid distr_init: {self.hparams.distr_init}")

        self._did_weight_init = True

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['iteration'] = self.iteration

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        log.info('Loading model weights')

        self._loaded_from_ckpt = True
        if 'iteration' in checkpoint:
            self.iteration = checkpoint['iteration']

        # Legacy format: ParameterList saved as log_cp_cores.0, log_cp_cores.1, ...
        state_dict = checkpoint.get("state_dict", None)
        if state_dict is None:
            return
        legacy_keys = [k for k in state_dict.keys() if k.startswith("log_cp_cores.")]
        if legacy_keys and "log_cp_cores" not in state_dict:
            legacy_keys.sort(key=lambda x: int(x.split(".")[-1]))
            stacked = torch.stack([state_dict[k] for k in legacy_keys], dim=0)
            if stacked.dim() == 3:
                if stacked.shape[1] == self.hparams.num_potentials and stacked.shape[2] == self.hparams.num_categories:
                    stacked = stacked.permute(0, 2, 1).contiguous()
                elif stacked.shape[1] == self.hparams.num_categories and stacked.shape[2] == self.hparams.num_potentials:
                    stacked = stacked.contiguous()
                else:
                    raise RuntimeError(f"Unexpected legacy log_cp_cores shape: {tuple(stacked.shape)}")
            else:
                raise RuntimeError(f"Unexpected legacy log_cp_cores dim: {stacked.dim()}")

            state_dict["log_cp_cores"] = stacked
            for k in legacy_keys:
                del state_dict[k]

    def setup(self, stage: Literal['fit', 'validate', 'test']):
        if stage in (None, "fit") and not self._did_weight_init and not self._loaded_from_ckpt:
            init_samples = None
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'benchmark'):
                benchmark = self.trainer.datamodule.benchmark
                init_samples: torch.Tensor = benchmark.sample_target(self.hparams.num_potentials)
                init_samples = init_samples.flatten(start_dim=1) # (num_potentials, dim)
            self.init_weights(init_samples)

    def kl_loss(
        self,
        true_logits: torch.Tensor, 
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        '''KL-divergence calculation.'''
        pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
        true_log_probs = torch.log_softmax(true_logits, dim=-1)
        return F.kl_div(pred_log_probs, true_log_probs, log_target=True, reduction='batchmean')
        
    def mse_loss(
        self,
        true_logits: torch.Tensor, 
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:        
        '''MSE calculation.'''
        pred_probs = torch.softmax(pred_logits, dim=-1)
        true_probs = torch.softmax(true_logits, dim=-1)
        mse_loss = F.mse_loss(pred_probs, true_probs, reduction='sum')
        return mse_loss / true_probs.shape[0]
    
    def entropy_loss(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=dim)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=dim)

    def _ent_lambda(self):
        frac = min(1.0, self.global_step / self.hparams.entropy_warmup_steps)
        return self.hparams.entropy_lambda * (1.0 - frac)

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
                batch_size, self.hparams.num_categories, self.hparams.num_categories
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
                batch_size, self.hparams.num_categories, self.hparams.num_categories
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

    def get_transition_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t) # [B, D, K]
        log_phi_tp1 = self._log_phi_tp1_all(log_u_t, tp1) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        transition_logits = log_phi_tp1 + onestep  
        return transition_logits.reshape(*input_shape, self.hparams.num_categories) # [B, ..., S]

    def optimal_projection(
        self,
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=0, high=self.hparams.num_timesteps + 1,
            size=(batch_size,), device=self.device
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)

        true_q_posterior_logits = self.prior.posterior_logits_reverse(true_x_end, x_t, t, logits=False)
        pred_p_transition_logits = self.get_transition_logits(x_t, t=t)

        loss, kl, mse  = 0, 0, 0
        posterior_eff_k, prior_eff_k = 0, 0
        if self.hparams.kl_loss_coeff > 0:
            kl = self.kl_loss(true_q_posterior_logits, pred_p_transition_logits)
            loss = loss + self.hparams.kl_loss_coeff * kl
        if self.hparams.mse_loss_coeff > 0:
            mse = self.mse_loss(true_q_posterior_logits, pred_p_transition_logits)
            loss = loss + self.hparams.mse_loss_coeff * mse
        if self._ent_lambda() > 0:
            log_z_t = self._log_u_t(
                x_t.flatten(start_dim=1), 
                self.hparams.num_timesteps + 1 - t
            ).sum(dim=1)
            ent = self.entropy_loss(self.log_alpha[None, :] + log_z_t, dim=-1)
            loss = loss - self._ent_lambda() * ent.mean()

            with torch.no_grad():
                posterior_eff_k = ent.exp().mean()
                prior_eff_k = self.entropy_loss(self.log_alpha, dim=0).exp()

        info = {
            'kl_loss': kl, 'mse_loss': mse, 
            'posterior_eff_k': posterior_eff_k,
            'prior_eff_k': prior_eff_k
        }
        return loss, info

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        x_start, x_end = batch

        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
            output_batch = Batch(
                encoded=(x_start, x_end),
                raw=(None, None),
            )
        else:
            output_batch = batch
        loss, info = self.optimal_projection(x_start, x_end)

        info = {f"train/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': output_batch}

    def on_train_epoch_end(self) -> None:
        self.iteration += 1

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        x_start, x_end = batch
        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)

        loss, info = self.optimal_projection(x_start, x_end)
        info = {f"val/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': batch}
    
    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        x_start, x_end = batch
        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)

        loss, info = self.optimal_projection(x_start, x_end)
        info = {f"test/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': batch}

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer = self.hparams.optimizer(
            params=[self.log_alpha, self.log_cp_cores]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

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
        t = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t) # [B, D, K]
        log_z_t = log_u_t.sum(dim=1) # [B, K]
        log_w_k = self.log_alpha[None, :] + log_z_t # [B, K]
        k_star = gumbel_sample(log_w_k, tau=self.hparams.tau, dim=-1) # [B]

        log_u_tp1_star = self._log_u_tp1_star_all(log_u_t, tp1, k_star) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        logits = onestep + log_u_tp1_star # [B, D, S]
        
        x_tp1 = gumbel_sample(logits, tau=self.hparams.tau, dim=-1).reshape(input_shape) # [B, ...]
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
                fill_value=self.hparams.num_timesteps + 1, 
                device=self.device 
            )
            log_u = self._log_u_t(x_start, last_timestep) # [B, D, K]
            log_z = log_u.sum(dim=1) # [B, K]

            log_w_k = self.log_alpha[None, :] + log_z # [B, K]
            k_star = gumbel_sample(log_w_k, dim=-1, tau=self.hparams.tau) # [B]
            k_idx = k_star.view(
                x_start.shape[0], 1, 1, 1 # [B, 1, 1, 1]
            ).expand(-1, self.hparams.dim, self.hparams.num_categories, 1) # [B, D, S, 1]
            log_cp_cores_star = torch.gather(
                self.log_cp_cores.unsqueeze(0).expand(
                    x_start.shape[0], -1, -1, -1 # [B, D, S, K]
                ), dim=-1, index=k_idx
            ).squeeze(-1) # [B, D, S]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_start) # [B, D, S]
            logits = log_pi_ref + log_cp_cores_star # [B, D, S]
            x_end = gumbel_sample(logits, dim=-1, tau=self.hparams.tau).reshape(input_shape) # [B, ...]
        else:
            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
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
            
            for t in range(1, self.hparams.num_timesteps + 1):
                t = torch.full((x_start.shape[0],), t, device=x_start.device)
                x_t = self.prior.sample_bridge(x_start, x_end, t)
                trajectory.append(x_t)
            trajectory.append(x_end)
            
        else:
            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
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
