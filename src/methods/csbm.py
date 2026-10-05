from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch_ema import ExponentialMovingAverage 

from .base import BaseMethod
from .workflows.imf import IMFPhase
from ..data.batch import Batch
from ..data.prior import Prior
from ..utils import optimize_coupling, gumbel_sample


HPARAMS = (
    'num_timesteps', 'kl_loss_coeff', 'ce_loss_coeff', 'mse_loss_coeff', 
    'use_mini_batch', 'ignore_index', 'accumulate_grad_batches',
    'optimizer', 'scheduler', 'argmax_mode', 'tau'
)

# NOTE: start and end is swapped because CSBM uses 
# reverse diffusion notation
class CSBM(BaseMethod):
    def __init__(
        self,
        num_timesteps: int,
        prior: Prior,
        model_forward: nn.Module,
        model_backward: nn.Module,
        ema: ExponentialMovingAverage, # partially initialized
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        kl_loss_coeff: float = 1.0,
        ce_loss_coeff: float = 0.001,
        mse_loss_coeff: float = 0.0,
        use_mini_batch: bool = False,
        ignore_index: int = -100,
        accumulate_grad_batches: int =1,
        argmax_mode: bool = True,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)
        self.bidirectional = True
        self.training_phase = IMFPhase(direction='forward', iteration=1)
        self.prior = prior
        
        # models explicitly stated to be able log parameters
        self.model_forward = model_forward
        self.model_backward = model_backward
        self.models = {
            'forward': self.model_forward,
            'backward': self.model_backward,
        }
        self.emas: Dict[str, ExponentialMovingAverage] = {
            'forward': ema(self.models['forward'].parameters()),
            'backward': ema(self.models['backward'].parameters())
        }
        self._ema_active_directions = set()
        self._ema_training_modes = {}
    
        self.automatic_optimization = False

    @property
    def fb(self) -> Literal['forward', 'backward']:
        return self.training_phase.direction

    @property
    def bf(self) -> Literal['forward', 'backward']:
        return self.training_phase.opposite_direction

    @property
    def iteration(self) -> int:
        return self.training_phase.iteration

    def kl_loss(
        self,
        true_logits: torch.Tensor, 
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        '''KL-divergence calculation.'''
        pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
        true_log_probs = torch.log_softmax(true_logits, dim=-1)
        kl_loss = F.kl_div(pred_log_probs, true_log_probs, log_target=True, reduction='none')
        kl_loss = kl_loss.sum(dim=-1).mean()
        return kl_loss
        
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

    def ce_loss(
        self,
        true_x_start: torch.Tensor, 
        pred_x_start_logits: torch.Tensor, 
    ) -> torch.Tensor:   
        '''CE calculation.'''         
        pred_x_start_logits = pred_x_start_logits.flatten(start_dim=0, end_dim=-2)
        true_x_start = true_x_start.flatten(start_dim=0, end_dim=-1)
        ce_loss = F.cross_entropy(pred_x_start_logits, true_x_start, ignore_index=self.hparams.ignore_index)
        return ce_loss

    # EMA does not inherit nn.Module so it must be 
    # putted to device manually 
    def setup(self, stage: Literal['fit', 'validate', 'test']) -> None:
        self.emas['forward'].to(self.device)
        self.emas['backward'].to(self.device)

    def markovian_projection(
        self,
        fb: Literal['forward', 'backward'],
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=1, high=self.hparams.num_timesteps + 2,
            size=(batch_size,), device=self.device
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)

        pred_x_start_logits = self.models[fb](x_t, t)
        true_q_posterior_logits = self.prior.posterior_logits(true_x_start, x_t, t, logits=False)
        pred_p_transition_logits = self.prior.posterior_logits(pred_x_start_logits, x_t, t, logits=True)

        loss, kl, ce, mse = 0, 0, 0, 0
        if self.hparams.kl_loss_coeff > 0:
            kl = self.kl_loss(true_q_posterior_logits, pred_p_transition_logits)
            loss += self.hparams.kl_loss_coeff * kl
        if self.hparams.ce_loss_coeff > 0:
            ce = self.ce_loss(true_x_start, pred_x_start_logits)
            loss += self.hparams.ce_loss_coeff * ce
        if self.hparams.mse_loss_coeff > 0:
            mse = self.mse_loss(true_q_posterior_logits, pred_p_transition_logits)
            loss += self.hparams.mse_loss_coeff * mse
        
        info = {f'kl_loss_{fb}': kl, f'ce_loss_{fb}': ce, f'mse_loss_{fb}': mse}
        return loss, info

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        self.models[self.fb].train()
        self.models[self.bf].eval()
        
        # DataLoader have the x_0, x_1 order so we need to swap
        if self.fb == 'forward':
            x_end, x_start = batch
            raw_x_end, raw_x_start = batch.raw
        else:
            x_start, x_end = batch
            raw_x_start, raw_x_end = batch.raw

        output_raw = (None, None)
        if self.iteration > 1 and not batch.cached:
            x_end = self.sample(x_start, fb=self.bf)
            output_raw = (None, raw_x_start)
        elif self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        else:
            output_raw = (raw_x_end, raw_x_start)
        output_batch = Batch(encoded=(x_end, x_start), raw=output_raw)
        optimizers = {
            'forward': self.optimizers()[0],
            'backward': self.optimizers()[1],
        }
        if self.lr_schedulers():
            schedulers = {
                'forward': self.lr_schedulers()[0],
                'backward': self.lr_schedulers()[1],
            }
        with self.toggled_optimizer(optimizers[self.fb]):
            loss, info = self.markovian_projection(self.fb, x_start, x_end)
            # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
            info = {f"train/{k}": v for k, v in info.items()}
            self.log_dict(info, prog_bar=True, sync_dist=True) 
            self.log('train/iteration', self.iteration, prog_bar=True)

            step_loss = loss.detach()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            # do gradient accumulation and clipping
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                self.clip_gradients(
                    optimizers[self.fb], gradient_clip_val=self.trainer.gradient_clip_val
                )
                optimizers[self.fb].step()
                if self.lr_schedulers():
                    # pass loss in case scheduler requires metrics like ReduceLROnPlateau
                    schedulers[self.fb].step(loss.detach()) 
                optimizers[self.fb].zero_grad()
                self.emas[self.fb].update()
        return {'loss': step_loss, 'batch': output_batch}

    def on_validation_start(self) -> None:
        for direction in self.models:
            if direction in self._ema_active_directions:
                continue
            model = self.models[direction]
            self._ema_training_modes[direction] = model.training
            self.emas[direction].store()
            self.emas[direction].copy_to()
            self._ema_active_directions.add(direction)
            model.eval()

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        # DataLoader have the x_0, x_1 order so we need to swap
        if self.fb == 'forward':
            x_end, x_start = batch
            raw_x_end, raw_x_start = batch.raw
        else:
            x_start, x_end = batch
            raw_x_start, raw_x_end = batch.raw
        output_batch = Batch(
            encoded=(x_end, x_start),
            raw=(raw_x_end, raw_x_start),
        )
        # if first iteration apply optional mini-batch sampling
        if self.iteration == 1 and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if self.iteration > 1:
            x_start, x_end = x_start, self.sample(x_start, fb=self.bf)

        loss, info = self.markovian_projection(self.fb, x_start, x_end)
        info = {f"val/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': output_batch}

    def on_validation_end(self) -> None:
        for direction in self.models:
            if direction not in self._ema_active_directions:
                continue
            self.emas[direction].restore()
            self._ema_active_directions.remove(direction)
            self.models[direction].train(
                self._ema_training_modes.pop(direction)
            )

    def on_test_start(self) -> None:
        for direction in self.models:
            if direction in self._ema_active_directions:
                continue
            model = self.models[direction]
            self._ema_training_modes[direction] = model.training
            self.emas[direction].store()
            self.emas[direction].copy_to()
            self._ema_active_directions.add(direction)
            model.eval()

    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        # DataLoader have the x_0, x_1 order so we need to swap
        if self.fb == 'forward':
            x_end, x_start = batch
            raw_x_end, raw_x_start = batch.raw
        else:
            x_start, x_end = batch
            raw_x_start, raw_x_end = batch.raw
        output_batch = Batch(
            encoded=(x_end, x_start),
            raw=(raw_x_end, raw_x_start),
        )
        # if first iteration apply optional mini-batch sampling
        if self.iteration == 1 and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if self.iteration > 1:
            x_start, x_end = x_start, self.sample(x_start, fb=self.bf)

        loss, info = self.markovian_projection(self.fb, x_start, x_end)
        info = {f"test/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': output_batch}

    def on_test_end(self) -> None:
        for direction in self.models:
            if direction not in self._ema_active_directions:
                continue
            self.emas[direction].restore()
            self._ema_active_directions.remove(direction)
            self.models[direction].train(
                self._ema_training_modes.pop(direction)
            )

    def on_predict_start(self) -> None:
        sampling_direction = self.training_phase.opposite_direction
        directions = self.models if sampling_direction == 'both' else (sampling_direction,)
        for direction in directions:
            if direction in self._ema_active_directions:
                continue
            model = self.models[direction]
            self._ema_training_modes[direction] = model.training
            self.emas[direction].store()
            self.emas[direction].copy_to()
            self._ema_active_directions.add(direction)
            model.eval()

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        sampling_direction = self.training_phase.opposite_direction
        if sampling_direction == 'both':
            x_end, x_start = batch
            return (
                self.sample(x_start, fb='backward'),
                self.sample(x_end, fb='forward'),
            )

        # input is always in the batch[0]
        # contract with datamodule classes
        if batch[0] is None:
            raise ValueError("Predict batch does not contain an input tensor.")
        return self.sample(batch[0], fb=sampling_direction)

    def on_predict_end(self) -> None:
        for direction in self.models:
            if direction not in self._ema_active_directions:
                continue
            self.emas[direction].restore()
            self._ema_active_directions.remove(direction)
            self.models[direction].train(
                self._ema_training_modes.pop(direction)
            )

    def teardown(self, stage: str) -> None:
        for direction in self.models:
            if direction not in self._ema_active_directions:
                continue
            self.emas[direction].restore()
            self._ema_active_directions.remove(direction)
            self.models[direction].train(
                self._ema_training_modes.pop(direction)
            )

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer_forward  = self.hparams.optimizer(params=self.models['forward'].parameters())
        optimizer_backward  = self.hparams.optimizer(params=self.models['backward'].parameters())

        if self.hparams.scheduler is not None:
            scheduler_forward = self.hparams.scheduler(optimizer=optimizer_forward)
            scheduler_backward = self.hparams.scheduler(optimizer=optimizer_backward)
            return [
                {'optimizer': optimizer_forward, 'lr_scheduler': scheduler_forward},
                {'optimizer': optimizer_backward, 'lr_scheduler': scheduler_backward}
            ]
        return [{'optimizer': optimizer_forward}, {'optimizer': optimizer_backward}]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['ema_forward'] = self.emas['forward'].state_dict()
        checkpoint['ema_backward'] = self.emas['backward'].state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'ema_forward' in checkpoint:
            self.emas['forward'].load_state_dict(checkpoint['ema_forward'])
            self.emas['forward'].to(self.device)
        if 'ema_backward' in checkpoint:
            self.emas['backward'].load_state_dict(checkpoint['ema_backward'])
            self.emas['backward'].to(self.device)

    @torch.no_grad()
    def get_transition_logits(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        fb: Optional[Literal['forward', 'backward']] = None
    ) -> torch.Tensor:
        fb = fb or self.fb
        was_training = self.models[fb].training
        self.models[fb].eval()

        # if ema_is_active don't override params each batch
        ema_is_active = fb in self._ema_active_directions
        ema_context = (
            nullcontext()
            if ema_is_active
            else self.emas[fb].average_parameters()
        )
        with ema_context:
            pred_x_start_logits = self.models[fb](x_t, t)

        if was_training: 
            self.models[fb].train()
            
        pred_transition_logits = self.prior.posterior_logits(pred_x_start_logits, x_t, t, logits=True)
        return pred_transition_logits

    def markov_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        fb: Literal['forward', 'backward'], 
        return_transitions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # we don't use here get_transition_logits avoid multiple EMA calls     
        pred_x_start_logits = self.models[fb](x_t, t)
        pred_transition_logits = self.prior.posterior_logits(pred_x_start_logits, x_t, t, logits=True)
        samples = gumbel_sample(
            pred_transition_logits, tau=self.hparams.tau, dim=-1
        )

        if self.hparams.argmax_mode:
            first_step = (t == 1).view((x_t.shape[0], *[1] * (x_t.dim() - 1)))        
            argmax_samples = pred_transition_logits.argmax(dim=-1)
            samples = torch.where(first_step, argmax_samples, samples)

        if return_transitions:
            return samples, pred_transition_logits
        return samples
        
    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, fb: Optional[Literal['forward', 'backward']] = None
    ) -> torch.Tensor:
        """Sample from the model starting from `x` returning the final sample."""
        fb = fb or self.fb

        was_training = self.models[fb].training
        self.models[fb].eval()

        # if ema_is_active don't override params each batch
        ema_is_active = fb in self._ema_active_directions
        ema_context = (
            nullcontext()
            if ema_is_active
            else self.emas[fb].average_parameters()
        )
        with ema_context:
            for t in reversed(range(1, self.hparams.num_timesteps + 2)):
                t = torch.full([x.shape[0]], t, device=self.device)
                x = self.markov_sample(x, t, fb, return_transitions=False)
        if was_training: 
            self.models[fb].train()
        return x
    
    @torch.no_grad()
    def sample_trajectory(
        self, x: torch.Tensor, 
        fb: Optional[Literal['forward', 'backward']] = None,
        return_transitions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample from the model starting from `x` returning the full trajectory."""
        fb = fb or self.fb

        trajectory, transitions = [x], []
        
        was_training = self.models[fb].training
        self.models[fb].eval()

        # if ema_is_active don't override params each batch
        ema_is_active = fb in self._ema_active_directions
        ema_context = (
            nullcontext()
            if ema_is_active
            else self.emas[fb].average_parameters()
        )
        with ema_context:
            for t in reversed(range(1, self.hparams.num_timesteps + 2)):
                t = torch.full([x.shape[0]], t, device=self.device)
                out = self.markov_sample(x, t, fb, return_transitions=return_transitions)
                if return_transitions:
                    x, logits = out
                    transitions.append(logits)
                else:
                    x = out
                trajectory.append(x)
        if was_training: 
            self.models[fb].train()
        
        trajectory = torch.stack(trajectory, dim=0)
        if return_transitions:
            transitions = torch.stack(transitions, dim=0)
            return trajectory, transitions
        return trajectory
