from typing import Any, Dict, List, Literal, Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch_ema import ExponentialMovingAverage

from .csbm import CSBM
from .workflows.imf import IMFPhase
from ..data.batch import Batch
from ..data.prior import Prior
from ..utils import optimize_coupling

# NOTE: start and end is swapped because alpha-CSBM uses 
# reverse diffusion notation
class AlphaCSBM(CSBM):
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
        argmax_mode: bool = True,
        tau: float = 1.0,
    ) -> None:
        super().__init__(
            num_timesteps=num_timesteps,
            prior=prior,
            model_forward=model_forward,
            model_backward=model_backward,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            kl_loss_coeff=kl_loss_coeff,
            ce_loss_coeff=ce_loss_coeff,
            mse_loss_coeff=mse_loss_coeff,
            use_mini_batch=use_mini_batch,
            ignore_index=ignore_index,
            accumulate_grad_batches=1,
            argmax_mode=argmax_mode,
            tau=tau,
        )
        self.bidirectional = False
        self.training_phase = IMFPhase(direction='both', iteration=1)
        self.automatic_optimization = True

    @property
    def fb(self) -> Literal['forward']:
        return 'forward'

    @property
    def bf(self) -> Literal['backward']:
        return 'backward'

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        # alpha-CSBM effectively uses only half of the batch size,
        # so we split the batch for cached batch it is already split in workflow
        if batch.cached:
            forward_batch, backward_batch = batch
            pred_x_end, x_start = forward_batch
            x_end, pred_x_start = backward_batch

            loss_forward, info_forward = self.markovian_projection(
                'forward', x_start, pred_x_end
            )
            loss_backward, info_backward = self.markovian_projection(
                'backward', x_end, pred_x_start
            )
            output_batch = Batch(
                encoded=(pred_x_end, x_start),
            )
        else:
            b = batch[0].shape[0] // 2
            x_end, x_start = batch
            raw_x_end, raw_x_start = batch.raw
            output_batch = Batch(
                encoded=(x_end, x_start),
                raw=(raw_x_end, raw_x_start),
            )

            if self.iteration == 1 and self.hparams.use_mini_batch:
                x_start, x_end = optimize_coupling(x_start, x_end)
                output_batch = Batch(encoded=(x_end, x_start))

            if self.iteration == 1:
                loss_forward, info_forward = self.markovian_projection(
                    'forward', x_start[:b], x_end[:b]
                )
                loss_backward, info_backward = self.markovian_projection(
                    'backward', x_end[b:], x_start[b:]
                )
            else:
                pred_x_end = self.sample(x_start[:b], fb='backward')
                pred_x_start = self.sample(x_end[:b], fb='forward')
                loss_forward, info_forward = self.markovian_projection(
                    'forward', x_start[:b], pred_x_end
                )
                loss_backward, info_backward = self.markovian_projection(
                    'backward', x_end[:b], pred_x_start
                )
                output_batch = Batch(
                    encoded=(pred_x_end, x_start[:b]),
                    raw=(None, None if raw_x_start is None else raw_x_start[:b]),
                )

        loss = (loss_forward + loss_backward) / 2

        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"train/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)

        return {'loss': loss, 'batch': output_batch}
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)
        self.emas['forward'].update()
        self.emas['backward'].update()

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        b = batch[0].shape[0] // 2
        x_end, x_start = batch
        output_batch = batch

        # if first iteration apply optional mini-batch sampling
        if self.iteration == 1 and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
            output_batch = Batch(encoded=(x_end, x_start))

        if self.iteration == 1:
            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], x_end[:b])
            loss_backward, info_backward = self.markovian_projection('backward', x_end[b:], x_start[b:])
        else:
            pred_x_end = self.sample(x_start[:b], fb='backward')
            pred_x_start = self.sample(x_end[:b], fb='forward')

            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], pred_x_end)
            loss_backward, info_backward = self.markovian_projection('backward', x_end[:b], pred_x_start)
        
        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"val/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        loss = (loss_forward + loss_backward) / 2
        return {'loss': loss, 'batch': output_batch}

    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        b = batch[0].shape[0] // 2
        x_end, x_start = batch
        output_batch = batch

        # if first iteration apply optional mini-batch sampling
        if self.iteration == 1 and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
            output_batch = Batch(encoded=(x_end, x_start))

        if self.iteration == 1:
            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], x_end[:b])
            loss_backward, info_backward = self.markovian_projection('backward', x_end[b:], x_start[b:])
        else:
            pred_x_end = self.sample(x_start[:b], fb='backward')
            pred_x_start = self.sample(x_end[:b], fb='forward')

            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], pred_x_end)
            loss_backward, info_backward = self.markovian_projection('backward', x_end[:b], pred_x_start)
        
        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"test/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        loss = (loss_forward + loss_backward) / 2
        return {'loss': loss, 'batch': output_batch}

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer  = self.hparams.optimizer(
            params=[
                {"params": self.model_forward.parameters()},
                {"params": self.model_backward.parameters()},
            ]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}
