from typing import Literal, Optional
import torch

from torchmetrics import MetricCollection
from lightning.pytorch import Trainer

from catsbench import BenchmarkHD
from catsbench.metrics import (
    ClassifierTwoSampleTest,
    ShapeScore,
    TrendScore,
    TrajectoryKLDivergence,
)

from .base import BaseMetricsCallback
from ..data.batch import Batch
from ..methods import BaseMethod, CSBM, AlphaCSBM
from ..utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class BenchmarkHDMetricsCallback(BaseMetricsCallback):
    benchmark: Optional[BenchmarkHD] = None

    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_cond_samples: int,
        num_timesteps: int,
        train_test_split: Optional[float] = 0.8,
        classifier_lr: Optional[float] = 1e-2,
        adjusted_tv: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps

        self.num_cond_samples = num_cond_samples
        self.train_test_split = train_test_split
        self.classifier_lr = classifier_lr
        self.adjusted_tv = adjusted_tv

        self.metrics: Optional[MetricCollection] = None
        self.cond_metrics: Optional[MetricCollection] = None
        self.c2st: Optional[ClassifierTwoSampleTest] = None
        self.forward_kl_div: Optional[TrajectoryKLDivergence] = None
        self.reverse_kl_div: Optional[TrajectoryKLDivergence] = None

    def _setup_callback(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: Literal['fit', 'validate', 'test'],
    ) -> None:
        if self.benchmark is None:
            assert hasattr(trainer.datamodule, 'benchmark'), \
                'Wrong datamodule! It should have `benchmark` attribute'
            self.benchmark: BenchmarkHD = trainer.datamodule.benchmark
        assert isinstance(self.benchmark, BenchmarkHD)

        # Keep metrics on the callback so their modules and state are not included
        # in the benchmark method's checkpoints.
        if self.metrics is None:
            self.metrics = MetricCollection(
                {
                    'shape_score': ShapeScore(
                        self.dim, self.num_categories, conditional=False, adjusted=self.adjusted_tv
                    ),
                    'trend_score': TrendScore(
                        self.dim, self.num_categories, conditional=False, adjusted=self.adjusted_tv
                    ),
                },
            )

        # initialize conditional metrics
        if self.benchmark.reverse:
            if self.c2st is None:
                self.c2st = ClassifierTwoSampleTest(
                    dim=2*self.dim, num_categories=self.num_categories, lr=self.classifier_lr
                )
        else:
            if self.cond_metrics is None:
                self.cond_metrics = MetricCollection(
                    {
                        'cond_shape_score': ShapeScore(
                            self.dim, self.num_categories, conditional=True, adjusted=self.adjusted_tv
                        ),
                        'cond_trend_score': TrendScore(
                            self.dim, self.num_categories, conditional=True, adjusted=self.adjusted_tv
                        ),
                    },
                )
            if callable(getattr(pl_module, 'get_transition_logits', None)):
                if self.forward_kl_div is None:
                    self.forward_kl_div = TrajectoryKLDivergence(
                        dim=self.dim,
                        num_timesteps=self.num_timesteps,
                        logits=True,
                    )
                if self.reverse_kl_div is None:
                    self.reverse_kl_div = TrajectoryKLDivergence(
                        dim=self.dim,
                        num_timesteps=self.num_timesteps,
                        logits=True,
                    )

        callback_metrics = (
            self.metrics,
            self.c2st,
            self.cond_metrics,
            self.forward_kl_div,
            self.reverse_kl_div,
        )
        for metric in callback_metrics:
            if metric is not None:
                metric.to(pl_module.device)

    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        batch: Batch,
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        assert isinstance(self.benchmark, BenchmarkHD)
        assert self.metrics is not None

        x_start, x_end = batch.encoded

        # update unconditional metrics
        pred_x_end = pl_module.sample(x_start)
        self.metrics.update(x_end, pred_x_end)

        # update conditional metrics
        if self.benchmark.reverse:
            assert self.c2st is not None
            loader_attr = "train_dataloader" if stage == "train" else f"{stage}_dataloaders"
            limit = getattr(trainer, f"limit_{stage}_batches")
            loader = getattr(trainer, loader_attr)
            num_batches = limit if limit is not None else len(loader)
            train_mode = batch_idx < int(num_batches * self.train_test_split)

            self.c2st.update(
                real_data=torch.cat([x_start, x_end], dim=-1),
                pred_data=torch.cat([x_start, pred_x_end], dim=-1),
                train=train_mode
            )
        else:
            assert self.cond_metrics is not None
            repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
            cond_x_end = self.benchmark.sample(repeated_x_start)
            cond_pred_x_end = pl_module.sample(repeated_x_start)
            self.cond_metrics.update(cond_x_end, cond_pred_x_end)

            if not callable(getattr(pl_module, 'get_transition_logits', None)):
                return
            assert self.forward_kl_div is not None
            assert self.reverse_kl_div is not None

            true_trajectory, true_transition_logits = self.benchmark.sample_trajectory(x_start, return_transitions=True)
            pred_trajectory, pred_transition_logits = pl_module.sample_trajectory(x_start, return_transitions=True)
            
            # we need only num_steps + 1 points to compute transitions
            true_trajectory = true_trajectory[:-1]
            pred_trajectory = pred_trajectory[:-1]
            
            timesteps = torch.arange(true_trajectory.shape[0], device=pl_module.device)
            timesteps = timesteps.repeat_interleave(true_trajectory.shape[1])
            
            true_trajectory = true_trajectory.flatten(end_dim=1)
            pred_trajectory = pred_trajectory.flatten(end_dim=1)
            true_transition_logits = true_transition_logits.flatten(end_dim=1)
            pred_transition_logits = pred_transition_logits.flatten(end_dim=1)

            # the KL div must be computed in cross fashion:
            # forward KL is KL with respect to true trajectory
            # reverse KL is KL with respect to predicted trajectory
            self.reverse_kl_div.update(
                p=pred_transition_logits, 
                q=self.benchmark.get_transition_logits(pred_trajectory, timesteps)
            )
            if isinstance(pl_module, (CSBM, AlphaCSBM)):
                timesteps = (pl_module.prior.num_timesteps + 1) - timesteps
            
            with torch.no_grad(): # remove grads from transitions of DLightSB methods
                self.forward_kl_div.update(
                    p=true_transition_logits, 
                    q=pl_module.get_transition_logits(true_trajectory, timesteps)
                )
            
    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        assert isinstance(self.benchmark, BenchmarkHD)
        assert self.metrics is not None

        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        # compute and log unconditional metrics
        metrics = self.metrics.compute()
        metrics = {f'{stage}/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        self.metrics.reset()

        # compute and log conditional metrics
        if self.benchmark.reverse:
            assert self.c2st is not None
            c2st = self.c2st.compute()
            pl_module.log(f'{stage}/c2st_{fb}', c2st)
            self.c2st.reset()
        else:
            assert self.cond_metrics is not None
            cond_metrics = self.cond_metrics.compute()
            cond_metrics = {f'{stage}/{k}_{fb}': v for k, v in cond_metrics.items()}
            pl_module.log_dict(cond_metrics)
            self.cond_metrics.reset()

            if not callable(getattr(pl_module, 'get_transition_logits', None)):
                return
            assert self.forward_kl_div is not None
            assert self.reverse_kl_div is not None

            forward_kl_div = self.forward_kl_div.compute()
            pl_module.log(f'{stage}/forward_kl_div_{fb}', forward_kl_div)
            self.forward_kl_div.reset()

            reverse_kl_div = self.reverse_kl_div.compute()
            pl_module.log(f'{stage}/reverse_kl_div_{fb}', reverse_kl_div)
            self.reverse_kl_div.reset()
