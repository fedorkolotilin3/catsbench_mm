import os
from typing import Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, CometLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from .base import BasePlotterCallback
from ..data.codec import BaseCodec
from ..methods import BaseMethod
from ..utils import fig2img


class ImagePlotterCallback(BasePlotterCallback):
    codec: Optional[BaseCodec] = None

    @staticmethod
    def _save_csv_image(
        logger: CSVLogger,
        img,
        stage: str,
        kind: str,
        fb: str,
        epoch: int,
        step: int,
    ) -> None:
        image_dir = os.path.join(logger.log_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{stage}_{kind}_{fb}_epoch_{epoch:03d}_step_{step}.png"
        img.save(os.path.join(image_dir, filename))

    def __init__(
        self,
        num_samples: int,
        num_trajectories: int,
        num_translations: int,
        input_label: str = "Digit 3",
        target_label: str = "Digit 2",
        samples_figsize: Optional[Tuple[int, int]] = None,
        trajectories_figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 100,
        log_zero_fraction: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        self.input_label = input_label
        self.log_zero_fraction = log_zero_fraction
        self.target_label = target_label

        self.samples_fig_config = {
            'figsize': (12, 4) if samples_figsize is None else samples_figsize,
            'dpi': dpi,
        }
        self.trajectories_fig_config = {
            'figsize': (8, 8) if trajectories_figsize is None else trajectories_figsize,
            'dpi': dpi,
        }

    def setup(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: Literal['fit', 'validate', 'test'],
    ) -> None:
        if self.codec is None:
            assert hasattr(trainer.datamodule, 'codec'), \
                'Wrong datamodule! It should have `codec` attribute'
            self.codec = cast(BaseCodec, trainer.datamodule.codec)
        assert self.codec is not None

    @rank_zero_only
    def _log_samples(
        self,
        x_start: torch.Tensor | np.ndarray,
        x_end: torch.Tensor | np.ndarray,
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        raw_x_start: torch.Tensor | np.ndarray | None = None,
        raw_x_end: torch.Tensor | np.ndarray | None = None,
        fb: Literal['forward', 'backward'] = 'forward',
    ):
        assert isinstance(x_start, torch.Tensor) and isinstance(x_end, torch.Tensor)
        assert self.codec is not None
        fb = getattr(pl_module, 'fb', None) or 'forward'
        pred_x_end = pl_module.sample(x_start)

        if self.log_zero_fraction:
            pl_module.logger.log_metrics(
                {
                    f'{stage}/zero_fraction_start_{fb}': (x_start == 0).float().mean(),
                    f'{stage}/zero_fraction_target_{fb}': (x_end == 0).float().mean(),
                    f'{stage}/zero_fraction_generated_{fb}': (
                        pred_x_end == 0
                    ).float().mean(),
                },
                step=pl_module.global_step,
            )

        self.codec.to(x_start.device)
        pred_x_end = self.codec.decode_to_image(pred_x_end).detach().cpu().clamp(0, 1)
        if isinstance(raw_x_start, torch.Tensor) and raw_x_start.is_floating_point():
            x_start = raw_x_start.detach().cpu().clamp(0, 1)
        else:
            x_start = self.codec.decode_to_image(x_start).detach().cpu().clamp(0, 1)
        if isinstance(raw_x_end, torch.Tensor) and raw_x_end.is_floating_point():
            x_end = raw_x_end.detach().cpu().clamp(0, 1)
        else:
            x_end = self.codec.decode_to_image(x_end).detach().cpu().clamp(0, 1)

        nrow = max(1, int(x_start.shape[0] ** 0.5))
        pred_x_end = make_grid(pred_x_end, nrow=nrow).permute(1, 2, 0).numpy()
        x_start = make_grid(x_start, nrow=nrow).permute(1, 2, 0).numpy()
        x_end = make_grid(x_end, nrow=nrow).permute(1, 2, 0).numpy()

        if fb == 'forward':
            labels = (
                self.input_label,
                self.target_label,
                f'Generated {self.target_label}',
            )
        else:
            labels = (
                self.target_label,
                self.input_label,
                f'Generated {self.input_label}',
            )

        fig, axes = plt.subplots(
            1, 3, squeeze=True, sharex=True, sharey=True,
            **self.samples_fig_config,
        )
        suptitle = f"Epoch {pl_module.current_epoch}"
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        fig.suptitle(suptitle)

        axes[0].imshow(x_start)
        axes[1].imshow(x_end)
        axes[2].imshow(pred_x_end)
        for i in range(3):
            axes[i].set_title(labels[i])
            axes[i].get_xaxis().set_ticklabels([])
            axes[i].get_yaxis().set_ticklabels([])
            axes[i].set_axis_off()

        fig.tight_layout(pad=0.5)
        img = fig2img(fig)

        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/samples_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            pl_module.logger.experiment.log_image(
                image_data=img, name=f'{stage}/samples_{fb}', step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, TensorBoardLogger):
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/samples_{fb}',
                img_tensor=np.asarray(img),
                global_step=pl_module.global_step,
                dataformats='HWC',
            )
        elif isinstance(pl_module.logger, CSVLogger):
            self._save_csv_image(
                pl_module.logger, img, stage, 'samples', fb,
                pl_module.current_epoch, pl_module.global_step,
            )
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected '
                'WandbLogger, TensorBoardLogger, CometLogger or CSVLogger.'
            )
        plt.close()

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray,
        x_end: torch.Tensor | np.ndarray,
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        raw_x_start: torch.Tensor | np.ndarray | None = None,
        raw_x_end: torch.Tensor | np.ndarray | None = None,
        fb: Literal['forward', 'backward'] = 'forward',
    ):
        assert isinstance(x_start, torch.Tensor)
        assert self.codec is not None

        fb = getattr(pl_module, 'fb', None) or 'forward'
        traj_start = x_start[:self.num_trajectories]
        repeats = [self.num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *x_start.shape[1:])
        trajectories = pl_module.sample_trajectory(traj_start)

        num_timesteps = trajectories.shape[0]
        trajectories = torch.stack([
            trajectories[0],
            trajectories[num_timesteps // 8],
            trajectories[num_timesteps // 2],
            trajectories[(num_timesteps * 7) // 8],
            trajectories[-1],
        ], dim=0)

        if self.log_zero_fraction:
            zero_fractions = (
                (trajectories == 0).float().flatten(start_dim=1).mean(dim=1)
            )
            pl_module.logger.log_metrics(
                {
                    f'{stage}/trajectory_zero_fraction_{fb}_{i}': value
                    for i, value in enumerate(zero_fractions)
                },
                step=pl_module.global_step,
            )

        self.codec.to(trajectories.device)
        trajectories = self.codec.decode_to_image(
            trajectories.reshape(-1, *trajectories.shape[2:])
        )
        trajectories = trajectories.reshape(
            5, traj_start.shape[0], *trajectories.shape[1:]
        )
        if isinstance(raw_x_start, torch.Tensor) and raw_x_start.is_floating_point():
            raw_traj_start = raw_x_start[:self.num_trajectories]
            repeats = [self.num_translations] + [1] * raw_traj_start.dim()
            raw_traj_start = raw_traj_start.unsqueeze(0).repeat(*repeats)
            trajectories[0] = raw_traj_start.reshape_as(trajectories[0])
        trajectories = trajectories.flatten(end_dim=1).detach().cpu().clamp(0, 1)
        trajectories = make_grid(
            trajectories, nrow=traj_start.shape[0]
        ).permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, 1, **self.trajectories_fig_config)
        suptitle = f"Epoch {pl_module.current_epoch}"
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        fig.suptitle(suptitle)

        ax.imshow(trajectories)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_axis_off()

        fig.tight_layout(pad=0.5)
        img = fig2img(fig)

        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/trajectories_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            pl_module.logger.experiment.log_image(
                image_data=img, name=f'{stage}/trajectories_{fb}', step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, TensorBoardLogger):
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/trajectories_{fb}',
                img_tensor=np.asarray(img),
                global_step=pl_module.global_step,
                dataformats='HWC',
            )
        elif isinstance(pl_module.logger, CSVLogger):
            self._save_csv_image(
                pl_module.logger, img, stage, 'trajectories', fb,
                pl_module.current_epoch, pl_module.global_step,
            )
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected '
                'WandbLogger, TensorBoardLogger, CometLogger or CSVLogger.'
            )
        plt.close()
