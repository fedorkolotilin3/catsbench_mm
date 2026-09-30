from typing import Literal, Optional, Tuple, cast
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, CometLogger, CSVLogger
from lightning.pytorch.utilities import rank_zero_only

from catsbench import BenchmarkHD

from .base import BasePlotterCallback
from ..methods import BaseMethod
from ..utils import convert_to_numpy, fig2img


class BenchmarkHDPlotterCallback(BasePlotterCallback):
    benchmark: Optional[BenchmarkHD] = None

    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_samples: int,
        num_trajectories: int, 
        num_translations: int,
        samples_figsize: Optional[Tuple[int, int]] = None,
        trajectories_figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 100
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        if dim > 2:
            self.pca = PCA(n_components=2)

        self.samples_fig_config = {
            'figsize': (12, 4) if samples_figsize is None else samples_figsize,
            'dpi': dpi,
        }
        self.samples_start_config = {
            'label': r'$p_{start}$', 'c': 'g', 's': 35, 'edgecolor': 'black'
        }
        self.samples_end_config = {
            'label': r'$p_{end}$', 'c': 'orange', 's': 35, 'edgecolor': 'black'
        }
        self.samples_pred_config = {
            'label': r'$p_{\theta}$', 'c': 'yellow', 's': 35, 'edgecolor': 'black'
        }
        self.trajectories_fig_config = {
            'figsize': (12, 6) if trajectories_figsize is None else trajectories_figsize,
            'dpi': dpi,
        }
        self.trajectories_pred_config = {
            'c': 'salmon', 's': 100, 'edgecolors': 'black', 
            'label': 'Fitted distribution', 'zorder': 1, 'linewidth': 0.8
        }
        self.trajectories_start_config = {
            'c': 'lime', 's': 150, 'edgecolors': 'black', 
            'label': r'Trajectory start ($x \sim p_0$)', 'zorder': 3
        }
        self.trajectories_end_config = {
            'c': 'yellow', 's': 80, 'edgecolors': 'black', 
            'label': r'Trajectory end (fitted)', 'zorder': 3
        }
        self.trajectory_lines_config = {
            'back': {'c': 'black', 'markeredgecolor': 'black', 'linewidth': 2, 'zorder': 2},
            'front': {'c': 'grey', 'markeredgecolor': 'black', 'linewidth': 1, 'zorder': 2}
        }

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

    def setup(
        self,
        trainer: Trainer, 
        pl_module: BaseMethod,
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if self.benchmark is None:
            assert hasattr(trainer.datamodule, 'benchmark'), \
                'Wrong datamodule! It should have `benchmark` attribute'
            self.benchmark = cast(BenchmarkHD, trainer.datamodule.benchmark)
        assert isinstance(self.benchmark, BenchmarkHD)

        if self.dim > 2 and not hasattr(self.pca, 'components_'):
            samples = torch.cat(
                self.benchmark.sample_input_target(num_samples=10_000),
                dim=0
            )
            samples = convert_to_numpy(samples)
            self.pca.fit(samples)

    @rank_zero_only
    def _log_samples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        **kwargs,
    ):
        fb = getattr(pl_module, 'fb', None) or 'forward'
        pred_x_end = convert_to_numpy(pl_module.sample(x_start))
        x_start = convert_to_numpy(x_start)
        x_end = convert_to_numpy(x_end)
        if self.dim > 2:
            pred_x_end = self.pca.transform(pred_x_end)
            x_start = self.pca.transform(x_start)
            x_end = self.pca.transform(x_end)

        fig, axes = plt.subplots(1, 3, **self.samples_fig_config)
        suptitle = f"Epoch {pl_module.current_epoch}"
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        fig.suptitle(suptitle)

        axes[0].scatter(x_start[:, 0], x_start[:, 1], **self.samples_start_config) 
        axes[1].scatter(x_end[:, 0], x_end[:, 1], **self.samples_end_config)
        axes[2].scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.samples_pred_config) 
        
        if self.dim > 2:
            max_value = np.abs(x_end).max()
            axlim = [-max_value - 5, max_value + 5]
        else:
            axlim = [0, self.num_categories - 1]
        r = axlim[1] - axlim[0]
        axlim = [axlim[0] + 0.03 * r, axlim[1] - 0.03 * r]
        for i in range(3):
            axes[i].grid()
            axes[i].set_xlim(axlim)
            axes[i].set_ylim(axlim)
            axes[i].legend(loc='lower left')
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
        elif isinstance(pl_module.logger, CSVLogger):
            self._save_csv_image(
                pl_module.logger, img, stage, "samples", fb,
                pl_module.current_epoch, pl_module.global_step,
            )
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, CometLogger or CSVLogger.'
            )
        plt.close()

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray,
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        **kwargs,
    ):
        assert isinstance(self.benchmark, BenchmarkHD)

        fb = getattr(pl_module, 'fb', None) or 'forward' 
        fig, axs = plt.subplots(1, 2, **self.trajectories_fig_config)
        for i in range(2):
            axs[i].get_xaxis().set_ticklabels([])
            axs[i].get_yaxis().set_ticklabels([])

        suptitle = f"Epoch {pl_module.current_epoch}"
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        fig.suptitle(suptitle)
        
        x_end = convert_to_numpy(x_end)
        pred_x_end = convert_to_numpy(pl_module.sample(x_start))
        if self.dim > 2:
            x_end = self.pca.transform(x_end)
            pred_x_end = self.pca.transform(pred_x_end)
        traj_start = x_start[:self.num_trajectories]
        repeats = [self.num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *x_start.shape[1:])

        # ground truth trajectories
        trajectories = self.benchmark.sample_trajectory(
            traj_start, use_onestep_sampling=True
        )
        num_timesteps = trajectories.shape[0]
        if num_timesteps > 17:
            trajectories = torch.stack([
                    trajectories[0], 
                    trajectories[num_timesteps // 4], 
                    trajectories[num_timesteps // 2], 
                    trajectories[(num_timesteps * 3) // 4], 
                    trajectories[-1]
                ], dim=0
            )
        trajectories = convert_to_numpy(trajectories.reshape(-1, self.dim))
        if self.dim > 2:
            trajectories = self.pca.transform(trajectories)
        trajectories = trajectories.reshape(-1, self.num_trajectories * self.num_translations, 2)

        axs[0].scatter(x_end[:, 0], x_end[:, 1], **self.trajectories_pred_config)
        axs[0].scatter(trajectories[0, :, 0], trajectories[0, :, 1], **self.trajectories_start_config)
        axs[0].scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], **self.trajectories_end_config)
        for i in range(self.num_trajectories * self.num_translations):
            axs[0].plot(trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['back'])
            axs[0].plot(
                trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['front'], 
                label='Trajectory (ground truth)' if i == 0 else ''
            )

        # model's trajectories
        pred_trajectories = pl_module.sample_trajectory(traj_start)
        num_timesteps = pred_trajectories.shape[0]
        if num_timesteps > 17:
            pred_trajectories = torch.stack([
                    pred_trajectories[0], 
                    pred_trajectories[num_timesteps // 4], 
                    pred_trajectories[num_timesteps // 2], 
                    pred_trajectories[(num_timesteps * 3) // 4], 
                    pred_trajectories[-1]
                ], dim=0
            )
        pred_trajectories = convert_to_numpy(pred_trajectories.reshape(-1, self.dim))
        if self.dim > 2:
            pred_trajectories = self.pca.transform(pred_trajectories)
        pred_trajectories = pred_trajectories.reshape(-1, self.num_trajectories * self.num_translations, 2)

        axs[1].scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.trajectories_pred_config)
        axs[1].scatter(pred_trajectories[0, :, 0], pred_trajectories[0, :, 1], **self.trajectories_start_config)
        axs[1].scatter(pred_trajectories[-1, :, 0], pred_trajectories[-1, :, 1], **self.trajectories_end_config)
        for i in range(self.num_trajectories * self.num_translations):
            axs[1].plot(pred_trajectories[:, i, 0], pred_trajectories[:, i, 1], **self.trajectory_lines_config['back'])
            axs[1].plot(
                pred_trajectories[:, i, 0], pred_trajectories[:, i, 1], **self.trajectory_lines_config['front'], 
                label='Trajectory (fitted)' if i == 0 else ''
            )
        
        if self.dim > 2:
            max_value = np.abs(x_end).max()
            axlim = [-max_value - 5, max_value + 5]
        else:
            axlim = [0, self.num_categories - 1]
        r = axlim[1] - axlim[0]
        axlim = [axlim[0] + 0.03 * r, axlim[1] - 0.03 * r]
        for i in range(2):
            axs[i].legend(loc='lower left')
            axs[i].set_xlim(axlim)
            axs[i].set_ylim(axlim)
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
        elif isinstance(pl_module.logger, CSVLogger):
            self._save_csv_image(
                pl_module.logger, img, stage, "trajectories", fb,
                pl_module.current_epoch, pl_module.global_step,
            )
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, CometLogger or CSVLogger.'
            )
        plt.close()
