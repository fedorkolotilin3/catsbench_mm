from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from .base import BenchmarkBase, BenchmarkBaseConfig
from ..utils import  continuous_to_discrete, convert_to_numpy
from ..utils import Logger


log = Logger(__name__, rank_zero_only=True)

@dataclass(kw_only=True)
class BenchmarkHDConfig(BenchmarkBaseConfig):
    input_distribution: Literal['gaussian', 'uniform']
    sample_quantize_range: Tuple[float, float] = (-3, 3)

class BenchmarkHD(BenchmarkBase):
    def __init__(
        self,
        config: BenchmarkHDConfig,
        *,
        num_timesteps: Optional[int] = None,
        init_benchmark: bool = True,
        device: str = 'cpu'
    ):
        self.input_distribution = config.input_distribution
        self.sample_quantize_range = tuple(config.sample_quantize_range)
        super().__init__(config, num_timesteps)
        self.register_buffers(init_benchmark, device)

    @property
    def name(self) -> str:
        return f'hd_' + super().name

    @torch.no_grad()
    def _sample_input(self, num_samples: int) -> torch.LongTensor:
        '''Sample independent source data'''
        if self.input_distribution == 'gaussian':
            samples = continuous_to_discrete(
                torch.randn(size=[num_samples, self.dim], device=self.device, dtype=self.log_cp_cores.dtype),
                self.num_categories, quantize_range=self.sample_quantize_range
            )
        elif self.input_distribution == 'uniform':
            samples = continuous_to_discrete(
                6 * torch.rand(size=(num_samples, self.dim), device=self.device, dtype=self.log_cp_cores.dtype) - 3,
                self.num_categories, quantize_range=self.sample_quantize_range
            )
        else:
            raise ValueError(f'Unknown input distribution: {self.input_distribution}')
        return samples

    def _plot_samples(self, num_samples: int, **kwargs):
        use_pca = self.dim > 2
        if use_pca:
            pca = PCA(n_components=2)
            pca.fit(convert_to_numpy(torch.cat(
                [self.input_dataset, self.target_dataset],
                dim=0
            )))

        # prepare samples
        x_start = convert_to_numpy(self.input_dataset[:num_samples])
        x_end = convert_to_numpy(self.target_dataset[:num_samples])
        if self.reverse:
            pred_x_end = convert_to_numpy(self.sample(
                self.target_dataset[:num_samples],
                use_onestep_sampling=True
            ))
        else:
            pred_x_end = convert_to_numpy(self.sample(
                self.input_dataset[:num_samples],
                use_onestep_sampling=True
            ))
        if use_pca:
            x_start = pca.transform(x_start)
            x_end = pca.transform(x_end)
            pred_x_end = pca.transform(pred_x_end)

        # plot samples
        fig, axs = plt.subplots(
            1, 3, dpi=kwargs.get('dpi', 200),
            figsize=kwargs.get('fig_size', (12, 4))
        )
        axs[0].scatter(
            x_start[:, 0], x_start[:, 1],
            label=r'$p_{start}$', s=kwargs.get('s', 35),
            c='green', edgecolor='black'
        )
        axs[1].scatter(
            x_end[:, 0], x_end[:, 1],
            label=r'$p_{end}$', s=kwargs.get('s', 35),
            c='orange', edgecolor='black'
        )
        axs[2].scatter(
            pred_x_end[:, 0], pred_x_end[:, 1],
            label=r'$p_{pred}$', s=kwargs.get('s', 35),
            c='salmon', edgecolor='black'
        )

        fig.suptitle('Benchmark samples', fontsize=16)
        if use_pca:
            max_value = np.abs(x_end).max()
            axlim = [-max_value - 5, max_value + 5]
        else:
            axlim = [0, self.num_categories - 1]
        r = axlim[1] - axlim[0]
        axlim = [axlim[0] + 0.03 * r, axlim[1] - 0.03 * r]
        for ax in axs:
            ax.grid()
            ax.set(xlim=axlim, ylim=axlim)
            ax.legend(loc='lower left')
        fig.tight_layout(pad=0.5)
        plt.show()
        plt.close()

    def _plot_trajectories(
        self,
        num_samples: int,
        num_trajectories: int,
        num_translations: int,
        **kwargs
    ):
        use_pca = self.dim > 2
        if use_pca:
            pca = PCA(n_components=2)
            pca.fit(convert_to_numpy(torch.cat(
                [self.input_dataset, self.target_dataset],
                dim=0
            )))

        # prepare samples
        x_start = convert_to_numpy(self.input_dataset[:num_samples])
        x_end = convert_to_numpy(self.target_dataset[:num_samples])
        traj_start = self.input_dataset[:num_trajectories]
        repeats = [num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *self.input_dataset.shape[1:])

        trajectories = self.sample_trajectory(
            traj_start, use_onestep_sampling=True
        )
        trajectories = convert_to_numpy(trajectories.reshape(-1, self.dim))
        if use_pca:
            x_end = pca.transform(x_end)
            trajectories = pca.transform(trajectories)
        trajectories = trajectories.reshape(-1, num_trajectories * num_translations, 2)

        # plot samples
        fig, ax = plt.subplots(
            1, 1, dpi=kwargs.get('dpi', 100),
            figsize=kwargs.get('fig_size', (8, 8))
        )
        ax.scatter(
            x_start[:, 0], x_start[:, 1],
            label='Start distribution', s=kwargs.get('s', 100),
            c='grey', edgecolor='black', zorder=1, alpha=0.3,
            linewidth=kwargs.get('linewidth', 0.8)
        )
        ax.scatter(
            x_end[:, 0], x_end[:, 1],
            label='Fitted distribution', s=kwargs.get('s', 100),
            c='salmon', edgecolor='black', zorder=2,
            linewidth=kwargs.get('linewidth', 0.8)
        )
        ax.scatter(
            trajectories[0, :, 0], trajectories[0, :, 1],
            label=r'Trajectory start ($x \sim p_{start}$)', s=kwargs.get('s', 150),
            c='lime', edgecolor='black', zorder=4
        )
        ax.scatter(
            trajectories[-1, :, 0], trajectories[-1, :, 1],
            label=r'Trajectory end ($y \sim p_{end}$)', s=kwargs.get('s', 80),
            c='yellow', edgecolor='black', zorder=4
        )
        for i in range(num_trajectories * num_translations):
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1],
                label='Trajectory (ground truth)' if i == 0 else '',
                c='black', markeredgecolor='black', linewidth=2, zorder=3
            )
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1],
                c='grey', markeredgecolor='black', linewidth=1, zorder=3
            )

        fig.suptitle('Benchmark trajectories', fontsize=16)
        if use_pca:
            max_value = np.abs(x_end).max()
            axlim = [-max_value - 5, max_value + 5]
        else:
            axlim = [0, self.num_categories - 1]
        r = axlim[1] - axlim[0]
        axlim = [axlim[0] + 0.03 * r, axlim[1] - 0.03 * r]
        ax.set(xlim=axlim, ylim=axlim)
        ax.legend(loc='lower left')
        fig.tight_layout(pad=0.5)
        plt.show()
        plt.close()
