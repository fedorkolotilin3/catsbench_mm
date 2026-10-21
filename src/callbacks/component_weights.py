"""Small component-weight snapshots for the local analysis notebook."""

import csv
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

from .train_only_throughput import TrainOnlyThroughputMonitor


class ComponentWeightsMonitor(Callback):
    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.every_n_steps = every_n_steps

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._timer = next(
            cb for cb in trainer.callbacks
            if isinstance(cb, TrainOnlyThroughputMonitor)
        )
        self._path = Path(trainer.default_root_dir) / "component_weights.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = ["epoch", "step", "train_time_seconds"] + [
            f"component_{k}" for k in range(pl_module.log_alpha.numel())
        ]
        with self._path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(self._fields)
        self._last_step = None
        self._save(trainer, pl_module)

    @torch.no_grad()
    def _save(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.global_step == self._last_step:
            return
        weights = pl_module.log_alpha.detach().double().softmax(dim=0).cpu().tolist()
        with self._path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                trainer.current_epoch + 1, trainer.global_step,
                self._timer._train_elapsed, *weights,
            ])
        self._last_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0:
            self._save(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer, pl_module)
