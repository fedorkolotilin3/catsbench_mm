from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class GpuMemoryMonitor(Callback):
    """Log only the peak CUDA memory metrics used by the analysis notebook."""

    def __init__(self, every_n_steps: int = 10) -> None:
        super().__init__()
        if not isinstance(every_n_steps, int) or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.strategy.root_device.type != "cuda":
            return

        step = trainer.global_step
        if step % self.every_n_steps:
            return

        pl_module.log(
            "gpu/allocated_bytes_peak",
            float(torch.cuda.max_memory_allocated()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "gpu/reserved_bytes_peak",
            float(torch.cuda.max_memory_reserved()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
