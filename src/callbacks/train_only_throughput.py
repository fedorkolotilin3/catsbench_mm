import time
from typing import Any

import torch
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn


class TrainOnlyThroughputMonitor(ThroughputMonitor):
    """Log Lightning-compatible throughput using train-batch time only.

    Lightning's standard monitor derives train time from one wall-clock timer
    and compensates for validation afterwards. Work performed by validation
    callbacks after the last validation batch can therefore leak into the
    next train measurement. This monitor measures every train batch directly,
    while retaining the standard validation throughput implementation.
    """

    def on_train_start(self, trainer, *args: Any) -> None:
        super().on_train_start(trainer, *args)
        self._train_elapsed = 0.0
        self._train_batch_started_at: float | None = None

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.strategy.root_device.type == "cuda":
            torch.cuda.synchronize()
        self._train_batch_started_at = time.perf_counter()

    @rank_zero_only
    @torch.inference_mode()
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._train_batch_started_at is None:
            raise RuntimeError("Train batch timer was not started")

        if trainer.strategy.root_device.type == "cuda":
            torch.cuda.synchronize()
        self._train_elapsed += time.perf_counter() - self._train_batch_started_at
        self._train_batch_started_at = None

        stage = RunningStage.TRAINING
        throughput = self._throughputs[stage]

        if self.length_fn is not None:
            self._lengths[stage] += self.length_fn(batch)

        if self._module_has_flops is None:
            self._module_has_flops = hasattr(pl_module, "flops_per_batch")
            if not self._module_has_flops:
                rank_zero_warn(
                    "When using the TrainOnlyThroughputMonitor, define "
                    f"flops_per_batch on {type(pl_module).__name__} to compute FLOPs."
                )

        flops_per_batch = (
            pl_module.flops_per_batch if self._module_has_flops else None
        )
        self._samples[stage] += self.batch_size_fn(batch)
        self._batches[stage] += 1

        throughput.update(
            time=self._train_elapsed,
            batches=self._batches[stage],
            samples=self._samples[stage],
            lengths=(
                None if self.length_fn is None else self._lengths[stage]
            ),
            flops=flops_per_batch,
        )

        if not trainer.fit_loop._should_accumulate():
            self._compute(trainer)
