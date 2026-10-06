from copy import deepcopy
import time
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class CompilationWarmup(Callback):
    """Warm up the real training path without changing training state.

    The first already-transferred training batch is executed repeatedly before
    Lightning starts processing it. Model parameters, optimizer state and RNG
    state are then restored, while compiled CUDA/Triton kernels remain cached.
    """

    def __init__(self, num_batches: int = 3) -> None:
        super().__init__()
        if not isinstance(num_batches, int) or num_batches < 1:
            raise ValueError("num_batches must be a positive integer")
        self.num_batches = num_batches
        self._completed = False

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._completed:
            return

        warmup_step = getattr(pl_module, "compilation_warmup_step", None)
        if not callable(warmup_step):
            raise RuntimeError(
                f"{type(pl_module).__name__} does not implement "
                "compilation_warmup_step"
            )

        optimizer = trainer.optimizers[0]
        saved_parameters = {
            name: parameter.detach().clone()
            for name, parameter in pl_module.named_parameters()
        }
        saved_gradients = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            for name, parameter in pl_module.named_parameters()
        }
        saved_optimizer = deepcopy(optimizer.state_dict())
        saved_cpu_rng = torch.get_rng_state()
        saved_cuda_rng = None

        if pl_module.device.type == "cuda":
            saved_cuda_rng = torch.cuda.get_rng_state_all()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started_at = time.perf_counter()

        try:
            for _ in range(self.num_batches):
                warmup_step(batch=batch, optimizer=optimizer)

            if pl_module.device.type == "cuda":
                torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - started_at
        finally:
            with torch.no_grad():
                for name, parameter in pl_module.named_parameters():
                    parameter.copy_(saved_parameters[name])
                    saved_gradient = saved_gradients[name]
                    parameter.grad = (
                        None
                        if saved_gradient is None
                        else saved_gradient.clone()
                    )

            optimizer.load_state_dict(saved_optimizer)
            torch.set_rng_state(saved_cpu_rng)

            if saved_cuda_rng is not None:
                torch.cuda.set_rng_state_all(saved_cuda_rng)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

        for logger in trainer.loggers:
            logger.log_metrics(
                {
                    "warmup/seconds": warmup_seconds,
                    "warmup/num_batches": self.num_batches,
                },
                step=trainer.global_step,
            )

        self._completed = True
