import bisect
import math
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer


class PeriodicCheckpoint(Callback):
    """Save full checkpoints at evenly spaced optimizer steps."""

    def __init__(self, num_checkpoints: int = 20, subdir: str = "history") -> None:
        if not isinstance(num_checkpoints, int) or num_checkpoints < 1:
            raise ValueError("num_checkpoints must be a positive integer")
        if not subdir or Path(subdir).is_absolute() or len(Path(subdir).parts) != 1:
            raise ValueError("subdir must be one relative directory name")
        self.num_checkpoints = num_checkpoints
        self.subdir = subdir
        self._steps: tuple[int, ...] = ()
        self._next_index = 0
        self._last_saved_step: int | None = None

    @staticmethod
    def checkpoint_steps(total_steps: int, num_checkpoints: int) -> tuple[int, ...]:
        if total_steps < 1 or num_checkpoints < 1:
            raise ValueError("total_steps and num_checkpoints must be positive")
        return tuple(sorted({
            math.ceil(index * total_steps / num_checkpoints)
            for index in range(1, num_checkpoints + 1)
        }))

    def _total_steps(self, trainer: Trainer) -> int:
        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is not None and int(max_steps) > 0:
            return int(max_steps)
        estimated = trainer.estimated_stepping_batches
        if not math.isfinite(float(estimated)) or int(estimated) < 1:
            raise ValueError("Periodic checkpoints require a finite positive training length")
        return int(estimated)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._steps = self.checkpoint_steps(
            self._total_steps(trainer), self.num_checkpoints
        )
        completed = bisect.bisect_right(self._steps, int(trainer.global_step))
        self._next_index = max(self._next_index, completed)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if self._next_index >= len(self._steps) or step < self._steps[self._next_index]:
            return

        while self._next_index < len(self._steps) and self._steps[self._next_index] <= step:
            self._next_index += 1
        if self._last_saved_step == step:
            return
        self._last_saved_step = step

        checkpoint_callback = trainer.checkpoint_callback
        checkpoint_root = (
            Path(checkpoint_callback.dirpath)
            if checkpoint_callback is not None and checkpoint_callback.dirpath is not None
            else Path(trainer.default_root_dir) / "checkpoints"
        )
        history_dir = checkpoint_root / self.subdir
        history_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            history_dir / f"step_{step:08d}.ckpt",
            weights_only=False,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_index": self._next_index,
            "last_saved_step": self._last_saved_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._next_index = int(state_dict.get("next_index", 0))
        self._last_saved_step = state_dict.get("last_saved_step")
