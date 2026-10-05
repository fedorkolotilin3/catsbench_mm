import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig
from torch.utils.data import Subset

from src.data.benchmark_datamodule import BenchmarkDataModule
from src.methods.base import BaseMethod
from src.utils import CoupleDataset, NumpyDataset, RepeatedDataset

from .fit import BaseWorkflow


@dataclass(frozen=True)
class IMFPhase:
    direction: Literal["forward", "backward", "both"]
    iteration: int

    def __post_init__(self) -> None:
        if self.direction not in ("forward", "backward", "both"):
            raise ValueError(f"Unknown IMF direction: {self.direction}")
        if self.iteration < 1:
            raise ValueError("The IMF iteration must be positive.")

    @property
    def opposite_direction(self) -> Literal["forward", "backward", "both"]:
        if self.direction == "both":
            return "both"
        return "backward" if self.direction == "forward" else "forward"


class IMFCacheWriter(BasePredictionWriter):
    """Write IMF targets one predict batch at a time."""

    def __init__(
        self,
        cache_dir,
        expected_size,
        direction,
        storage_dtype="uint16",
    ):
        super().__init__(write_interval="batch")
        self.cache_dir = Path(cache_dir)
        self.directions = (
            ("forward", "backward")
            if direction == "both"
            else (direction,)
        )
        self.targets_dirs = {
            direction: self.cache_dir / direction / "targets"
            for direction in self.directions
        }
        self.success_paths = {
            direction: self.cache_dir / direction / "_SUCCESS"
            for direction in self.directions
        }
        self.expected_size = expected_size
        self.storage_dtype = getattr(torch, storage_dtype)
        completed_ids = [
            set(self._target_paths(direction))
            for direction in self.directions
        ]
        self.completed_ids = set.intersection(*completed_ids)
        self.pending_ids = [
            index
            for index in range(expected_size)
            if index not in self.completed_ids
        ]
        self.is_complete = all(
            self.success_paths[direction].is_file()
            for direction in self.directions
        )

    def _target_paths(self, direction):
        paths = {}
        for path in sorted(self.targets_dirs[direction].glob("*.npy")):
            try:
                input_id = int(path.stem.split("-rank-")[0])
            except ValueError:
                continue
            paths.setdefault(input_id, path)
        return paths

    def on_predict_start(self, trainer, pl_module):
        """Prepare the cache directory across all ranks."""
        if trainer.is_global_zero:
            for targets_dir in self.targets_dirs.values():
                targets_dir.mkdir(parents=True, exist_ok=True)
        trainer.strategy.barrier("imf-cache-start")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Save the current prediction batch to the cache."""
        if batch_indices is None:
            raise RuntimeError("Lightning did not provide predict batch indices.")

        input_ids = [self.pending_ids[index] for index in batch_indices]
        predictions = (
            prediction
            if len(self.directions) == 2
            else (prediction,)
        )
        for direction, targets in zip(self.directions, predictions):
            targets = targets.detach().cpu().to(self.storage_dtype)
            for input_id, target in zip(input_ids, targets):
                path = self.targets_dirs[direction] / (
                    f"{input_id:08d}-rank-{trainer.global_rank}.npy"
                )
                if path.exists():
                    continue
                temporary = path.with_name(f".{path.name}.tmp")
                with temporary.open("wb") as file:
                    np.save(file, target.numpy())
                os.replace(temporary, path)
        self.completed_ids.update(input_ids)

    def on_predict_end(self, trainer, pl_module):
        """Finalize caching"""
        trainer.strategy.barrier("imf-cache-targets")
        completed_ids = [
            set(self._target_paths(direction))
            for direction in self.directions
        ]
        self.completed_ids = set.intersection(*completed_ids)
        if trainer.is_global_zero:
            expected_ids = set(range(self.expected_size))
            for direction, direction_ids in zip(
                self.directions,
                completed_ids,
            ):
                if direction_ids == expected_ids:
                    success_path = self.success_paths[direction]
                    temporary = success_path.with_suffix(".tmp")
                    temporary.touch()
                    os.replace(temporary, success_path)
        trainer.strategy.barrier("imf-cache-finish")
        self.is_complete = all(
            self.success_paths[direction].is_file()
            for direction in self.directions
        )

    def get_target_dataset(self, direction=None):
        if direction is None:
            if len(self.directions) != 1:
                raise ValueError("Cache direction must be specified.")
            direction = self.directions[0]

        if not self.success_paths[direction].is_file():
            raise RuntimeError(
                f"IMF cache is incomplete: {self.cache_dir / direction}"
            )

        paths = self._target_paths(direction)
        if set(paths) != set(range(self.expected_size)):
            raise RuntimeError(
                f"IMF cache is incomplete: {self.cache_dir / direction}"
            )
        return NumpyDataset([paths[index] for index in range(self.expected_size)])


class IMFWorkflow(BaseWorkflow):
    """Implement the IMF training procedure."""

    def __init__(
        self,
        trainer_config: DictConfig,
        callbacks_config: Optional[DictConfig],
        logger_config: Optional[DictConfig],
        num_first_iterations: int,
        cache_dir: Optional[str] = None,
        online: bool = False,
    ) -> None:
        super().__init__(
            trainer_config=trainer_config,
            callbacks_config=callbacks_config,
            logger_config=logger_config,
        )
        if num_first_iterations < 1:
            raise ValueError("num_first_iterations must be positive.")
        self.num_first_iterations = num_first_iterations
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.online = online

    def make_trainer(
        self,
        hparams: Mapping[str, Any],
        **overrides,
    ) -> Trainer:
        """Override to ensure that ModelCheckpoint(save_last=True) is used."""
        trainer = super().make_trainer(hparams, **overrides)
        checkpoint_callback = trainer.checkpoint_callback
        if (
            not isinstance(checkpoint_callback, ModelCheckpoint)
            or not checkpoint_callback.save_last
        ):
            raise ValueError(
                "IMFWorkflow requires ModelCheckpoint(save_last=True)."
            )
        return trainer

    def get_phase_by_epoch(self, epoch: int) -> IMFPhase:
        """Return the phase by the epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        if self.online:
            iteration = max(1, epoch - self.num_first_iterations + 2)
            return IMFPhase(direction="both", iteration=iteration)

        if epoch < self.num_first_iterations:
            return IMFPhase(direction="forward", iteration=1)
        if epoch < 2 * self.num_first_iterations:
            return IMFPhase(direction="backward", iteration=1)

        offset = epoch - 2 * self.num_first_iterations
        iteration = 2 + offset // 2
        direction = "forward" if offset % 2 == 0 else "backward"
        return IMFPhase(direction=direction, iteration=iteration)

    def get_phase_max_epoch(self, epoch: int) -> int:
        """Return the last epoch of the current phase."""
        if self.online:
            if epoch < self.num_first_iterations:
                return self.num_first_iterations
            return epoch + 1

        if epoch < self.num_first_iterations:
            return self.num_first_iterations
        if epoch < 2 * self.num_first_iterations:
            return 2 * self.num_first_iterations
        return epoch + 1

    def build_cache(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        phase: IMFPhase,
        ckpt_path: str,
    ) -> None:
        if self.cache_dir is None:
            raise ValueError(
                "IMFWorkflow requires cache_dir after the first iteration."
            )

        # since each phase the trainer objet is new
        # we need to obtain datasets that ONLY initialized in the setup method
        datamodule.setup("predict")
        # CSBM predicts one marginal at once, while Alpha-CSBM predicts
        # both directions from the initial coupling simultaneously.
        initial_coupling = datamodule.initial_coupling
        if phase.direction == "forward":
            predict_dataset = initial_coupling.target_dataset
        elif phase.direction == "backward":
            predict_dataset = initial_coupling.input_dataset
        else:
            predict_dataset = initial_coupling

        # the writer loads completed ids and computes pending ids
        writer = IMFCacheWriter(
            self.cache_dir
            / f"iteration_{phase.iteration:03d}",
            len(predict_dataset),
            phase.direction,
        )

        if not writer.is_complete:
            # the subset therefore is used to predict only the pending ids
            datamodule.predict_dataset = Subset(
                predict_dataset,
                writer.pending_ids,
            )
            predictor = instantiate(
                self.trainer_config,
                callbacks=[writer, TQDMProgressBar()],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )
            predictor.predict(
                method,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                return_predictions=False,
            )
            datamodule.predict_dataset = None

        if phase.direction == "forward":
            pairs = CoupleDataset(
                writer.get_target_dataset(),
                predict_dataset,
                cached=True,
            )
        elif phase.direction == "backward":
            pairs = CoupleDataset(
                predict_dataset,
                writer.get_target_dataset(),
                cached=True,
            )
        else:
            forward_pairs = CoupleDataset(
                writer.get_target_dataset("forward"),
                initial_coupling.target_dataset,
                cached=True,
            )
            backward_pairs = CoupleDataset(
                initial_coupling.input_dataset,
                writer.get_target_dataset("backward"),
                cached=True,
            )
            pairs = CoupleDataset(
                forward_pairs,
                backward_pairs,
                cached=True,
            )

        # alpha-CSBM effectively uses only half of the batch size,
        # thus we reduce it to avoid unnecessary computation  
        train_length = len(datamodule.data_train)
        if phase.direction == "both" and not getattr(
            datamodule.data_train, "cached", False
        ):
            if datamodule.hparams.batch_size % 2 != 0:
                raise ValueError(
                    "Online IMF cache requires an even train batch size."
                )
            datamodule.hparams.batch_size //= 2
            train_length //= 2

        datamodule.data_train = RepeatedDataset(pairs, length=train_length)

    def test(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: str,
        hparams: Mapping[str, Any],
    ) -> Trainer:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        method.training_phase = self.get_phase_by_epoch(
            int(checkpoint["epoch"])
        )
        del checkpoint
        return super().test(method, datamodule, ckpt_path, hparams)

    def train(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: Optional[str],
        hparams: Mapping[str, Any],
    ) -> Optional[Trainer]:
        if self.cache_dir is not None and isinstance(
            datamodule,
            BenchmarkDataModule,
        ):
            raise ValueError(
                "IMF cache is not supported for BenchmarkDataModule."
            )

        # get max and current (from checkpoint) epoch
        max_epochs = int(self.trainer_config.max_epochs)
        current_epoch = 0
        if ckpt_path is not None:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            current_epoch = int(checkpoint["epoch"]) + 1
            del checkpoint

        if current_epoch >= max_epochs:
            return None

        while current_epoch < max_epochs:
            phase = self.get_phase_by_epoch(current_epoch)
            method.training_phase = phase

            if self.cache_dir is not None and phase.iteration > 1:
                self.build_cache(method, datamodule, phase, ckpt_path)

            # FIT STAGE
            trainer = self.make_trainer(
                hparams,
                max_epochs=min(
                    self.get_phase_max_epoch(current_epoch),
                    max_epochs
                )
            )
            trainer.fit(method, datamodule=datamodule, ckpt_path=ckpt_path)

            # check that ModelCheckpoint saved a last checkpoint for the next phase
            ckpt_path = trainer.checkpoint_callback.last_model_path
            if not ckpt_path:
                raise RuntimeError("ModelCheckpoint did not save a last checkpoint.")

            current_epoch = trainer.current_epoch

        return trainer
