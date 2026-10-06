import time
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks import TrainOnlyThroughputMonitor


class _SlowValidation(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        time.sleep(0.15)


class _TinyModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        time.sleep(0.01)
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        return self.weight.detach().square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TrainOnlyThroughputMonitorTest(unittest.TestCase):
    def test_validation_time_does_not_leak_into_train_time(self) -> None:
        dataset = TensorDataset(torch.arange(8, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=2)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            logger = CSVLogger(temp_dir, name="csv", version="train")
            monitor = TrainOnlyThroughputMonitor(
                batch_size_fn=lambda batch: len(batch[0]),
                window_size=2,
            )
            trainer = Trainer(
                accelerator="cpu",
                max_steps=4,
                val_check_interval=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                callbacks=[monitor, _SlowValidation()],
                logger=logger,
            )

            wall_started = time.perf_counter()
            trainer.fit(
                _TinyModel(),
                train_dataloaders=loader,
                val_dataloaders=loader,
            )
            wall_elapsed = time.perf_counter() - wall_started

            metrics = pd.read_csv(Path(logger.log_dir) / "metrics.csv")

        train_time = metrics["train/time"].dropna()

        self.assertTrue(train_time.is_monotonic_increasing)
        self.assertLess(train_time.iloc[-1], 0.3)
        self.assertGreaterEqual(wall_elapsed - train_time.iloc[-1], 0.5)
        self.assertEqual(metrics["train/batches"].dropna().iloc[-1], 4)
        self.assertEqual(metrics["train/samples"].dropna().iloc[-1], 8)
        self.assertIn("train/device/batches_per_sec", metrics.columns)
        self.assertIn("train/device/samples_per_sec", metrics.columns)
        self.assertIn("validate/time", metrics.columns)


if __name__ == "__main__":
    unittest.main()
