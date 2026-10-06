import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks import CompilationWarmup, TrainOnlyThroughputMonitor


class _WarmupModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.warmup_calls = 0
        self.rng_before_warmup = None
        self.training_random = None

    def compilation_warmup_step(self, batch, optimizer) -> None:
        if self.rng_before_warmup is None:
            self.rng_before_warmup = torch.get_rng_state()
        self.warmup_calls += 1
        time.sleep(0.04)
        optimizer.zero_grad(set_to_none=True)
        self.weight.square().backward()
        optimizer.step()
        torch.rand(1)

    def training_step(self, batch, batch_idx):
        time.sleep(0.01)
        self.training_random = torch.rand(1)
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


class CompilationWarmupTest(unittest.TestCase):
    def test_warmup_restores_state_and_is_excluded_from_train_time(self) -> None:
        dataset = TensorDataset(torch.arange(2, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=2)
        model = _WarmupModel()

        torch.manual_seed(123)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            logger = CSVLogger(temp_dir, name="csv", version="train")
            trainer = Trainer(
                accelerator="cpu",
                max_steps=1,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                callbacks=[
                    CompilationWarmup(num_batches=3),
                    TrainOnlyThroughputMonitor(
                        batch_size_fn=lambda batch: len(batch[0]),
                        window_size=2,
                    ),
                ],
                logger=logger,
            )
            trainer.fit(model, train_dataloaders=loader)
            metrics = pd.read_csv(Path(logger.log_dir) / "metrics.csv")

        self.assertEqual(model.warmup_calls, 3)
        self.assertAlmostEqual(model.weight.item(), 0.8, places=6)
        final_rng = torch.get_rng_state()
        torch.set_rng_state(model.rng_before_warmup)
        expected_training_random = torch.rand(1)
        torch.set_rng_state(final_rng)
        self.assertTrue(
            torch.equal(model.training_random, expected_training_random)
        )
        self.assertGreaterEqual(metrics["warmup/seconds"].dropna().iloc[-1], 0.1)
        self.assertLess(metrics["train/time"].dropna().iloc[-1], 0.1)
        self.assertEqual(trainer.global_step, 1)


if __name__ == "__main__":
    unittest.main()
