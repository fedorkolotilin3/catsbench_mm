"""A saved pair of independent marginals, repeated intact at every epoch."""

from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule

from .batch import Batch


class FixedBenchmarkDataModule(LightningDataModule):
    def __init__(self, path, dim=2, num_categories=50, num_workers=0, batch_size=None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if num_workers != 0:
            raise ValueError("The fixed CPU experiment uses num_workers=0")
        self.payload = None

    def setup(self, stage=None):
        if self.payload is not None:
            return
        path = Path(self.hparams.path)
        if not path.is_file():
            raise FileNotFoundError(f"Run python -m scripts.prepare_mm_experiment first: {path}")
        self.file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        for name in ("train_x", "train_y", "test_x", "test_y"):
            x = self.payload[name]
            if (x.dtype != torch.long or x.ndim != 2 or x.shape[1] != self.hparams.dim
                    or len(x) == 0 or x.min() < 0 or x.max() >= self.hparams.num_categories):
                raise ValueError(f"Invalid fixed data: {name}")

    def train_dataloader(self):
        if self.hparams.batch_size is not None:
            return DataLoader(TensorDataset(self.payload["train_x"], self.payload["train_y"]),
                              batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
        # One element IS the full batch. No resampling, pairing, drop_last or shuffle.
        return DataLoader([(self.payload["train_x"], self.payload["train_y"])],
                          batch_size=None, num_workers=0)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return Batch(encoded=tuple(batch), raw=tuple(batch))
