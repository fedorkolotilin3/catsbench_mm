from typing import Any, Optional, Tuple, Union

from sklearn.datasets import make_swiss_roll
import torch
from torch.utils.data import Dataset, DataLoader

from lightning import LightningDataModule
from ..utils import CoupleDataset, RepeatedDataset, continuous_to_discrete
from .batch import Batch

class DiscreteUniformDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):
        dataset = 6 * torch.rand(size=(num_samples, dim)) - 3
        if not train and dim == 2:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset  

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteGaussianDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):          
        dataset = torch.randn(size=[num_samples, dim])
        if not train and dim == 2:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
class DiscreteSwissRollDataset(Dataset):
    def __init__(
            self, num_samples: int, noise: float = 0.8, num_categories: int = 100, train: bool = True
        ):
        dataset = make_swiss_roll(
            n_samples=num_samples,
            noise=noise
        )[0][:, [0, 2]]  / 7.5
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        dataset = continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset   

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class ToyDataModule(LightningDataModule):
    def __init__(
        self,
        input_dataset: Union[DiscreteUniformDataset, DiscreteGaussianDataset, DiscreteSwissRollDataset],
        target_dataset: Union[DiscreteUniformDataset, DiscreteGaussianDataset, DiscreteSwissRollDataset],
        dim: int,
        num_categories: int,
        num_samples: int,
        train_test_split: Tuple[float, float],
        batch_size: int,
        val_batch_size: int,
        num_train_batches: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        assert dim == 2, "This datamodule is designed for 2D data only."
        assert len(train_test_split) == 2, ( 
            "train_test_split must be a tuple of two floats "
            "representing the proportions for train and val sets."
        )
        assert sum(train_test_split) == 1.0, \
            "The sum of train_test_split must be equal to 1.0."

        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.initial_coupling: Optional[CoupleDataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data by seting variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # here is an `if` because the `setup` method is called multiple times 
        # for trainer.fit, trainer.validate, trainer.test, etc.
        if not self.data_train and not self.data_val and not self.data_test:
            ###################### TRAINING DATASET ######################
            size_train = int(self.hparams.num_samples * self.hparams.train_test_split[0])
            coupled_train = CoupleDataset(
                input_dataset=self.hparams.input_dataset(num_samples=size_train),
                target_dataset=self.hparams.target_dataset(num_samples=size_train),
            )
            self.data_train = RepeatedDataset(
                coupled_train,
                length=self.hparams.num_train_batches * self.hparams.batch_size,
            )
            self.initial_coupling = coupled_train

            ####################### VALIDATION DATASET ######################
            size_val = int(self.hparams.num_samples * self.hparams.train_test_split[1])
            self.data_val = CoupleDataset(
                input_dataset=self.hparams.input_dataset(num_samples=size_val, train=False), 
                target_dataset=self.hparams.target_dataset(num_samples=size_val, train=False)
            )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Used for caching logic in predict mode."""
        # under contract with method.predict and workflow
        # one element batch (CSBM case) always uses first position
        # for Alpha-CSBM predict uses two inputs simultaneously
        if self.trainer.predicting:
            if isinstance(batch, (list, tuple)):
                return Batch(encoded=tuple(batch))
            return Batch(encoded=(batch, None))
        cached = self.trainer.training and getattr(
            self.data_train, "cached", False
        )
        return Batch(encoded=tuple(batch), raw=tuple(batch), cached=cached)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
