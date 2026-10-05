import os
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..utils import CoupleDataset, RepeatedDataset
from .batch import Batch
from .codec import BaseCodec


class DiscreteColoredMNISTDataset(Dataset):
    def __init__(self, target_digit: int, data_dir: str, train: bool = True, img_size: int = 32) -> None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(self._get_random_colored_images),
        ])
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        self.dataset = (255 * torch.stack(
            [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == target_digit],
            dim=0,
        )).to(dtype=torch.int64)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _get_random_colored_images(image: torch.Tensor) -> torch.Tensor:
        hue = 360 * torch.rand(1)
        image_min = 0
        image_diff = (image - image_min) * (hue % 60) / 60
        image_inc, image_dec = image_diff, image - image_diff
        colored_image = torch.zeros((3, image.shape[1], image.shape[2]))
        hue_sector = torch.round(hue / 60) % 6
        if hue_sector == 0:
            colored_image[0], colored_image[1], colored_image[2] = image, image_inc, image_min
        elif hue_sector == 1:
            colored_image[0], colored_image[1], colored_image[2] = image_dec, image, image_min
        elif hue_sector == 2:
            colored_image[0], colored_image[1], colored_image[2] = image_min, image, image_inc
        elif hue_sector == 3:
            colored_image[0], colored_image[1], colored_image[2] = image_min, image_dec, image
        elif hue_sector == 4:
            colored_image[0], colored_image[1], colored_image[2] = image_inc, image_min, image
        else:
            colored_image[0], colored_image[1], colored_image[2] = image, image_min, image_dec
        return colored_image


class CelebaDataset(Dataset):
    transform: Optional[transforms.Compose] = None
    
    def __init__(
        self, 
        sex: Literal['male', 'female', 'both'], 
        data_dir: str,
        size: Optional[int] = None, 
        train: bool = True,
        split: int | float = 0.9, # original CSBM CelebA experiment setup
        use_quantized: bool = True,
        return_names: bool = False
    ):
        self.train = train
        self.use_quantized = train and use_quantized
        self.size = size
        self.return_names = return_names
        self.data_dir= data_dir

        subset = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))

        if isinstance(split, int): 
            # this logic mathches setup of previously trained models
            subset = subset.iloc[:split] if train else subset.iloc[split:]
            if sex == 'male':
                subset = subset[subset['Male'] != -1]
            elif sex == 'female':
                subset = subset[subset['Male'] == -1]
            else:
                subset = subset
        else:
            # this logic mathches asbm setup
            male_subset = subset[subset['Male'] != -1]
            female_subset = subset[subset['Male'] == -1]
            male_split_index, female_split_index = int(len(male_subset) * split), int(len(female_subset) * split)
            
            male_subset = male_subset[:male_split_index] if train else male_subset.iloc[male_split_index:]
            female_subset = female_subset[:female_split_index] if train else female_subset.iloc[female_split_index:]

            if sex == 'male':
                subset = male_subset
            elif sex == 'female':
                subset = female_subset
            else:
                subset = pd.concat([male_subset, female_subset], ignore_index=True)
                subset = subset.sort_values(by='image_id').reset_index(drop=True)

        if self.use_quantized:
            sub_folder = 'quantized'
            subset['image_id'] = subset['image_id'].str.removesuffix('.jpg') + '.npy'
        else:
            sub_folder = 'raw'

        self.image_names = subset['image_id']
        self.dataset = [
            os.path.join(data_dir, 'img_align_celeba', sub_folder, image)
            for image in self.image_names.tolist()
        ]

    def __getitem__(self, index):
        if self.use_quantized:
            image = torch.from_numpy(np.load(self.dataset[index]))
        else:
            transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])
            image = Image.open(self.dataset[index])
            image = image.convert('RGB')
            image = transform(image)

        if self.return_names:
           return image, self.dataset[index].split('/')[-1]
        return image

    def __len__(self):
        return len(self.dataset)
    
    def get_by_filename(self, index):
        transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
        ])
        image = Image.open(
            os.path.join(self.data_dir, 'img_align_celeba', 'raw', index)
        )
        image = image.convert('RGB')
        image = transform(image)
        return image

class ImageDataModule(LightningDataModule):
    """Generic paired image marginals with a shared categorical image codec."""

    def __init__(
        self,
        input_dataset: Callable[..., Dataset],
        target_dataset: Callable[..., Dataset],
        codec: BaseCodec,
        batch_size: int,
        val_batch_size: int,
        num_train_batches: int,
        train_data_mode: Literal["raw", "encoded"] = "raw",
        eval_data_mode: Literal["raw", "encoded"] = "raw",
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_categories: int = 256,
        dim: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("codec",), logger=False)
        self.codec = codec
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.initial_coupling: Optional[CoupleDataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data by setting `self.data_train`, `self.data_val`, and `self.data_test`."""
        # setup is called multiple times for fit, validate, and test.
        if not self.data_train and not self.data_val and not self.data_test:
            coupled_train = CoupleDataset(
                input_dataset=self.hparams.input_dataset(),
                target_dataset=self.hparams.target_dataset(),
            )
            self.data_train = RepeatedDataset(
                coupled_train,
                length=self.hparams.num_train_batches * self.hparams.batch_size,
            )
            self.initial_coupling = coupled_train
            self.data_val = CoupleDataset(
                input_dataset=self.hparams.input_dataset(train=False),
                target_dataset=self.hparams.target_dataset(train=False),
            )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Used for caching logic in predict mode."""
        if self.trainer.predicting:
            # for Alpha-CSBM predict uses two inputs simultaneously
            if isinstance(batch, (list, tuple)):
                x, y = batch
                if self.hparams.train_data_mode == "raw":
                    self.codec.to(x.device)
                    x = self.codec.encode_to_cats(x)
                    y = self.codec.encode_to_cats(y)
                return Batch(encoded=(x, y))

            # under contract with method.predict and workflow
            # one element batch (CSBM case) always uses first position
            if self.hparams.train_data_mode == "raw":
                self.codec.to(batch.device)
                batch = self.codec.encode_to_cats(batch)
            return Batch(encoded=(batch, None))

        if self.trainer.training:
            data_mode = self.hparams.train_data_mode
        else:
            data_mode = self.hparams.eval_data_mode
        cached = self.trainer.training and getattr(
            self.data_train, "cached", False
        )

        if data_mode == "encoded":
            return Batch(encoded=tuple(batch), cached=cached)
        x, y = batch
        self.codec.to(x.device)
        return Batch(
            encoded=(self.codec.encode_to_cats(x), self.codec.encode_to_cats(y)),
            raw=(x, y),
            cached=cached,
        )
    
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and self.hparams.num_workers > 0
            ),
            drop_last=True,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.predict_dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and self.hparams.num_workers > 0
            ),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and self.hparams.num_workers > 0
            ),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and self.hparams.num_workers > 0
            ),
        )
