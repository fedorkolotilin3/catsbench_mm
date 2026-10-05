from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import ot
import torch
from torch.utils.data import Dataset


def get_batch_size(batch: Any) -> int:
    """Return the per-device sample count for Lightning throughput logging."""
    samples = batch.encoded[0] if hasattr(batch, "encoded") else batch[0]
    if samples is None:
        raise ValueError("Cannot determine batch size from an empty batch")
    return len(samples)


def broadcast(tensor: torch.Tensor, num_add_dims: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += tensor.dim() + 1
    shape = [*tensor.shape[:dim], *([1] * num_add_dims), *tensor.shape[dim:]]
    return tensor.reshape(*shape)

def gumbel_sample(logits: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    finfo = torch.finfo(logits.dtype)
    noise = torch.rand_like(logits)
    noise = torch.clamp(noise, min=finfo.tiny, max=1. - finfo.eps)
    gumbel_noise = -torch.log(-torch.log(noise))
    return torch.argmax((logits / tau) + gumbel_noise, dim=dim)

def convert_to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x
    
def convert_to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return x

def continuous_to_discrete(
    batch: Union[torch.Tensor, np.ndarray], 
    num_categories: int,
    quantize_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
):
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).contiguous()
    if quantize_range is None:
        quantize_range = (-3, 3)
    bin_edges = torch.linspace(
        quantize_range[0], 
        quantize_range[1], 
        num_categories - 1
    )
    discrete_batch = torch.bucketize(batch, bin_edges)
    return discrete_batch

class CoupleDataset(Dataset):
    """A dataset that couples two datasets together, allowing for paired sampling."""
    def __init__(
        self,
        input_dataset: torch.Tensor,
        target_dataset: torch.Tensor,
        cached: bool = False,
    ):
        self.input_dataset, self.target_dataset = input_dataset, target_dataset
        self.len_input, self.len_target = len(input_dataset), len(target_dataset)
        self.length = max(self.len_input, self.len_target)
        self.cached = cached

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.input_dataset[idx % self.len_input],
                self.target_dataset[idx % self.len_target])


class NumpyDataset(Dataset):
    """Load one tensor from an NPY file for each item."""

    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, index):
        return torch.from_numpy(np.load(self.paths[index])).long()

    def __len__(self):
        return len(self.paths)

class RepeatedDataset(Dataset):
    """Expose a finite virtual epoch by repeating a smaller dataset."""
    def __init__(self, dataset: Dataset, length: int) -> None:
        self.dataset = dataset
        self.length = length

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self) -> int:
        return self.length

    @property
    def cached(self) -> bool:
        return getattr(self.dataset, "cached", False)

class SampledCoupleDataset(Dataset):
    """Expose freshly sampled marginals through a finite map-style dataset."""
    def __init__(self, length: int, sample_input: Callable, sample_target: Callable) -> None:
        self.length = length
        self.sample_input = sample_input
        self.sample_target = sample_target

    def __getitem__(self, idx: int):
        return self.sample_input(1)[0], self.sample_target(1)[0]

    def __getitems__(self, indices):
        input_batch = self.sample_input(len(indices))
        target_batch = self.sample_target(len(indices))
        return list(zip(input_batch, target_batch))

    def __len__(self) -> int:
        return self.length

def optimize_coupling(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutes batches of data to optimize the coupling between them using Euclidian distance."""
    # get optimal transport coupling between two batches
    input_shape, input_dtype = x.shape[1:], x.dtype
    x = x.flatten(start_dim=1).float()
    y = y.flatten(start_dim=1).float()
    a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    M = torch.cdist(x, y) ** 2
    pi: np.ndarray = ot.emd(a, b, M.detach().cpu().numpy()) # type: ignore
    
    # sample mapping
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    x = x[i].reshape(-1, *input_shape).to(dtype=input_dtype)
    y = y[j].reshape(-1, *input_shape).to(dtype=input_dtype)
    return x, y
