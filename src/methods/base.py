from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from lightning import LightningModule


class BaseMethod(LightningModule, ABC):
    """Common base class for all benchmark methods."""

    # ThroughputMonitor treats this attribute as optional. Keeping it explicit
    # disables FLOP estimation without emitting a warning on every run.
    flops_per_batch: int | None = None

    @abstractmethod
    def sample(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_trajectory(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError
