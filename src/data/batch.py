from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch


@dataclass
class Batch(Sequence[torch.Tensor | None]):
    encoded: tuple[torch.Tensor | None, torch.Tensor | None]
    raw: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
    cached: bool = False

    def __getitem__(self, index):
        return self.encoded[index]

    def __len__(self) -> int:
        return len(self.encoded)

    def __iter__(self) -> Iterator[torch.Tensor | None]:
        return iter(self.encoded)
