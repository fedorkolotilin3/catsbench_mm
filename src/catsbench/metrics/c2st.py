from typing import Tuple, Iterator
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional import auroc


class ClassifierTwoSampleTest(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    probs: torch.Tensor
    targets: torch.Tensor

    def __init__(
        self, 
        dim: int,
        num_categories: int,
        lr: float = 1e-2,
        hid_size: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.lr = lr
        self.hid_size = hid_size
        self.model = nn.Sequential(
            nn.Linear(dim * num_categories, hid_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hid_size, 1, dtype=torch.float64)
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.add_state('probs', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    # These two methods are overridden to avoid hide parameters from outside optimizers
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter(())

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        return iter(())

    def reset(self) -> None:
        super().reset()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def _ddp_average_grads_before_step(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size()
        for p in self.model.parameters():
            if p.grad is None:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world_size)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().long()
        x_onehot = F.one_hot(x, num_classes=self.num_categories)
        return x_onehot.flatten(start_dim=1).to(dtype=next(self.model.parameters()).dtype)

    def update(self, real_data: torch.Tensor, pred_data: torch.Tensor, train: bool) -> None:
        if train:
            self.model.train()
            with torch.inference_mode(False), torch.enable_grad():
                x_real = self._process_input(real_data)
                x_pred = self._process_input(pred_data)

                x = torch.cat([x_real, x_pred], dim=0)
                y = torch.cat([
                    torch.ones(x_real.shape[0], device=x.device, dtype=x.dtype),
                    torch.zeros(x_pred.shape[0], device=x.device, dtype=x.dtype)
                ], dim=0)
                loss = self.criterion(self.model(x).squeeze(-1), y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._ddp_average_grads_before_step()
                self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                x_real = self._process_input(real_data)
                x_pred = self._process_input(pred_data)

                x = torch.cat([x_real, x_pred], dim=0)
                y = torch.cat([
                    torch.ones(x_real.shape[0], device=x.device, dtype=x.dtype),
                    torch.zeros(x_pred.shape[0], device=x.device, dtype=x.dtype)
                ], dim=0)
                probs = torch.sigmoid(self.model(x).squeeze(-1)).detach()
            self.probs.append(probs)
            self.targets.append(y.detach().long())

    def compute(self) -> torch.Tensor:
        if len(self.probs) == 0:
            return torch.tensor(-1.2345, dtype=torch.float64)
        probs = dim_zero_cat(self.probs)
        targets = dim_zero_cat(self.targets)
        auroc_value = auroc(probs, targets, task='binary')
        score = 1.0 - torch.abs(auroc_value - 0.5) * 2.0
        return score
