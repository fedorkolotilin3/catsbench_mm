from typing import Dict, Literal, Tuple, Union

import torch
from torch.nn import functional as F

from .csbm import CSBM
from ..utils import gumbel_sample


class C2SBM(CSBM):
    get_transition_logits = None

    def kl_loss(
        self, true_logits: torch.Tensor, pred_logits: torch.Tensor
    ) -> torch.Tensor:
        """Joint chain-rule KL: sum coordinates, then average the batch."""
        true_log_probs = true_logits.log_softmax(dim=-1)
        pred_log_probs = pred_logits.log_softmax(dim=-1)
        true_probs = true_log_probs.exp()
        support = true_probs > 0
        log_ratio = torch.where(
            support, true_log_probs - pred_log_probs, torch.zeros_like(true_probs)
        )
        coordinate_kl = (true_probs * log_ratio).sum(dim=-1)
        return coordinate_kl.flatten(start_dim=1).sum(dim=-1).mean()

    def ce_loss(
        self, true_x_prev: torch.Tensor, pred_logits: torch.Tensor
    ) -> torch.Tensor:
        """Joint transition NLL: sum coordinates, then average the batch."""
        per_coordinate = F.cross_entropy(
            pred_logits.movedim(-1, 1),
            true_x_prev,
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        )
        return per_coordinate.flatten(start_dim=1).sum(dim=-1).mean()

    def markovian_projection(
        self,
        fb: Literal["forward", "backward"],
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=1,
            high=self.hparams.num_timesteps + 2,
            size=(batch_size,),
            device=self.device,
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)
        true_logits = self.prior.posterior_logits(
            true_x_start, x_t, t, logits=False
        )

        true_x_prev = gumbel_sample(true_logits, tau=1.0, dim=-1)
        pred_logits = self.models[fb](x_t, t, x_prev=true_x_prev)

        loss = true_logits.new_zeros(())
        kl = true_logits.new_zeros(())
        ce = true_logits.new_zeros(())
        mse = true_logits.new_zeros(())
        if self.hparams.kl_loss_coeff > 0:
            kl = self.kl_loss(true_logits, pred_logits)
            loss = loss + self.hparams.kl_loss_coeff * kl
        if self.hparams.ce_loss_coeff > 0:
            ce = self.ce_loss(true_x_prev, pred_logits)
            loss = loss + self.hparams.ce_loss_coeff * ce
        if self.hparams.mse_loss_coeff > 0:
            mse = self.mse_loss(true_logits, pred_logits)
            loss = loss + self.hparams.mse_loss_coeff * mse

        info = {
            f"kl_loss_{fb}": kl,
            f"ce_loss_{fb}": ce,
            f"mse_loss_{fb}": mse,
        }
        return loss, info

    def markov_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        fb: Literal["forward", "backward"],
        return_transitions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        model = self.models[fb]
        batch_size = x_t.shape[0]
        event_size = x_t[0].numel()
        event_shape = x_t.shape[1:]
        samples = torch.full_like(x_t, model.mask_token_id)
        samples_flat = samples.reshape(batch_size, event_size)

        transition_logits = []
        for index in range(event_size):
            logits = model(x_t, t, x_prev=samples.clone())
            logits = logits.reshape(batch_size, event_size, -1)[:, index]
            transition_logits.append(logits)

            token = gumbel_sample(logits, tau=self.hparams.tau, dim=-1)
            if self.hparams.argmax_mode:
                token = torch.where(t == 1, logits.argmax(dim=-1), token)
            samples_flat[:, index] = token

        transition_logits = torch.stack(transition_logits, dim=1).reshape(
            batch_size, *event_shape, transition_logits[0].shape[-1]
        )
        if return_transitions:
            return samples, transition_logits
        return samples
