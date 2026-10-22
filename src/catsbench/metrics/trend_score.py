import math
from typing import List, Literal

import torch

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class TrendScore(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    scores: List[torch.Tensor]
    real_counts: torch.Tensor
    pred_counts: torch.Tensor

    def __init__(
        self,
        dim: int,
        num_categories: int,
        conditional: bool = False,
        reduction: Literal['mean', 'none'] = 'mean',
        adjusted: bool = False,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("TrendScore requires at least two dimensions")
        self.dim = dim
        self.num_categories = num_categories
        self.conditional = conditional
        self.reduction = reduction
        self.adjusted = adjusted
        
        if conditional:
            self.add_state("scores", default=[], dist_reduce_fx='cat')
        else:
            self.add_state(
                "real_counts",
                default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.int),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "pred_counts",
                default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.int),
                dist_reduce_fx="sum",
            )

    def _expected_null_tvd(
        self,
        real_counts: torch.Tensor,
        pred_counts: torch.Tensor,
        real_totals: torch.Tensor,
        pred_totals: torch.Tensor,
    ) -> torch.Tensor:
        pooled_counts = real_counts + pred_counts
        pooled_totals = real_totals + pred_totals
        pooled_probs = pooled_counts.to(dtype=torch.float64) / (
            pooled_totals + torch.finfo(pooled_totals.dtype).eps
        )

        inv_real = torch.where(
            real_totals > 0,
            1.0 / real_totals.clamp_min(1.0),
            torch.zeros_like(real_totals),
        )
        inv_pred = torch.where(
            pred_totals > 0,
            1.0 / pred_totals.clamp_min(1.0),
            torch.zeros_like(pred_totals),
        )
        variance = pooled_probs * (1.0 - pooled_probs) * (inv_real + inv_pred)
        expected_l1 = math.sqrt(2.0 / math.pi) * torch.sqrt(
            variance.clamp_min(0.0)
        ).sum(dim=[2, 3])
        return (0.5 * expected_l1).clamp(max=1.0 - torch.finfo(expected_l1.dtype).eps)

    def _compute_score(self, real_counts: torch.Tensor, pred_counts: torch.Tensor) -> torch.Tensor:
        real_totals = real_counts.sum(dim=[2, 3], keepdim=True).to(dtype=torch.float64) # (D, D, 1, 1)
        pred_totals = pred_counts.sum(dim=[2, 3], keepdim=True).to(dtype=torch.float64) # (D, D, 1, 1)

        reals = real_counts.to(dtype=torch.float64) / (real_totals + torch.finfo(real_totals.dtype).eps)
        preds = pred_counts.to(dtype=torch.float64) / (pred_totals + torch.finfo(pred_totals.dtype).eps)

        tvd = 0.5 * torch.abs(reals - preds).sum(dim=[2, 3]) # (D, D)
        if not self.adjusted:
            return 1.0 - tvd

        # Normal approximation to the TVD expected from two iid empirical histograms.
        null_tvd = self._expected_null_tvd(
            real_counts, pred_counts, real_totals, pred_totals
        )
        denom = (1.0 - null_tvd).clamp_min(torch.finfo(tvd.dtype).eps)
        score = 1.0 - (tvd - null_tvd).clamp_min(0.0) / denom
        return score.clamp(0.0, 1.0)

    def _offdiag_scores(self, scores: torch.Tensor) -> torch.Tensor:
        mask = ~torch.eye(self.dim, dtype=torch.bool, device=scores.device)
        return scores[..., mask]

    def _fill_symmetric_counts(
        self, 
        target_tensor: torch.Tensor, 
        flat_tri_counts: torch.Tensor, 
        ii: torch.Tensor, 
        jj: torch.Tensor
    ) -> None:
        '''
        Populates a (D, D, C, C) tensor using flattened triangular counts (K, C, C).
        Handles symmetrization: (i,j) gets Count, (j,i) gets Count.T
        '''
        diag_mask = (ii == jj)
        if diag_mask.any():
            i_d = ii[diag_mask]
            j_d = jj[diag_mask]
            target_tensor.index_put_((i_d, j_d), flat_tri_counts[diag_mask], accumulate=True)

        off_mask = ~diag_mask
        if off_mask.any():
            i_o = ii[off_mask]
            j_o = jj[off_mask]
            counts_o = flat_tri_counts[off_mask] # (N_off, C, C)

            target_tensor.index_put_((i_o, j_o), counts_o, accumulate=True)
            target_tensor.index_put_((j_o, i_o), counts_o.transpose(-2, -1), accumulate=True)

    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        if real_data.shape != pred_data.shape or real_data.ndim != 2:
            raise ValueError("Expect real_data/pred_data of shape (batch_size, dim).")
        if real_data.shape[1] != self.dim:
            raise ValueError(f"Expected second dim = {self.dim}, got {real_data.shape[1]}")
        if real_data.min() < 0 or pred_data.min() < 0:
            raise ValueError("Data contains negative category values")
        if real_data.max() >= self.num_categories or pred_data.max() >= self.num_categories:
            raise ValueError(f"Data contains values >= num_categories ({self.num_categories})")

        if not hasattr(self, "_tri_i") or \
            self._tri_i.device != real_data.device or \
                self._tri_i.numel() != (self.dim * (self.dim - 1)) // 2:
            ii, jj = torch.triu_indices(self.dim, self.dim, offset=1, device=real_data.device)
            self._tri_i = ii # (K,)
            self._tri_j = jj # (K,)
            K = ii.numel()
            self._tri_block_offsets = torch.arange(
                K, device=real_data.device, dtype=torch.long
            ) * (self.num_categories * self.num_categories)

        r_pair_tri = (real_data.index_select(1, self._tri_i) \
                      * self.num_categories + real_data.index_select(1, self._tri_j))  # (B, K)
        p_pair_tri = (pred_data.index_select(1, self._tri_i) \
                      * self.num_categories + pred_data.index_select(1, self._tri_j))  # (B, K)

        r_code = (self._tri_block_offsets.view(1, -1) + r_pair_tri).reshape(-1)  # (B*K,)
        p_code = (self._tri_block_offsets.view(1, -1) + p_pair_tri).reshape(-1)  # (B*K,)

        K = self._tri_i.numel()
        bins = K * self.num_categories * self.num_categories
        r_cnt_tri = torch.bincount(r_code, minlength=bins).reshape(K, self.num_categories, self.num_categories).int()
        p_cnt_tri = torch.bincount(p_code, minlength=bins).reshape(K, self.num_categories, self.num_categories).int()

        # for conditional, compute and store score immediately 
        if self.conditional:
            batch_r = torch.zeros(
                self.dim, self.dim, self.num_categories, self.num_categories, 
                device=real_data.device, dtype=torch.int
            )
            batch_p = torch.zeros_like(batch_r)
            
            self._fill_symmetric_counts(batch_r, r_cnt_tri, self._tri_i, self._tri_j)
            self._fill_symmetric_counts(batch_p, p_cnt_tri, self._tri_i, self._tri_j)
            
            score = self._compute_score(batch_r, batch_p) # to enable stacking using cat
            self.scores.append(score.unsqueeze(0))
        
        # for unconditional, accumulate counts
        else:
            self._fill_symmetric_counts(self.real_counts, r_cnt_tri, self._tri_i, self._tri_j)
            self._fill_symmetric_counts(self.pred_counts, p_cnt_tri, self._tri_i, self._tri_j)

    def compute(self) -> torch.Tensor:
        if self.conditional:
            all_scores = dim_zero_cat(self.scores)
            if self.reduction == "mean":
                return self._offdiag_scores(all_scores).mean()
            return all_scores # (N, D, D)
        
        scores = self._compute_score(self.real_counts, self.pred_counts) # (D, D)

        if self.reduction == "mean":
            return self._offdiag_scores(scores).mean()
        return scores
    
