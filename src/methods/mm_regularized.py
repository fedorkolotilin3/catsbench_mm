"""Block-MM with a zero-centred quadratic penalty on log parameters.

The E-step, block coefficients, sampling, Lightning integration and acceptance
check are inherited from DLightSBMM. Only the penalty and scalar minimizer differ.
"""

import math

import torch

from .dlight_sb_mm import DLightSBMM


def _quadratic_log_update(log_a, log_b, current_log_z, strength):
    """Minimize a*exp(u) - b*u + strength*u**2/2 by vectorized bisection.

The derivative is strictly increasing. For a>0 a lower bound is
min(0, log(strength)-log(a))-1 and an upper bound is
max(0, log(b)-log(a)). For b=0 the upper bound is zero.
Derivative signs are evaluated in log space to avoid exponentiating a.
"""
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if log_a.shape != log_b.shape or log_a.shape != current_log_z.shape:
        raise ValueError("Block tensors must have the same shape")
    if not torch.isfinite(current_log_z).all():
        raise ValueError("Log regularization requires strictly positive parameters")
    for value in (log_a, log_b):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log coefficients must be finite or -inf")

    log_strength = math.log(strength)
    positive_a = torch.isfinite(log_a)
    safe_a = torch.where(positive_a, log_a, 0.0)
    lower = (log_strength - safe_a).clamp_max(0) - 1
    upper = (log_b - safe_a).clamp_min(0)
    # Float32/64 iteration budgets exceed their mantissa precision for the
    # coefficient ranges encountered here. No device synchronization per step.
    for _ in range(80 if log_a.dtype == torch.float64 else 48):
        middle = lower + (upper - lower) * 0.5
        log_abs_linear = log_strength + middle.abs().log()
        # a*exp(u) + lambda*u > b, with negative terms moved to the RHS.
        left = torch.where(
            middle >= 0,
            torch.logaddexp(safe_a + middle, log_abs_linear),
            safe_a + middle,
        )
        right = torch.where(
            middle >= 0, log_b, torch.logaddexp(log_b, log_abs_linear)
        )
        above = left > right
        upper = torch.where(above, middle, upper)
        lower = torch.where(above, lower, middle)
    result = lower + (upper - lower) * 0.5
    # If a=0 the problem is quadratic, including a=b=0 with minimizer u=0.
    result = torch.where(positive_a, result, (log_b - log_strength).exp())
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite regularized block minimizer")
    return result


class DLightSBMMRegularized(DLightSBMM):
    """Minimize data loss + lambda/2 * (sum(log_beta**2)+sum(log_r**2)).

    The penalty is a sum, not a mean. Lambda=0 restores the original MM,
    including support preservation for zero batch counts. For lambda>0 all
    initial log parameters must be finite. Loss and stopping refer to the
    regularized objective; data_loss and regularization are logged separately.
    """

    def __init__(self, *args, regularization_lambda=0.01, **kwargs):
        if not math.isfinite(regularization_lambda) or regularization_lambda < 0:
            raise ValueError("regularization_lambda must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("regularization_lambda")

    def _regularization(self, log_beta, log_r):
        strength = self.hparams.regularization_lambda
        if strength == 0:
            return super()._regularization(log_beta, log_r)
        if not torch.isfinite(log_beta).all() or not torch.isfinite(log_r).all():
            raise ValueError("Log regularization requires strictly positive parameters")
        return (strength / 2) * (log_beta.square().sum() + log_r.square().sum())

    def _update_block(self, log_a, log_b, current_log_z):
        if self.hparams.regularization_lambda == 0:
            return super()._update_block(log_a, log_b, current_log_z)
        return _quadratic_log_update(
            log_a, log_b, current_log_z, self.hparams.regularization_lambda
        )

    def loss(self, true_x_start, true_x_end):
        data_loss, info = super().loss(true_x_start, true_x_end)
        penalty = self._regularization(self.log_alpha, self.log_cp_cores)
        return data_loss + penalty, {
            **info, "data_loss": data_loss, "regularization": penalty,
        }

    @torch.no_grad()
    def mm_step(self, x0, x1, *, inner_sweeps=None):
        penalty_before = self._regularization(self.log_alpha, self.log_cp_cores).item()
        info = super().mm_step(x0, x1, inner_sweeps=inner_sweeps)
        penalty = self._regularization(self.log_alpha, self.log_cp_cores).item()
        return {
            **info,
            "data_loss": info["loss"] - penalty,
            "data_loss_before": info["loss_before"] - penalty_before,
            "regularization": penalty,
            "regularization_before": penalty_before,
        }
