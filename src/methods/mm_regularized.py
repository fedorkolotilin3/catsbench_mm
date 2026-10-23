"""Block-MM with normalized exponential-linear penalties on log parameters.

The E-step, block coefficients, sampling, Lightning integration and acceptance
check are inherited from DLightSBMM. Only the penalty and scalar minimizer differ.
"""

import math

import torch

from .dlight_sb_mm import DLightSBMM


@torch.no_grad()
def _exponential_log_update(
    log_a,
    log_b,
    current_log_z,
    strength,
    min_component_value=1e-12,
):
    """Minimize (a+lambda)*exp(u) - (b+lambda)*u without root finding."""
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if not math.isfinite(min_component_value) or min_component_value <= 0:
        raise ValueError("min_component_value must be finite and positive")
    if log_a.shape != log_b.shape or log_a.shape != current_log_z.shape:
        raise ValueError("Block tensors must have the same shape")
    if not torch.isfinite(current_log_z).all():
        raise ValueError("Log regularization requires strictly positive parameters")
    for value in (log_a, log_b):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log coefficients must be finite or -inf")
    log_strength = log_a.new_tensor(math.log(strength))
    result = torch.logaddexp(log_b, log_strength) - torch.logaddexp(log_a, log_strength)
    result = result.clamp_min(math.log(min_component_value))
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite regularized block minimizer")
    return result


@torch.no_grad()
def _exponential_log_penalty(u):
    """exp(u)-u-1, with a local series to avoid cancellation near zero."""
    small = u.abs() < 1e-3
    local = torch.where(small, u, 0.0)
    series = local.square() * (0.5 + local / 6 + local.square() / 24)
    return torch.where(small, series, torch.expm1(u) - u)


class DLightSBMMRegularized(DLightSBMM):
    """Minimize data loss plus separate mean penalties for weights and cores.

    ``regularization_lambda_alpha`` multiplies the mean penalty over mixture
    weights. ``regularization_lambda_cores`` multiplies the mean penalty over
    all categorical core parameters. Zero coefficients restore the original
    MM update for the corresponding block. Loss and stopping refer to the
    regularized objective; the scalar minimizer is clipped to the configured
    positive component floor. Data loss and regularization are logged
    separately.
    """

    def __init__(
        self,
        *args,
        regularization_lambda_alpha=0.01,
        regularization_lambda_cores=0.5,
        **kwargs,
    ):
        coefficients = {
            "regularization_lambda_alpha": regularization_lambda_alpha,
            "regularization_lambda_cores": regularization_lambda_cores,
        }
        for name, value in coefficients.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(*coefficients)

    @torch.no_grad()
    def _regularization(self, log_beta, log_r):
        alpha_strength = self.hparams.regularization_lambda_alpha
        cores_strength = self.hparams.regularization_lambda_cores
        if alpha_strength == 0 and cores_strength == 0:
            return super()._regularization(log_beta, log_r)
        if not torch.isfinite(log_beta).all() or not torch.isfinite(log_r).all():
            raise ValueError("Log regularization requires strictly positive parameters")
        return (alpha_strength * _exponential_log_penalty(log_beta).mean()
                + cores_strength * _exponential_log_penalty(log_r).mean())

    def _update_block(self, log_a, log_b, current_log_z, *, block):
        if block == "alpha":
            strength = (
                self.hparams.regularization_lambda_alpha
                / self.log_alpha.numel()
            )
        elif block == "cores":
            strength = (
                self.hparams.regularization_lambda_cores
                / self.log_cp_cores.numel()
            )
        else:
            raise ValueError(f"Unknown MM parameter block: {block}")
        if strength == 0:
            return super()._update_block(
                log_a, log_b, current_log_z, block=block
            )
        return _exponential_log_update(
            log_a,
            log_b,
            current_log_z,
            strength,
            self.hparams.min_component_value,
        )

    @torch.no_grad()
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
