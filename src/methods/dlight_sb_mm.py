"""Block-MM for the unregularized DLightSB loss (see theory.md).

Usage:
    solver = DLightSBMM.from_model(model, state_dict=initial_state)
    history = solver.fit_mm(train_x, train_y, max_iter=200)
    predictions = solver.sample(test_x)

Lightning calls training_step: one batch is one complete MM update.
The update monotonically decreases the empirical loss of that current batch.
With SampledCoupleDataset, successive steps may use fresh marginal samples.
The original loss, prior and sampling methods are inherited unchanged.
"""

from copy import deepcopy
import math
import time

try:
    import resource
except ImportError:  # resource is unavailable in native Windows Python
    resource = None

import torch
from catsbench.utils import lse_matmul

from .dlight_sb import DLightSB


def _support_preserving_update_core(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    current_log_z: torch.Tensor,
    log_floor: float,
    clip_updates: bool,
) -> torch.Tensor:
    """Tensor-only ordinary MM block kernel suitable for ``torch.compile``."""
    positive_a = torch.isfinite(log_a)
    positive_b = torch.isfinite(log_b)
    interior = positive_a & positive_b
    interior_log_z = log_b - log_a
    log_z = current_log_z
    if clip_updates:
        log_z = log_z.clamp_min(log_floor)
        interior_log_z = interior_log_z.clamp_min(log_floor)
    return torch.where(interior, interior_log_z, log_z)


_compiled_support_preserving_update_core = torch.compile(
    _support_preserving_update_core,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


@torch.no_grad()
def _support_preserving_update_from_logs(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    current_log_z: torch.Tensor,
    min_component_value: float = 1e-12,
    clip_updates: bool = True,
    compile_kernel: bool = True,
) -> torch.Tensor:
    """Apply a support-preserving MM update to ``a*z - b*log(z)``.

    The explicit solution is ``z = b / a`` when both coefficients are
    positive. When ``clip_updates`` is enabled, the result is clipped to
    ``z >= min_component_value``. Inputs and the result stay in the log domain.
    If ``b = 0``, the current value is preserved, with the optional floor
    applied. This keeps support for categories that may occur in later
    stochastic batches. The remaining case, ``a = 0 < b``, still has no finite
    minimizer and is rejected explicitly.
    """
    if log_a.shape != log_b.shape or log_a.shape != current_log_z.shape:
        raise ValueError("log_a, log_b and current_log_z must have the same shape")
    if not isinstance(clip_updates, bool):
        raise ValueError("clip_updates must be a boolean")
    if not math.isfinite(min_component_value) or min_component_value <= 0:
        raise ValueError("min_component_value must be finite and positive")
    if not isinstance(compile_kernel, bool):
        raise ValueError("compile_kernel must be a boolean")
    # Python truth conversion synchronizes CUDA. The final candidate check is
    # the single synchronization point for CUDA, while CPU keeps rich errors.
    if not log_a.is_cuda:
        for value in (log_a, log_b, current_log_z):
            if torch.isnan(value).any() or torch.isposinf(value).any():
                raise ValueError("Log parameters may contain finite values and -inf only")
        positive_a = torch.isfinite(log_a)
        positive_b = torch.isfinite(log_b)
        if ((~positive_a) & positive_b).any():
            raise FloatingPointError(
                "The nonnegative MM block is unbounded: a=0 and b>0"
            )

    kernel = (
        _compiled_support_preserving_update_core
        if compile_kernel and log_a.is_cuda
        else _support_preserving_update_core
    )
    result = kernel(
        log_a,
        log_b,
        current_log_z,
        math.log(min_component_value),
        clip_updates,
    )
    # reduce-overhead may reuse CUDA Graph output buffers across block calls.
    return result.clone() if compile_kernel and log_a.is_cuda else result


def _exponential_log_update_core(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    log_strength: torch.Tensor,
    log_floor: float,
    clip_updates: bool,
) -> torch.Tensor:
    result = torch.logaddexp(log_b, log_strength) - torch.logaddexp(
        log_a, log_strength
    )
    return result.clamp_min(log_floor) if clip_updates else result


_compiled_exponential_log_update_core = torch.compile(
    _exponential_log_update_core,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


@torch.no_grad()
def _exponential_log_update(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    current_log_z: torch.Tensor,
    strength: float,
    min_component_value: float = 1e-12,
    clip_updates: bool = True,
    compile_kernel: bool = True,
) -> torch.Tensor:
    """Minimize the normalized exponential-regularized scalar MM bound."""
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if not isinstance(clip_updates, bool):
        raise ValueError("clip_updates must be a boolean")
    if not isinstance(compile_kernel, bool):
        raise ValueError("compile_kernel must be a boolean")
    if not math.isfinite(min_component_value) or min_component_value <= 0:
        raise ValueError("min_component_value must be finite and positive")
    if log_a.shape != log_b.shape or log_a.shape != current_log_z.shape:
        raise ValueError("Block tensors must have the same shape")
    if not log_a.is_cuda:
        if not torch.isfinite(current_log_z).all():
            raise ValueError("Log regularization requires strictly positive parameters")
        for value in (log_a, log_b):
            if torch.isnan(value).any() or torch.isposinf(value).any():
                raise ValueError("Log coefficients must be finite or -inf")

    log_strength = log_a.new_tensor(math.log(strength))
    kernel = (
        _compiled_exponential_log_update_core
        if compile_kernel and log_a.is_cuda
        else _exponential_log_update_core
    )
    result = kernel(
        log_a,
        log_b,
        log_strength,
        math.log(min_component_value),
        clip_updates,
    )
    return result.clone() if compile_kernel and log_a.is_cuda else result


@torch.no_grad()
def _exponential_log_penalty(u: torch.Tensor) -> torch.Tensor:
    """Evaluate ``exp(u) - u - 1`` accurately near zero."""
    small = u.abs() < 1e-3
    local = torch.where(small, u, 0.0)
    series = local.square() * (0.5 + local / 6 + local.square() / 24)
    return torch.where(small, series, torch.expm1(u) - u)


@torch.no_grad()
def _project_potentials_to_simplex(
    log_beta: torch.Tensor,
    log_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize all cores and beta while preserving the potential up to scale."""
    if log_beta.ndim != 1 or log_r.ndim != 3:
        raise ValueError("Expected log_beta [K] and log_r [D, S, K]")
    if log_r.shape[-1] != log_beta.shape[0]:
        raise ValueError("Beta and core component counts must match")
    if not log_beta.is_cuda:
        for value in (log_beta, log_r):
            if torch.isnan(value).any() or torch.isposinf(value).any():
                raise ValueError("Log parameters may contain finite values and -inf only")

    log_core_mass = torch.logsumexp(log_r, dim=1)
    projected_log_r = log_r - log_core_mass[:, None, :]
    compensated_log_beta = log_beta + log_core_mass.sum(dim=0)
    projected_log_beta = compensated_log_beta - torch.logsumexp(
        compensated_log_beta, dim=0
    )
    return projected_log_beta, projected_log_r


def _grouped_logsumexp_core(
    log_values: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Vectorized group-wise logsumexp used for MM sufficient statistics."""
    index = groups[:, None].expand_as(log_values)
    maxima = log_values.new_full(
        (num_groups, log_values.shape[1]), -torch.inf
    )
    maxima.scatter_reduce_(
        0, index, log_values, reduce="amax", include_self=True
    )
    gathered_maxima = maxima.gather(0, index)
    terms = torch.where(
        torch.isfinite(log_values),
        (log_values - gathered_maxima).exp(),
        torch.zeros_like(log_values),
    )
    totals = torch.zeros_like(maxima).scatter_add_(0, index, terms)
    return torch.where(totals > 0, maxima + totals.log(), maxima)


_compiled_grouped_logsumexp_core = torch.compile(
    _grouped_logsumexp_core,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


@torch.no_grad()
def _grouped_logsumexp(
    log_values: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
    compile_kernel: bool = True,
) -> torch.Tensor:
    if log_values.ndim != 2 or groups.ndim != 1 or len(groups) != len(log_values):
        raise ValueError("Expected log_values [N,K] and groups [N]")
    kernel = (
        _compiled_grouped_logsumexp_core
        if compile_kernel and log_values.is_cuda
        else _grouped_logsumexp_core
    )
    result = kernel(log_values, groups, num_groups)
    return result.clone() if compile_kernel and log_values.is_cuda else result


@torch.no_grad()
def _logsumexp_matmul(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Use the shared Triton LSE kernel on CUDA and its CPU reference fallback."""
    return lse_matmul(log_a, log_b, implementation="triton")

class _MMOptimizer(torch.optim.Optimizer):
    """Execute an analytic update through Lightning's optimizer bookkeeping.

    No gradient or dummy SGD step: the closure itself updates the parameters.
    The Lightning wrapper advances global_step and supports max_steps/checkpoints.
    """

    def __init__(self, params):
        super().__init__(params, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("An MM update closure is required")
        return closure()


class DLightSBMM(DLightSB):
    """Configurable ordinary block-MM for the DLightSB objective.

    The bound uses the tangent to log(c) and Jensen for -log(v).
    beta is updated first, then dimensions sequentially, with fixed E-step
    statistics. Separate normalized exponential penalties can be applied to
    alpha and cores. Zero coefficients recover ordinary unregularized MM.
    Clipping, compiled block kernels and an optional post-step simplex
    reparameterization are controlled by configuration.
    """

    def __init__(
        self,
        *args,
        inner_sweeps=1,
        tol=None,
        min_component_value=1e-12,
        clip_updates=True,
        regularization_lambda_alpha=0.0,
        regularization_lambda_cores=0.0,
        project_to_simplex_after_step=False,
        compile_update_kernels=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.hparams.entropy_lambda != 0 or self.hparams.scheduler is not None:
            raise ValueError("MM requires entropy_lambda=0 and scheduler=None")
        if not isinstance(inner_sweeps, int) or inner_sweeps < 1:
            raise ValueError("inner_sweeps must be a positive integer")
        if tol is not None and (not math.isfinite(tol) or tol < 0):
            raise ValueError("tol must be None or a finite nonnegative number")
        if not math.isfinite(min_component_value) or min_component_value <= 0:
            raise ValueError("min_component_value must be finite and positive")
        if not isinstance(clip_updates, bool):
            raise ValueError("clip_updates must be a boolean")
        coefficients = {
            "regularization_lambda_alpha": regularization_lambda_alpha,
            "regularization_lambda_cores": regularization_lambda_cores,
        }
        for name, value in coefficients.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(project_to_simplex_after_step, bool):
            raise ValueError("project_to_simplex_after_step must be a boolean")
        if not isinstance(compile_update_kernels, bool):
            raise ValueError("compile_update_kernels must be a boolean")
        if project_to_simplex_after_step and any(coefficients.values()):
            raise ValueError(
                "Post-step simplex projection is valid only with zero regularization"
            )
        self.save_hyperparameters(
            "inner_sweeps",
            "tol",
            "min_component_value",
            "clip_updates",
            "regularization_lambda_alpha",
            "regularization_lambda_cores",
            "project_to_simplex_after_step",
            "compile_update_kernels",
        )
        self.automatic_optimization = False
        self.to(dtype=self.prior.dtype)
        self.requires_grad_(False)

    def configure_optimizers(self):
        return _MMOptimizer([self.log_alpha, self.log_cp_cores])

    @torch.no_grad()
    def _update_block(self, log_a, log_b, current_log_z, *, block):
        if block not in {"alpha", "cores"}:
            raise ValueError(f"Unknown MM parameter block: {block}")
        strength = (
            self.hparams.regularization_lambda_alpha / self.log_alpha.numel()
            if block == "alpha"
            else self.hparams.regularization_lambda_cores
            / self.log_cp_cores.numel()
        )
        if strength > 0:
            return _exponential_log_update(
                log_a,
                log_b,
                current_log_z,
                strength,
                self.hparams.min_component_value,
                self.hparams.clip_updates,
                self.hparams.compile_update_kernels,
            )
        return _support_preserving_update_from_logs(
            log_a,
            log_b,
            current_log_z,
            self.hparams.min_component_value,
            self.hparams.clip_updates,
            self.hparams.compile_update_kernels,
        )

    @torch.no_grad()
    def _regularization(self, log_beta, log_r):
        """Same penalty is added to the objective and its touching bound."""
        alpha_strength = self.hparams.regularization_lambda_alpha
        cores_strength = self.hparams.regularization_lambda_cores
        if alpha_strength == 0 and cores_strength == 0:
            return log_beta.new_zeros(())
        if not log_beta.is_cuda and (
            not torch.isfinite(log_beta).all() or not torch.isfinite(log_r).all()
        ):
            raise ValueError("Log regularization requires strictly positive parameters")
        return (
            alpha_strength * _exponential_log_penalty(log_beta).mean()
            + cores_strength * _exponential_log_penalty(log_r).mean()
        )

    @torch.no_grad()
    def loss(self, true_x_start, true_x_end):
        data_loss, info = super().loss(true_x_start, true_x_end)
        penalty = self._regularization(self.log_alpha, self.log_cp_cores)
        return data_loss + penalty, {
            **info,
            "data_loss": data_loss,
            "regularization": penalty,
        }

    def on_train_start(self):
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("MM requires float32 or float64 parameters")
        if self.prior.dtype != self.dtype:
            raise ValueError("MM parameters and prior must use the same dtype")
        if self.hparams.tol is not None and self.trainer.num_training_batches != 1:
            raise ValueError("MM tolerance requires one full-dataset batch per epoch")

    @torch.no_grad()
    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        update_started = time.perf_counter()
        info = self.optimizers().step(closure=lambda: self.mm_step(x0, x1))
        info["update_seconds"] = time.perf_counter() - update_started
        # Linux reports ru_maxrss in KiB. The supported experiment environments
        # are Linux/WSL/Colab, so expose the process peak directly in MiB.
        if resource is not None:
            info["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        info["batch_size"] = len(x0)
        # Manual optimization advances global_step inside optimizer.step().
        info["samples_seen"] = self.global_step * len(x0)
        # Compare the same empirical objective before/after the accepted step.
        # None disables convergence stopping; zero requires exact equality.
        relative_change = abs(info["loss"] - info["loss_before"]) / max(1.0, abs(info["loss_before"]))
        converged = self.hparams.tol is not None and relative_change <= self.hparams.tol
        info.update(relative_loss_change=relative_change, converged=float(converged))
        if converged:
            # Finish this step normally, including logging and final callbacks.
            # Lightning also respects any configured min_steps/min_epochs.
            self.trainer.should_stop = True
        self.log_dict({f"train/{k}": v for k, v in info.items()},
                      on_step=True, on_epoch=False, batch_size=len(x0))
        self.log("train/iteration", self.iteration, prog_bar=True)
        return {"loss": self.log_alpha.new_tensor(info["loss"]), "batch": batch}

    @torch.no_grad()
    def compilation_warmup_step(self, batch, optimizer) -> None:
        """Execute the complete analytic MM path without Lightning logging."""
        x0, x1 = batch
        optimizer.step(closure=lambda: self.mm_step(x0, x1))

    @classmethod
    def from_model(cls, model: DLightSB, *, state_dict=None, tol=None, **solver_kwargs):
        """Copy a DLightSB; optionally restore its pre-training weights.

        The copied solver preserves the model's device and floating-point dtype.
        """
        if model.hparams.entropy_lambda != 0:
            raise ValueError("MM supports entropy_lambda=0 only")
        keys = (
            "dim", "num_categories", "num_potentials", "num_timesteps",
            "distr_init", "entropy_warmup_steps",
            "entropy_lambda", "sample_prob", "tau", "freeze_beta",
        )
        with torch.device(model.device):
            solver = cls(
                prior=deepcopy(model.prior),
                optimizer=None, scheduler=None, tol=tol, **solver_kwargs,
                **{key: model.hparams[key] for key in keys},
            ).to(device=model.device, dtype=model.dtype)
        solver.load_state_dict(model.state_dict() if state_dict is None else state_dict)
        solver._did_weight_init = True
        return solver

    @torch.no_grad()
    def fit_mm(self, x0, x1, *, max_iter=200, tol=1e-7, inner_sweeps=1):
        """Backward-compatible notebook interface, using the same MM step.

        New experiments should use Trainer.fit. Returns loss including step 0.
        """
        if max_iter < 1 or inner_sweeps < 1 or tol < 0:
            raise ValueError("Require max_iter >= 1, inner_sweeps >= 1 and tol >= 0")
        self.history_, self.surrogate_decreases_ = [], []
        for _ in range(max_iter):
            info = self.mm_step(x0, x1, inner_sweeps=inner_sweeps)
            if not self.history_:
                self.history_.append(info["loss_before"])
            self.history_.append(info["loss"])
            self.surrogate_decreases_.append(info["surrogate_decrease"])
            if abs(info["loss_before"] - info["loss"]) <= tol * max(1.0, abs(info["loss_before"])):
                break
        self.eval()
        return self.history_

    @torch.no_grad()
    def mm_step(self, x0, x1, *, inner_sweeps=None):
        """One MM update of the empirical loss on the supplied marginals.

        Every observation is retained with its empirical uniform weight. A new
        bound is built once, then minimized blockwise. Commit only a checked
        candidate. No persistent data cache: another batch/checkpoint cannot
        reuse stale Q.
        """
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("MM requires float32 or float64 parameters")
        if self.prior.dtype != self.dtype:
            raise ValueError("MM parameters and prior must use the same dtype")
        if self.hparams.entropy_lambda != 0:
            raise ValueError("MM supports entropy_lambda=0 only")
        inner_sweeps = self.hparams.inner_sweeps if inner_sweeps is None else inner_sweeps
        if not isinstance(inner_sweeps, int) or inner_sweeps < 1:
            raise ValueError("inner_sweeps must be a positive integer")
        for x in (x0, x1):
            if not isinstance(x, torch.Tensor) or x.device != self.device:
                raise ValueError("Training data and MM parameters must use the same device")
            if x.ndim < 2 or x.flatten(1).shape[1] != self.hparams.dim or len(x) == 0:
                raise ValueError("Training data must flatten to nonempty [N, dim] tensors")
            if x.dtype != torch.long or x.min() < 0 or x.max() >= self.hparams.num_categories:
                raise ValueError("Use torch.long categories in [0, num_categories)")
        if not self._did_weight_init and not self._loaded_from_ckpt:
            raise ValueError("Call init_weights first, or use DLightSBMM.from_model")

        x0, x1 = x0.flatten(1), x1.flatten(1)
        source = x0
        target = x1
        weight0 = self.log_alpha.new_full((len(source),), 1.0 / len(source))
        weight1 = self.log_alpha.new_full((len(target),), 1.0 / len(target))
        dim, categories, components = self.log_cp_cores.shape
        log_q = self.prior.extract_last_cum_matrix(source).permute(1, 0, 2).contiguous()
        if not torch.isfinite(log_q).all():
            raise ValueError("MM requires a strictly positive finite reference transition")

        # Work with the model's native nonnegative, unnormalized parameters.
        # A value of -inf is an exact zero and is allowed by the MM domain.
        log_r = self.log_cp_cores.detach().clone()
        log_beta = self.log_alpha.detach().clone()
        invalid_log_r = torch.isnan(log_r).any() or torch.isposinf(log_r).any()
        invalid_log_beta = torch.isnan(log_beta).any() or torch.isposinf(log_beta).any()
        if invalid_log_r or invalid_log_beta or not torch.isfinite(log_beta).any():
            raise ValueError("Invalid initial potential parameters")
        log_u = torch.stack([
            _logsumexp_matmul(log_q[d], log_r[d]) for d in range(dim)
        ])  # [D, N0, K]

        def log_normalizer():
            return torch.logsumexp(log_u.sum(dim=0) + log_beta, dim=1)

        def target_logits():
            logits = log_beta.expand(len(target), components).clone()
            for d in range(dim):
                logits += log_r[d, target[:, d]]
            return logits

        def objective_terms():
            source_term = log_normalizer()
            target_term = torch.logsumexp(target_logits(), dim=1)
            return source_term, target_term

        def objective():
            source_term, target_term = objective_terms()
            return (weight0 @ source_term - weight1 @ target_term
                    + self._regularization(log_beta, log_r))

        initial_loss = objective()
        if not torch.isfinite(initial_loss):
            source_term, target_term = objective_terms()
            raise FloatingPointError(
                "Non-finite initial MM objective: "
                f"log_u_finite={bool(torch.isfinite(log_u).all())}, "
                f"source_finite={bool(torch.isfinite(source_term).all())}, "
                f"target_finite={bool(torch.isfinite(target_term).all())}"
            )
        old_log_c = log_normalizer().clone()
        log_weight0 = weight0.log()
        log_weight1 = weight1.log()
        log_gamma = target_logits().log_softmax(dim=1)
        log_weighted_gamma = log_weight1[:, None] + log_gamma
        log_n = torch.logsumexp(log_weighted_gamma, dim=0)
        log_m = torch.stack(
            [
                _grouped_logsumexp(
                    log_weighted_gamma,
                    target[:, d],
                    categories,
                    self.hparams.compile_update_kernels,
                )
                for d in range(dim)
            ]
        )

        def surrogate():
            # Constant C_t is unnecessary when comparing this same bound.
            ratio = (log_normalizer() - old_log_c).exp()
            beta_term = torch.where(
                torch.isfinite(log_n), log_n.exp() * log_beta, 0.0
            ).sum()
            core_term = torch.where(
                torch.isfinite(log_m), log_m.exp() * log_r, 0.0
            ).sum()
            return (weight0 @ ratio - beta_term - core_term
                    + self._regularization(log_beta, log_r))

        bound_before_tensor = surrogate()
        penalty_before_tensor = self._regularization(log_beta, log_r)
        for _ in range(inner_sweeps):
            if not self.hparams.freeze_beta:
                log_product = log_u.sum(dim=0)
                log_a = torch.logsumexp(
                    log_weight0[:, None] + log_product - old_log_c[:, None], dim=0
                )
                log_beta = self._update_block(
                    log_a, log_n, log_beta, block="alpha"
                )
            for d in range(dim):
                # Do not form sum(log_u) - log_u[d]: exact-zero components
                # would produce -inf - -inf = NaN. Sum the other dimensions.
                log_other = sum(
                    (log_u[e] for e in range(dim) if e != d),
                    torch.zeros_like(log_u[d]),
                )
                log_coefficients = (
                    log_weight0[:, None]
                    + log_beta[None]
                    + log_other
                    - old_log_c[:, None]
                )
                log_b_linear = _logsumexp_matmul(
                    log_q[d].T, log_coefficients
                )  # [S,K]
                log_r[d] = self._update_block(
                    log_b_linear.T, log_m[d].T, log_r[d].T, block="cores"
                ).T
                log_u[d] = _logsumexp_matmul(log_q[d], log_r[d])

        bound_after_tensor = surrogate()
        candidate_loss_tensor = objective()
        projection_delta_tensor = candidate_loss_tensor.new_zeros(())
        simplex_beta_residual = candidate_loss_tensor.new_zeros(())
        simplex_core_residual = candidate_loss_tensor.new_zeros(())
        if self.hparams.project_to_simplex_after_step:
            log_beta, log_r = _project_potentials_to_simplex(log_beta, log_r)
            log_u = torch.stack(
                [_logsumexp_matmul(log_q[d], log_r[d]) for d in range(dim)]
            )
            new_loss_tensor = objective()
            projection_delta_tensor = new_loss_tensor - candidate_loss_tensor
            simplex_beta_residual = log_beta.exp().sum().sub(1).abs()
            simplex_core_residual = log_r.exp().sum(dim=1).sub(1).abs().amax()
        else:
            new_loss_tensor = candidate_loss_tensor
        penalty_after_tensor = self._regularization(log_beta, log_r)
        (
            previous_loss,
            bound_before,
            bound_after,
            candidate_loss,
            new_loss,
            penalty_before,
            penalty_after,
            projection_delta,
            beta_residual,
            core_residual,
        ) = torch.stack(
            (
                initial_loss,
                bound_before_tensor,
                bound_after_tensor,
                candidate_loss_tensor,
                new_loss_tensor,
                penalty_before_tensor,
                penalty_after_tensor,
                projection_delta_tensor,
                simplex_beta_residual,
                simplex_core_residual,
            )
        ).detach().cpu().tolist()
        scale = max(1.0, abs(previous_loss), abs(bound_before))
        allowance = 64 * torch.finfo(self.dtype).eps * scale
        if (
            not math.isfinite(bound_after) or not math.isfinite(new_loss)
            or bound_after > bound_before + allowance
            or candidate_loss > previous_loss + allowance
        ):
            raise FloatingPointError(
                f"MM step failed monotonicity: "
                f"G {bound_before:.16g} -> {bound_after:.16g}, "
                f"loss {previous_loss:.16g} -> {new_loss:.16g}; "
                "no candidate weights committed"
            )
        if self.hparams.project_to_simplex_after_step:
            projection_scale = max(1.0, abs(candidate_loss), abs(new_loss))
            projection_allowance = (
                128 * torch.finfo(self.dtype).eps * projection_scale
            )
            if abs(projection_delta) > projection_allowance:
                raise FloatingPointError(
                    "Simplex projection changed the scale-invariant loss: "
                    f"{candidate_loss:.16g} -> {new_loss:.16g}; "
                    "no candidate weights committed"
                )

        self.log_alpha.copy_(log_beta)
        self.log_cp_cores.copy_(log_r)
        info = {
            "loss": new_loss,
            "loss_before": previous_loss,
            "data_loss": new_loss - penalty_after,
            "data_loss_before": previous_loss - penalty_before,
            "regularization": penalty_after,
            "regularization_before": penalty_before,
            "surrogate_decrease": bound_before - bound_after,
        }
        if self.hparams.project_to_simplex_after_step:
            info.update(
                projection_loss_delta=projection_delta,
                simplex_beta_residual=beta_residual,
                simplex_core_residual_max=core_residual,
            )
        return info
