"""Block-MM for the unregularized DLightSB loss (see theory.md).

Usage:
    solver = DLightSBMM.from_model(model, state_dict=initial_state)
    history = solver.fit_mm(train_x, train_y, max_iter=200)
    predictions = solver.sample(test_x)

Lightning calls training_step: one batch is one complete MM update.
Finite Newton candidates are committed even when the inexact step is nonmonotone.
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


@torch.no_grad()
def _simplex_newton_step_from_logs(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    initial_log_z: torch.Tensor,
    previous_lambda: torch.Tensor,
    initialized: torch.Tensor,
):
    """Take one safeguarded Newton step for each simplex multiplier.

    Coefficients are rescaled row-wise for numerical stability. The stored
    multiplier uses this dimensionless coordinate system, which is invariant
    to a common rescaling of all coefficients in a row.
    """
    if (
        log_a.shape != log_b.shape
        or log_a.shape != initial_log_z.shape
        or log_a.ndim != 2
    ):
        raise ValueError("log_a, log_b and initial_log_z must have the same 2D shape")
    expected_state_shape = (log_a.shape[0], 1)
    for name, value in (
        ("previous_lambda", previous_lambda),
        ("initialized", initialized),
    ):
        if value.shape != expected_state_shape:
            raise ValueError(f"{name} must have shape {expected_state_shape}")
    for value in (log_a, log_b):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log coefficients may contain finite values and -inf only")

    log_scale = torch.maximum(
        log_a.amax(dim=1, keepdim=True),
        log_b.amax(dim=1, keepdim=True),
    )
    log_scale = torch.where(torch.isfinite(log_scale), log_scale, 0.0)
    a = torch.exp(log_a - log_scale)
    b = torch.exp(log_b - log_scale)
    smallest = torch.nextafter(b.new_tensor(0.0), b.new_tensor(1.0))
    b = torch.where(torch.isfinite(log_b) & (b == 0), smallest, b)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (b < 0).any():
        raise ValueError("Expected finite coefficients and nonnegative counts")

    positive = b > 0
    has_counts = positive.any(dim=1, keepdim=True)
    minimum = a.masked_fill(~positive, torch.inf).amin(dim=1, keepdim=True)
    minimum = torch.where(has_counts, minimum, a.amin(dim=1, keepdim=True))
    shifted = a - minimum
    lower_delta = b.masked_fill(shifted != 0, 0).sum(dim=1, keepdim=True)
    upper_delta = b.sum(dim=1, keepdim=True)
    initial_z = initial_log_z.exp()
    bootstrap_lambda = b.sum(dim=1, keepdim=True) - (a * initial_z).sum(
        dim=1,
        keepdim=True,
    )

    lambda_start = torch.where(initialized, previous_lambda, bootstrap_lambda)
    # Work with delta = lambda + min(a). Forming lambda near -min(a) and
    # adding min(a) back can erase a tiny positive delta in float32, making
    # an active denominator exactly zero. The shifted coordinate avoids that
    # cancellation while describing the identical Newton equation.
    delta_start = torch.maximum(
        torch.minimum(lambda_start + minimum, upper_delta),
        lower_delta,
    )
    denominator = torch.where(positive, shifted + delta_start, torch.ones_like(a))
    mass = (b / denominator).sum(dim=1, keepdim=True)
    curvature = (b / denominator.square()).sum(dim=1, keepdim=True)
    correction = torch.where(has_counts, (mass - 1) / curvature, 0.0)
    delta_new = delta_start + correction
    delta_new = torch.maximum(
        torch.minimum(delta_new, upper_delta),
        lower_delta,
    )

    minimum_zero = a.masked_fill(positive, torch.inf).amin(dim=1, keepdim=True)
    boundary_delta = minimum - minimum_zero
    boundary = has_counts & (boundary_delta > delta_new)
    delta_new = torch.where(boundary, boundary_delta, delta_new)
    lambda_new = delta_new - minimum
    denominator = torch.where(positive, shifted + delta_new, torch.ones_like(a))
    z = b / denominator
    zero_minima = ~positive & (a == minimum_zero)
    residual_mass = (1 - z.sum(dim=1, keepdim=True)).clamp_min(0)
    z = z + torch.where(
        boundary,
        zero_minima * residual_mass / zero_minima.sum(dim=1, keepdim=True).clamp_min(1),
        torch.zeros_like(z),
    )
    minima = a == a.amin(dim=1, keepdim=True)
    linear_solution = minima.to(a.dtype) / minima.sum(dim=1, keepdim=True)
    z = torch.where(has_counts, z, linear_solution)
    z = torch.where(z == 0, smallest, z)

    return {
        "log_z": z.log(),
        "lambda": lambda_new,
        "simplex_residual": (z.sum(dim=1, keepdim=True) - 1).abs(),
        "newton_correction": correction.abs(),
    }


def _log_newton_iteration(
    log_delta,
    scaled_log_b,
    log_shifted,
    log_lower,
    log_upper,
    mu0,
):
    """Apply Newton or a monotone fixed-point step according to conditioning."""
    log_denominator = torch.logaddexp(log_shifted, log_delta)
    log_mass = torch.logsumexp(
        scaled_log_b - log_denominator,
        dim=1,
        keepdim=True,
    )
    log_second_sum = torch.logsumexp(
        scaled_log_b - 2 * log_denominator,
        dim=1,
        keepdim=True,
    )
    log_mu = log_delta + log_second_sum - log_mass
    mu = log_mu.exp()
    newton_correction = log_mass / mu
    use_newton = (mu >= mu0) | (log_mass.abs() <= mu)
    correction = torch.where(use_newton, newton_correction, log_mass)
    candidate = log_delta + correction
    candidate = torch.where(torch.isfinite(candidate), candidate, log_delta)
    return (
        torch.maximum(torch.minimum(candidate, log_upper), log_lower),
        correction.abs(),
    )


def _simplex_log_newton_4_core(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    initial_log_z: torch.Tensor,
    previous_lambda: torch.Tensor,
    initialized: torch.Tensor,
    mu0: float,
):
    """Run four fused condition-aware root corrections after shared preparation."""
    finite_b = torch.isfinite(log_b)
    has_counts = finite_b.any(dim=1, keepdim=True)

    log_scale = torch.maximum(
        log_a.amax(dim=1, keepdim=True),
        log_b.amax(dim=1, keepdim=True),
    )
    log_scale = torch.where(torch.isfinite(log_scale), log_scale, 0.0)
    scaled_log_a = log_a - log_scale
    scaled_log_b = log_b - log_scale

    active_minimum_log_a = scaled_log_a.masked_fill(
        ~finite_b,
        torch.inf,
    ).amin(dim=1, keepdim=True)
    all_minimum_log_a = scaled_log_a.amin(dim=1, keepdim=True)
    minimum_log_a = torch.where(
        has_counts,
        active_minimum_log_a,
        all_minimum_log_a,
    )

    greater = scaled_log_a > minimum_log_a
    log_shifted = torch.where(
        greater,
        scaled_log_a
        + torch.log1p(-torch.exp(minimum_log_a - scaled_log_a)),
        -torch.inf,
    )
    at_minimum = finite_b & ~greater
    active_log_lower = torch.logsumexp(
        scaled_log_b.masked_fill(~at_minimum, -torch.inf),
        dim=1,
        keepdim=True,
    )
    log_upper = torch.logsumexp(
        scaled_log_b.masked_fill(~finite_b, -torch.inf),
        dim=1,
        keepdim=True,
    )

    zero_count = ~finite_b
    has_zero_count = zero_count.any(dim=1, keepdim=True)
    minimum_zero_log_a = scaled_log_a.masked_fill(
        ~zero_count,
        torch.inf,
    ).amin(dim=1, keepdim=True)
    has_boundary = (
        has_counts
        & has_zero_count
        & (minimum_log_a > minimum_zero_log_a)
    )
    boundary_log_ratio = torch.where(
        has_boundary,
        minimum_zero_log_a - minimum_log_a,
        torch.zeros_like(minimum_log_a),
    )
    log_boundary_delta = torch.where(
        has_boundary,
        minimum_log_a + torch.log1p(-boundary_log_ratio.exp()),
        torch.full_like(minimum_log_a, -torch.inf),
    )

    zero = torch.zeros_like(log_upper)
    active_log_lower = torch.where(has_counts, active_log_lower, zero)
    log_upper = torch.where(has_counts, log_upper, zero)
    log_lower = torch.maximum(active_log_lower, log_boundary_delta)

    shifted_a = log_shifted.exp()
    initial_z = initial_log_z.exp()
    upper_delta = log_upper.exp()
    bootstrap_delta = upper_delta - (shifted_a * initial_z).sum(
        dim=1,
        keepdim=True,
    )
    previous_delta = previous_lambda + minimum_log_a.exp()
    delta_start = torch.where(initialized, previous_delta, bootstrap_delta)
    positive_start = delta_start > 0
    safe_delta_start = torch.where(
        positive_start,
        delta_start,
        torch.ones_like(delta_start),
    )
    log_delta = torch.where(
        positive_start,
        safe_delta_start.log(),
        log_lower,
    )
    log_delta = torch.maximum(torch.minimum(log_delta, log_upper), log_lower)

    log_delta, correction_1 = _log_newton_iteration(
        log_delta,
        scaled_log_b,
        log_shifted,
        log_lower,
        log_upper,
        mu0,
    )
    log_delta, correction_2 = _log_newton_iteration(
        log_delta,
        scaled_log_b,
        log_shifted,
        log_lower,
        log_upper,
        mu0,
    )
    log_delta, correction_3 = _log_newton_iteration(
        log_delta,
        scaled_log_b,
        log_shifted,
        log_lower,
        log_upper,
        mu0,
    )
    log_delta, correction_4 = _log_newton_iteration(
        log_delta,
        scaled_log_b,
        log_shifted,
        log_lower,
        log_upper,
        mu0,
    )
    correction_max = torch.maximum(
        torch.maximum(correction_1, correction_2),
        torch.maximum(correction_3, correction_4),
    )

    log_denominator = torch.logaddexp(log_shifted, log_delta)
    active_log_z = scaled_log_b - log_denominator
    smallest = torch.nextafter(
        log_a.new_tensor(0.0),
        log_a.new_tensor(1.0),
    )
    log_smallest = smallest.log()
    active_log_z = torch.where(
        finite_b,
        active_log_z,
        log_smallest,
    )

    boundary = has_boundary & (log_boundary_delta >= log_delta)
    boundary_z = torch.where(
        finite_b,
        (scaled_log_b - log_denominator).exp(),
        torch.zeros_like(log_a),
    )
    zero_minima = zero_count & (scaled_log_a == minimum_zero_log_a)
    residual_mass = (1 - boundary_z.sum(dim=1, keepdim=True)).clamp_min(0)
    boundary_z = boundary_z + (
        zero_minima
        * residual_mass
        / zero_minima.sum(dim=1, keepdim=True).clamp_min(1)
    )
    boundary_z = torch.where(boundary_z == 0, smallest, boundary_z)
    log_z = torch.where(boundary, boundary_z.log(), active_log_z)

    minima = scaled_log_a == all_minimum_log_a
    linear_z = minima.to(log_a.dtype) / minima.sum(dim=1, keepdim=True)
    linear_z = torch.where(linear_z == 0, smallest, linear_z)
    log_z = torch.where(has_counts, log_z, linear_z.log())

    lambda_new = log_delta.exp() - minimum_log_a.exp()
    lambda_new = torch.where(has_counts, lambda_new, previous_lambda)
    correction_max = torch.where(
        has_counts,
        correction_max,
        torch.zeros_like(correction_max),
    )
    return {
        "log_z": log_z,
        "lambda": lambda_new,
        "simplex_residual": (
            log_z.exp().sum(dim=1, keepdim=True) - 1
        ).abs(),
        "newton_correction": correction_max,
    }


_compiled_simplex_log_newton_4_core = torch.compile(
    _simplex_log_newton_4_core,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


@torch.no_grad()
def _simplex_log_newton_4_from_logs(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    initial_log_z: torch.Tensor,
    previous_lambda: torch.Tensor,
    initialized: torch.Tensor,
    mu0: float = 0.03,
):
    """Apply four jointly compiled condition-aware log-Newton corrections."""
    if (
        log_a.shape != log_b.shape
        or log_a.shape != initial_log_z.shape
        or log_a.ndim != 2
    ):
        raise ValueError("log_a, log_b and initial_log_z must have the same 2D shape")
    expected_state_shape = (log_a.shape[0], 1)
    if previous_lambda.shape != expected_state_shape:
        raise ValueError(f"previous_lambda must have shape {expected_state_shape}")
    if initialized.shape != expected_state_shape:
        raise ValueError(f"initialized must have shape {expected_state_shape}")
    if not math.isfinite(mu0) or not 0 <= mu0 <= 1:
        raise ValueError("mu0 must be finite and lie in [0, 1]")

    kernel = (
        _compiled_simplex_log_newton_4_core
        if log_a.is_cuda
        else _simplex_log_newton_4_core
    )
    result = kernel(
        log_a,
        log_b,
        initial_log_z,
        previous_lambda,
        initialized,
        mu0,
    )
    # reduce-overhead uses CUDA Graphs, whose output buffers are reused by the
    # next invocation with the same graph. MM calls this kernel repeatedly for
    # different blocks before all preceding outputs have been consumed, so
    # materialize independent eager tensors at the compiled boundary.
    return {name: value.clone() for name, value in result.items()}


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
    """DLightSB with one inexact block-MM sweep per Lightning training_step.

    The bound uses the tangent to log(c) and Jensen for -log(v).
    beta is updated first, then dimensions sequentially, with fixed E-step
    statistics. Each simplex multiplier gets four jointly compiled corrections
    warm-started from the preceding MM step. Well-conditioned rows use Newton;
    flat rows with a large proposed move use a monotone fixed-point correction.
    No explicit regularization is added.
    """

    def __init__(
        self,
        *args,
        inner_sweeps=1,
        tol=None,
        log_newton_mu0=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.hparams.entropy_lambda != 0 or self.hparams.scheduler is not None:
            raise ValueError("MM requires entropy_lambda=0 and scheduler=None")
        if not isinstance(inner_sweeps, int) or inner_sweeps < 1:
            raise ValueError("inner_sweeps must be a positive integer")
        if tol is not None and (not math.isfinite(tol) or tol < 0):
            raise ValueError("tol must be None or a finite nonnegative number")
        if not math.isfinite(log_newton_mu0) or not 0 <= log_newton_mu0 <= 1:
            raise ValueError("log_newton_mu0 must be finite and lie in [0, 1]")
        self.save_hyperparameters("inner_sweeps", "tol", "log_newton_mu0")
        self.automatic_optimization = False
        self.to(dtype=self.prior.dtype)
        self.requires_grad_(False)
        _, _, components = self.log_cp_cores.shape
        state_options = {"device": self.device, "dtype": self.dtype}
        self.register_buffer(
            "simplex_beta_lambda",
            torch.zeros((1, 1), **state_options),
            persistent=False,
        )
        self.register_buffer(
            "simplex_beta_initialized",
            torch.zeros((1, 1), device=self.device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "simplex_core_lambda",
            torch.zeros((self.hparams.dim, components, 1), **state_options),
            persistent=False,
        )
        self.register_buffer(
            "simplex_core_initialized",
            torch.zeros(
                (self.hparams.dim, components, 1),
                device=self.device,
                dtype=torch.bool,
            ),
            persistent=False,
        )

    @torch.no_grad()
    def loss(self, true_x_start, true_x_end):
        return super().loss(true_x_start, true_x_end)

    def configure_optimizers(self):
        return _MMOptimizer([self.log_alpha, self.log_cp_cores])

    def _update_block(
        self,
        log_a,
        log_b,
        current_log_z,
        previous_lambda,
        initialized,
    ):
        squeeze = log_a.ndim == 1
        if squeeze:
            log_a = log_a.unsqueeze(0)
            log_b = log_b.unsqueeze(0)
            current_log_z = current_log_z.unsqueeze(0)
        result = _simplex_log_newton_4_from_logs(
            log_a,
            log_b,
            current_log_z,
            previous_lambda,
            initialized,
            self.hparams.log_newton_mu0,
        )
        if squeeze:
            result["log_z"] = result["log_z"].squeeze(0)
        return result

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["simplex_newton_state"] = {
            "beta_lambda": self.simplex_beta_lambda.detach().clone(),
            "beta_initialized": self.simplex_beta_initialized.detach().clone(),
            "core_lambda": self.simplex_core_lambda.detach().clone(),
            "core_initialized": self.simplex_core_initialized.detach().clone(),
        }

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        state = checkpoint.get("simplex_newton_state")
        if state is None:
            return
        self.simplex_beta_lambda.copy_(state["beta_lambda"])
        self.simplex_beta_initialized.copy_(state["beta_initialized"])
        self.simplex_core_lambda.copy_(state["core_lambda"])
        self.simplex_core_initialized.copy_(state["core_initialized"])

    def _regularization(self, log_beta, log_r):
        """Same penalty is added to the objective and its touching bound."""
        return log_beta.new_zeros(())

    def _enforce_monotonicity(self):
        """Four inexact log-Newton corrections are not an exact minimizer."""
        return False

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
            "entropy_lambda", "sample_prob", "tau",
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

        log_r = self.log_cp_cores.detach().clone()
        log_beta = self.log_alpha.detach().clone()
        newton_initialized = bool(
            self.simplex_beta_initialized.all()
            and self.simplex_core_initialized.all()
        )
        if not newton_initialized:
            # Canonicalize arbitrary initial DLightSB weights once. Subsequent
            # iterations retain the unnormalized one-step Newton result.
            normalizers = torch.logsumexp(log_r, dim=1, keepdim=True)
            log_r -= normalizers
            log_beta += normalizers.squeeze(1).sum(dim=0)
            log_beta -= torch.logsumexp(log_beta, dim=0)
        invalid_log_r = torch.isnan(log_r).any() or torch.isposinf(log_r).any()
        invalid_log_beta = torch.isnan(log_beta).any() or torch.isposinf(log_beta).any()
        empty_core = ~torch.isfinite(log_r).any(dim=1).all()
        if invalid_log_r or invalid_log_beta or empty_core or not torch.isfinite(log_beta).all():
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
        log_m = log_r.new_full((dim, categories, components), -torch.inf)
        for d in range(dim):
            for category in range(categories):
                selected = log_weighted_gamma[target[:, d] == category]
                if len(selected) > 0:
                    log_m[d, category] = torch.logsumexp(selected, dim=0)

        beta_lambda = self.simplex_beta_lambda.clone()
        beta_initialized = self.simplex_beta_initialized.clone()
        core_lambda = self.simplex_core_lambda.clone()
        core_initialized = self.simplex_core_initialized.clone()
        beta_residual = log_beta.new_zeros((1, 1))
        core_residual = log_r.new_zeros((dim, components, 1))
        correction_max = log_beta.new_zeros(())

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

        bound_before = surrogate().item()
        for _ in range(inner_sweeps):
            log_product = log_u.sum(dim=0)
            log_a = torch.logsumexp(
                log_weight0[:, None] + log_product - old_log_c[:, None], dim=0
            )
            beta_update = self._update_block(
                log_a,
                log_n,
                log_beta,
                beta_lambda,
                beta_initialized,
            )
            log_beta = beta_update["log_z"]
            beta_lambda = beta_update["lambda"]
            beta_initialized = torch.ones_like(beta_initialized)
            beta_residual = beta_update["simplex_residual"]
            correction_max = torch.maximum(
                correction_max,
                beta_update["newton_correction"].amax(),
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
                core_update = self._update_block(
                    log_b_linear.T,
                    log_m[d].T,
                    log_r[d].T,
                    core_lambda[d],
                    core_initialized[d],
                )
                log_r[d] = core_update["log_z"].T
                core_lambda[d] = core_update["lambda"]
                core_initialized[d] = True
                core_residual[d] = core_update["simplex_residual"]
                correction_max = torch.maximum(
                    correction_max,
                    core_update["newton_correction"].amax(),
                )
                log_u[d] = _logsumexp_matmul(log_q[d], log_r[d])

        bound_after, new_loss = surrogate().item(), objective().item()
        previous_loss = initial_loss.item()
        scale = max(1.0, abs(previous_loss), abs(bound_before))
        allowance = 64 * torch.finfo(self.dtype).eps * scale
        nonfinite = not math.isfinite(bound_after) or not math.isfinite(new_loss)
        nonmonotone = (
            bound_after > bound_before + allowance
            or new_loss > previous_loss + allowance
        )
        if nonfinite or (self._enforce_monotonicity() and nonmonotone):
            failed_check = "finiteness" if nonfinite else "monotonicity"
            raise FloatingPointError(
                f"MM step failed {failed_check}: "
                f"G {bound_before:.16g} -> {bound_after:.16g}, "
                f"loss {previous_loss:.16g} -> {new_loss:.16g}; "
                "no candidate weights committed"
            )

        self.log_alpha.copy_(log_beta)
        self.log_cp_cores.copy_(log_r)
        self.simplex_beta_lambda.copy_(beta_lambda)
        self.simplex_beta_initialized.copy_(beta_initialized)
        self.simplex_core_lambda.copy_(core_lambda)
        self.simplex_core_initialized.copy_(core_initialized)
        return {"loss": new_loss, "loss_before": previous_loss,
                "surrogate_decrease": bound_before - bound_after,
                "simplex_beta_residual": beta_residual.amax().item(),
                "simplex_core_residual_max": core_residual.amax().item(),
                "simplex_core_residual_mean": core_residual.mean().item(),
                "simplex_newton_correction_max": correction_max.item()}
