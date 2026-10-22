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


def _simplex_minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimize ``a*z - b*log(z)`` row-wise on the closed simplex."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must have the same two-dimensional shape")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (b < 0).any():
        raise ValueError("Expected finite coefficients and nonnegative counts")

    positive = b > 0
    has_counts = positive.any(dim=1, keepdim=True)
    minimum = a.masked_fill(~positive, torch.inf).amin(dim=1, keepdim=True)
    minimum = torch.where(has_counts, minimum, a.amin(dim=1, keepdim=True))
    shifted = a - minimum

    lower = b.masked_fill(shifted != 0, 0).sum(dim=1, keepdim=True)
    upper = b.sum(dim=1, keepdim=True)
    lo = torch.where(has_counts, lower, torch.ones_like(lower)).log()
    hi = torch.where(has_counts, upper, torch.ones_like(upper)).log()
    for _ in range(100):
        log_mid = (lo + hi) / 2
        delta = log_mid.exp()
        denominator = torch.where(positive, shifted + delta, torch.ones_like(a))
        mass = (b / denominator).sum(dim=1, keepdim=True)
        lo = torch.where(mass > 1, log_mid, lo)
        hi = torch.where(mass > 1, hi, log_mid)

    delta = ((lo + hi) / 2).exp()
    minimum_zero = a.masked_fill(positive, torch.inf).amin(dim=1, keepdim=True)
    boundary_delta = minimum - minimum_zero
    boundary = has_counts & (boundary_delta > delta)
    delta = torch.where(boundary, boundary_delta, delta)
    denominator = torch.where(positive, shifted + delta, torch.ones_like(a))
    z = b / denominator

    interior = has_counts & ~boundary
    mass = z.sum(dim=1, keepdim=True)
    z = torch.where(interior, z / mass.clamp_min(torch.finfo(a.dtype).tiny), z)
    zero_minima = ~positive & (a == minimum_zero)
    residual = (1 - z.sum(dim=1, keepdim=True)).clamp_min(0)
    z = z + torch.where(
        boundary,
        zero_minima * residual / zero_minima.sum(dim=1, keepdim=True).clamp_min(1),
        torch.zeros_like(z),
    )

    minima = a == a.amin(dim=1, keepdim=True)
    linear_solution = minima.to(a.dtype) / minima.sum(dim=1, keepdim=True)
    z = torch.where(has_counts, z, linear_solution)
    smallest = torch.nextafter(a.new_tensor(0.0), a.new_tensor(1.0))
    return torch.where(z == 0, smallest, z)


@torch.no_grad()
def _simplex_minimum_from_logs(
    log_a: torch.Tensor, log_b: torch.Tensor
) -> torch.Tensor:
    """Solve the simplex block problem from log coefficients."""
    if log_a.shape != log_b.shape or log_a.ndim != 2:
        raise ValueError("log_a and log_b must have the same two-dimensional shape")
    for value in (log_a, log_b):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log coefficients may contain finite values and -inf only")

    shift = torch.maximum(
        log_a.amax(dim=1, keepdim=True),
        log_b.amax(dim=1, keepdim=True),
    )
    shift = torch.where(torch.isfinite(shift), shift, 0.0)
    a = torch.exp(log_a - shift)
    b = torch.exp(log_b - shift)
    smallest = torch.nextafter(b.new_tensor(0.0), b.new_tensor(1.0))
    b = torch.where(torch.isfinite(log_b) & (b == 0), smallest, b)
    return _simplex_minimum(a, b)


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
    """DLightSB with one analytic block-MM sweep per Lightning training_step.

    The bound uses the tangent to log(c) and Jensen for -log(v).
    beta is updated first, then dimensions sequentially, with fixed E-step
    statistics. The component weights and every categorical core are optimized
    on closed simplexes. No explicit regularization is added.
    """

    def __init__(self, *args, inner_sweeps=1, tol=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams.entropy_lambda != 0 or self.hparams.scheduler is not None:
            raise ValueError("MM requires entropy_lambda=0 and scheduler=None")
        if not isinstance(inner_sweeps, int) or inner_sweeps < 1:
            raise ValueError("inner_sweeps must be a positive integer")
        if tol is not None and (not math.isfinite(tol) or tol < 0):
            raise ValueError("tol must be None or a finite nonnegative number")
        self.save_hyperparameters("inner_sweeps", "tol")
        self.automatic_optimization = False
        self.to(dtype=self.prior.dtype)
        self.requires_grad_(False)

    @torch.no_grad()
    def loss(self, true_x_start, true_x_end):
        return super().loss(true_x_start, true_x_end)

    def configure_optimizers(self):
        return _MMOptimizer([self.log_alpha, self.log_cp_cores])

    def _update_block(self, log_a, log_b, current_log_z):
        del current_log_z
        squeeze = log_a.ndim == 1
        if squeeze:
            log_a = log_a.unsqueeze(0)
            log_b = log_b.unsqueeze(0)
        result = _simplex_minimum_from_logs(log_a, log_b).log()
        return result.squeeze(0) if squeeze else result

    def _regularization(self, log_beta, log_r):
        """Same penalty is added to the objective and its touching bound."""
        return log_beta.new_zeros(())

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

        # Return to the original simplex parameterization while preserving the
        # represented potential through the compensating beta rescaling.
        log_r = self.log_cp_cores.detach().clone()
        normalizers = torch.logsumexp(log_r, dim=1, keepdim=True)
        log_r -= normalizers
        log_beta = self.log_alpha.detach().clone()
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
            log_beta = self._update_block(log_a, log_n, log_beta)
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
                    log_b_linear.T, log_m[d].T, log_r[d].T
                ).T
                log_u[d] = _logsumexp_matmul(log_q[d], log_r[d])

        bound_after, new_loss = surrogate().item(), objective().item()
        previous_loss = initial_loss.item()
        scale = max(1.0, abs(previous_loss), abs(bound_before))
        allowance = 64 * torch.finfo(self.dtype).eps * scale
        if (
            not math.isfinite(bound_after) or not math.isfinite(new_loss)
            or bound_after > bound_before + allowance
            or new_loss > previous_loss + allowance
        ):
            raise FloatingPointError(
                f"MM step failed monotonicity: "
                f"G {bound_before:.16g} -> {bound_after:.16g}, "
                f"loss {previous_loss:.16g} -> {new_loss:.16g}; "
                "no candidate weights committed"
            )

        self.log_alpha.copy_(log_beta)
        self.log_cp_cores.copy_(log_r)
        return {"loss": new_loss, "loss_before": previous_loss,
                "surrogate_decrease": bound_before - bound_after}
