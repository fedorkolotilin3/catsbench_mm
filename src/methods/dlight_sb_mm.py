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

from .dlight_sb import DLightSB


def _simplex_minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimize a*z - b*log(z) on each row's simplex, including b=0.

    Inputs are finite float64 tensors [number_of_blocks, block_size].
    The Lagrange multiplier is found by bisection; zero-count coordinates
    receive residual mass only when their linear coefficient is minimal.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must have the same two-dimensional shape")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (b < 0).any():
        raise ValueError("Expected finite coefficients and nonnegative counts")

    positive = b > 0
    has_counts = positive.any(dim=1, keepdim=True)
    minimum = a.masked_fill(~positive, torch.inf).amin(dim=1, keepdim=True)
    minimum = torch.where(has_counts, minimum, a.amin(dim=1, keepdim=True))
    shifted = a - minimum
    # Solve for delta = lambda + min(a on positive-count coordinates).
    # The root can be arbitrarily close to zero when counts are tiny.
    # At the root: sum(b where shifted=0) <= delta <= sum(b).
    # Bisect in log(delta), so 100 iterations resolve relative accuracy even
    # across the full float64 exponent range. Linear bisection would fail here.
    lower = b.masked_fill(shifted != 0, 0).sum(dim=1, keepdim=True)
    upper = b.sum(dim=1, keepdim=True)
    lo = torch.where(has_counts, lower, torch.ones_like(lower)).log()
    hi = torch.where(has_counts, upper, torch.ones_like(upper)).log()
    for _ in range(100):
        log_mid = (lo + hi) / 2
        mid = log_mid.exp()
        denominator = torch.where(positive, shifted + mid, torch.ones_like(a))
        mass = (b / denominator).sum(dim=1, keepdim=True)
        above = mass > 1
        lo = torch.where(above, log_mid, lo)
        hi = torch.where(above, hi, log_mid)

    delta = ((lo + hi) / 2).exp()
    minimum_zero = a.masked_fill(positive, torch.inf).amin(dim=1, keepdim=True)
    boundary_delta = minimum - minimum_zero
    boundary = has_counts & (boundary_delta > delta)
    delta = torch.where(boundary, boundary_delta, delta)
    denominator = torch.where(positive, shifted + delta, torch.ones_like(a))
    z = b / denominator

    # Interior root: correct only bisection roundoff, not an arbitrary update.
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
    # With no logarithmic terms the problem is linear.
    minima = a == a.amin(dim=1, keepdim=True)
    linear_solution = minima.to(a.dtype) / minima.sum(dim=1, keepdim=True)
    z = torch.where(has_counts, z, linear_solution)
    # A positive count must not get log(0) just because division underflowed.
    # Use the smallest representable positive float, not pseudo-count smoothing.
    smallest = torch.nextafter(a.new_tensor(0.0), a.new_tensor(1.0))
    return torch.where(positive & (z == 0), smallest, z)


def _stable_log_matmul(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(log_a) @ exp(log_b)) without unscaled exponentiation.

    This is the two-dimensional MM counterpart of catsbench.lse_matmul. Row
    and column maxima make the common path a memory-efficient BLAS product.
    If that scaled product still underflows, recompute only the affected output
    rows with an exact logsumexp reduction.
    """
    if log_a.ndim != 2 or log_b.ndim != 2 or log_a.shape[1] != log_b.shape[0]:
        raise ValueError("Expected compatible two-dimensional log matrices")
    for value in (log_a, log_b):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log matrices may contain finite values and -inf only")

    a_max = log_a.amax(dim=1, keepdim=True)
    b_max = log_b.amax(dim=0, keepdim=True)
    safe_a_max = torch.where(torch.isfinite(a_max), a_max, 0.0)
    safe_b_max = torch.where(torch.isfinite(b_max), b_max, 0.0)
    product = torch.exp(log_a - safe_a_max) @ torch.exp(log_b - safe_b_max)
    result = product.log() + safe_a_max + safe_b_max

    # A zero scaled dot product is possible when the two maxima have almost
    # disjoint support. Keep memory bounded by reducing one output row at a time.
    bad_rows = torch.nonzero((product == 0).any(dim=1), as_tuple=False).flatten()
    for row in bad_rows.tolist():
        exact = torch.logsumexp(log_a[row, :, None] + log_b, dim=0)
        result[row] = torch.where(product[row] == 0, exact, result[row])
    return result


def _simplex_minimum_from_logs(
    log_a: torch.Tensor, log_b: torch.Tensor
) -> torch.Tensor:
    """Solve the simplex subproblem from log coefficients without underflow.

    Multiplying both ``a`` and ``b`` in one row by the same positive constant
    does not change its minimizer. Row-wise scaling therefore preserves the MM
    update while keeping the inputs to ``_simplex_minimum`` representable.
    """
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

    # Preserve every mathematically positive count even when its scaled value
    # lies below the normal floating-point range.
    smallest = torch.nextafter(b.new_tensor(0.0), b.new_tensor(1.0))
    b = torch.where(torch.isfinite(log_b) & (b == 0), smallest, b)
    return _simplex_minimum(a, b)


class _MMOptimizer(torch.optim.Optimizer):
    """Execute an analytic update through Lightning's optimizer bookkeeping.

    No gradient or dummy SGD step: the closure itself updates the parameters.
    The Lightning wrapper advances global_step and supports max_steps/checkpoints.
    """

    def __init__(self, params):
        super().__init__(params, defaults={})

    def step(self, closure=None):
        if closure is None:
            raise ValueError("An MM update closure is required")
        return closure()


class DLightSBMM(DLightSB):
    """DLightSB with one analytic block-MM sweep per Lightning training_step.

    The bound uses the tangent to log(c) and Jensen for -log(v).
    beta is updated first, then dimensions sequentially, with fixed E-step
    statistics. This is the simple bound in theory.md, not the newer mm.md.
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
        self.double()

    def configure_optimizers(self):
        return _MMOptimizer([self.log_alpha, self.log_cp_cores])

    def on_train_start(self):
        if self.dtype != torch.float64:
            raise ValueError("Use trainer precision='64-true' for MM")
        if self.hparams.tol is not None and self.trainer.num_training_batches != 1:
            raise ValueError("MM tolerance requires one full-dataset batch per epoch")

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

    @classmethod
    def from_model(cls, model: DLightSB, *, state_dict=None, tol=None):
        """Copy a DLightSB; optionally restore its pre-training weights.

        The prior is copied too, so float64 conversion never changes model.
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
                prior=deepcopy(model.prior).double(),
                optimizer=None, scheduler=None, tol=tol,
                **{key: model.hparams[key] for key in keys},
            ).double()
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
        self.double()
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
        if self.dtype != torch.float64 or self.prior.dtype != torch.float64:
            raise ValueError("MM requires float64 parameters and prior")
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

        # Normalize cores AND compensate component weights, preserving q_theta.
        log_r = self.log_cp_cores.detach().clone()
        normalizers = torch.logsumexp(log_r, dim=1, keepdim=True)
        log_r -= normalizers
        log_beta = self.log_alpha + normalizers.squeeze(1).sum(dim=0)
        log_beta = log_beta - torch.logsumexp(log_beta, dim=0)
        invalid_log_r = torch.isnan(log_r).any() or torch.isposinf(log_r).any()
        empty_core = ~torch.isfinite(log_r).any(dim=1).all()
        if invalid_log_r or empty_core or not torch.isfinite(log_beta).all():
            raise ValueError("Invalid initial potential parameters")
        log_u = torch.stack([
            _stable_log_matmul(log_q[d], log_r[d]) for d in range(dim)
        ])  # [D, N0, K]

        def log_normalizer():
            return torch.logsumexp(log_u.sum(dim=0) + log_beta, dim=1)

        def target_logits():
            logits = log_beta.expand(len(target), components).clone()
            for d in range(dim):
                logits += log_r[d, target[:, d]]
            return logits

        def objective():
            return weight0 @ log_normalizer() - weight1 @ torch.logsumexp(target_logits(), dim=1)

        initial_loss = objective()
        if not torch.isfinite(initial_loss):
            raise ValueError("Initial potential must be positive on all target observations")
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
            return weight0 @ ratio - beta_term - core_term

        bound_before = surrogate().item()
        for _ in range(inner_sweeps):
            log_product = log_u.sum(dim=0)
            log_a = torch.logsumexp(
                log_weight0[:, None] + log_product - old_log_c[:, None], dim=0
            )
            beta = _simplex_minimum_from_logs(log_a[None], log_n[None])[0]
            log_beta = beta.log()
            for d in range(dim):
                log_other = log_u.sum(dim=0) - log_u[d]
                log_coefficients = (
                    log_weight0[:, None]
                    + log_beta[None]
                    + log_other
                    - old_log_c[:, None]
                )
                log_b_linear = _stable_log_matmul(
                    log_q[d].T, log_coefficients
                )  # [S,K]
                r_d = _simplex_minimum_from_logs(
                    log_b_linear.T, log_m[d].T
                ).T
                log_r[d] = r_d.log()
                log_u[d] = _stable_log_matmul(log_q[d], log_r[d])

        bound_after, new_loss = surrogate().item(), objective().item()
        previous_loss = initial_loss.item()
        allowance = 1e-10 * max(1.0, abs(previous_loss), abs(bound_before))
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
