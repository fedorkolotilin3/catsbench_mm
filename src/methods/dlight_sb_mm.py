"""CPU block-MM for the unregularized DLightSB loss (see theory.md).

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

import torch

from .dlight_sb import DLightSB


def _simplex_minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimize a*z - b*log(z) on each row's simplex, including b=0.

    Inputs are finite CPU float64 tensors [number_of_blocks, block_size].
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
        if self.trainer.world_size != 1 or self.device.type != "cpu":
            raise ValueError("MM currently supports a single CPU process")
        if self.dtype != torch.float64:
            raise ValueError("Use trainer precision='64-true' for MM")
        if self.hparams.tol is not None and self.trainer.num_training_batches != 1:
            raise ValueError("MM tolerance requires one full-dataset batch per epoch")

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        info = self.optimizers().step(closure=lambda: self.mm_step(x0, x1))
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
        """Copy a CPU DLightSB; optionally restore its pre-training weights.

        The prior is copied too, so float64 conversion never changes model.
        """
        if model.device.type != "cpu":
            raise ValueError("DLightSBMM requires a CPU model")
        if model.hparams.entropy_lambda != 0:
            raise ValueError("MM supports entropy_lambda=0 only")
        keys = (
            "dim", "num_categories", "num_potentials", "num_timesteps",
            "distr_init", "entropy_warmup_steps",
            "entropy_lambda", "sample_prob", "tau",
        )
        with torch.device("cpu"):
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

        Duplicates are compressed with exact empirical weights. A new bound is
        built once, then minimized blockwise. Commit only a checked candidate.
        No persistent data cache: another batch/checkpoint cannot reuse stale Q.
        """
        if self.device.type != "cpu" or self.prior.log_p_cum.device.type != "cpu":
            raise ValueError("MM runs on CPU only")
        if self.dtype != torch.float64 or self.prior.dtype != torch.float64:
            raise ValueError("MM requires float64 parameters and prior")
        if self.hparams.entropy_lambda != 0:
            raise ValueError("MM supports entropy_lambda=0 only")
        inner_sweeps = self.hparams.inner_sweeps if inner_sweeps is None else inner_sweeps
        if not isinstance(inner_sweeps, int) or inner_sweeps < 1:
            raise ValueError("inner_sweeps must be a positive integer")
        for x in (x0, x1):
            if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
                raise ValueError("Training data must be CPU tensors")
            if x.ndim < 2 or x.flatten(1).shape[1] != self.hparams.dim or len(x) == 0:
                raise ValueError("Training data must flatten to nonempty [N, dim] tensors")
            if x.dtype != torch.long or x.min() < 0 or x.max() >= self.hparams.num_categories:
                raise ValueError("Use torch.long categories in [0, num_categories)")
        if not self._did_weight_init and not self._loaded_from_ckpt:
            raise ValueError("Call init_weights first, or use DLightSBMM.from_model")

        x0, x1 = x0.flatten(1), x1.flatten(1)
        source, counts0 = torch.unique(x0, dim=0, return_counts=True)
        target, counts1 = torch.unique(x1, dim=0, return_counts=True)
        weight0 = counts0.double() / len(x0)
        weight1 = counts1.double() / len(x1)
        dim, categories, components = self.log_cp_cores.shape
        q = self.prior.extract_last_cum_matrix(source).exp().permute(1, 0, 2).contiguous()
        if not torch.isfinite(q).all() or (q <= 0).any():
            raise ValueError("MM requires a strictly positive finite reference transition")

        # Normalize cores AND compensate component weights, preserving q_theta.
        log_r = self.log_cp_cores.detach().clone()
        normalizers = torch.logsumexp(log_r, dim=1, keepdim=True)
        log_r -= normalizers
        log_beta = self.log_alpha + normalizers.squeeze(1).sum(dim=0)
        log_beta = log_beta - torch.logsumexp(log_beta, dim=0)
        r, beta = log_r.exp(), log_beta.exp()
        if not torch.isfinite(r).all() or not torch.isfinite(beta).all():
            raise ValueError("Invalid initial potential parameters")
        u = torch.stack([q[d] @ r[d] for d in range(dim)])  # [D, unique_N0, K]

        def log_normalizer():
            return torch.logsumexp(u.log().sum(dim=0) + beta.log(), dim=1)

        def target_logits():
            logits = beta.log().expand(len(target), components).clone()
            for d in range(dim):
                logits += r[d, target[:, d]].log()
            return logits

        def objective():
            return weight0 @ log_normalizer() - weight1 @ torch.logsumexp(target_logits(), dim=1)

        initial_loss = objective()
        if not torch.isfinite(initial_loss):
            raise ValueError("Initial potential must be positive on all target observations")
        old_log_c = log_normalizer().clone()
        gamma = target_logits().softmax(dim=1)
        weighted_gamma = weight1[:, None] * gamma
        n = weighted_gamma.sum(dim=0)
        m = torch.zeros_like(r)
        for d in range(dim):
            m[d].index_add_(0, target[:, d], weighted_gamma)

        def surrogate():
            # Constant C_t is unnecessary when comparing this same bound.
            ratio = (log_normalizer() - old_log_c).exp()
            return weight0 @ ratio - torch.xlogy(n, beta).sum() - torch.xlogy(m, r).sum()

        bound_before = surrogate().item()
        for _ in range(inner_sweeps):
            log_product = u.log().sum(dim=0)
            a = (weight0[:, None] * (log_product - old_log_c[:, None]).exp()).sum(dim=0)
            beta = _simplex_minimum(a[None], n[None])[0]
            for d in range(dim):
                log_other = u.log().sum(dim=0) - u[d].log()
                coefficients = (
                    weight0[:, None]
                    * (beta.log()[None] + log_other - old_log_c[:, None]).exp()
                )
                b_linear = q[d].T @ coefficients  # [S,K]
                r[d] = _simplex_minimum(b_linear.T, m[d].T).T
                u[d] = q[d] @ r[d]

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

        self.log_alpha.copy_(beta.log())
        self.log_cp_cores.copy_(r.log())
        return {"loss": new_loss, "loss_before": previous_loss,
                "surrogate_decrease": bound_before - bound_after}
