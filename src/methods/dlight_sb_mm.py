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


def _support_preserving_update_from_logs(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    current_log_z: torch.Tensor,
) -> torch.Tensor:
    """Apply a support-preserving MM update to ``a*z - b*log(z)``.

    The explicit solution is ``z = b / a`` when both coefficients are
    positive. Inputs and the result stay in the log domain. If ``b = 0``, the
    current value is preserved instead of taking the boundary minimizer zero.
    This leaves that coordinate's surrogate contribution unchanged and keeps
    support for categories that may occur in later stochastic batches. If all
    ``b`` coefficients in a block are zero, the complete block is therefore
    left unchanged. The remaining case, ``a = 0 < b``, has no finite
    minimizer.
    """
    if log_a.shape != log_b.shape or log_a.shape != current_log_z.shape:
        raise ValueError("log_a, log_b and current_log_z must have the same shape")
    for value in (log_a, log_b, current_log_z):
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("Log parameters may contain finite values and -inf only")

    positive_a = torch.isfinite(log_a)
    positive_b = torch.isfinite(log_b)
    if ((~positive_a) & positive_b).any():
        raise FloatingPointError(
            "The nonnegative MM block is unbounded: a=0 and b>0"
        )

    log_z = current_log_z.clone()
    interior = positive_a & positive_b
    log_z = torch.where(interior, log_b - log_a, log_z)
    if torch.isnan(log_z).any() or torch.isposinf(log_z).any():
        raise FloatingPointError("Non-finite support-preserving MM block update")
    return log_z


@torch.compile(fullgraph=True)
def _logsumexp_matmul(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Compile the broadcast addition and library reduction together.

    Keep inputs in log space, including exact zeros represented by -inf.
    TorchInductor can fuse the expression without a full broadcast temporary.
    Requires a working Inductor toolchain (Triton on CUDA or a C++ compiler
    on CPU). The first call for a new specialization includes compilation.
    """
    return torch.logsumexp(log_a.unsqueeze(-1) + log_b.unsqueeze(-3), dim=-2)


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
    statistics. Parameters are constrained only to be nonnegative; each block
    uses the explicit minimizer ``z = b / a`` for positive sufficient
    statistics and preserves coordinates with zero stochastic-batch counts.
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

    def configure_optimizers(self):
        return _MMOptimizer([self.log_alpha, self.log_cp_cores])

    def on_train_start(self):
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("MM requires float32 or float64 parameters")
        if self.prior.dtype != self.dtype:
            raise ValueError("MM parameters and prior must use the same dtype")
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

    @torch.no_grad()
    def compilation_warmup_step(self, batch, optimizer) -> None:
        """Execute the complete analytic MM path without Lightning logging."""
        x0, x1 = batch
        optimizer.step(closure=lambda: self.mm_step(x0, x1))

    @classmethod
    def from_model(cls, model: DLightSB, *, state_dict=None, tol=None):
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
                optimizer=None, scheduler=None, tol=tol,
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
            return weight0 @ source_term - weight1 @ target_term

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
            return weight0 @ ratio - beta_term - core_term

        bound_before = surrogate().item()
        for _ in range(inner_sweeps):
            log_product = log_u.sum(dim=0)
            log_a = torch.logsumexp(
                log_weight0[:, None] + log_product - old_log_c[:, None], dim=0
            )
            log_beta = _support_preserving_update_from_logs(log_a, log_n, log_beta)
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
                log_r[d] = _support_preserving_update_from_logs(
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
