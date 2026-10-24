"""Simplex-constrained block-MM with a compiled log-bisection solver."""

import torch

from .dlight_sb_mm import DLightSBMM, _project_potentials_to_simplex


def _simplex_minimum_core(
    a: torch.Tensor,
    b: torch.Tensor,
    bisection_steps: int,
) -> torch.Tensor:
    """Run fixed-step bisection for row-wise simplex block minimizers."""
    positive = b > 0
    has_counts = positive.any(dim=1, keepdim=True)
    minimum = a.masked_fill(~positive, torch.inf).amin(dim=1, keepdim=True)
    minimum = torch.where(has_counts, minimum, a.amin(dim=1, keepdim=True))
    shifted = a - minimum

    lower = b.masked_fill(shifted != 0, 0).sum(dim=1, keepdim=True)
    upper = b.sum(dim=1, keepdim=True)
    lo = torch.where(has_counts, lower, torch.ones_like(lower)).log()
    hi = torch.where(has_counts, upper, torch.ones_like(upper)).log()
    for _ in range(bisection_steps):
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


def _simplex_minimum(
    a: torch.Tensor,
    b: torch.Tensor,
    bisection_steps: int = 32,
) -> torch.Tensor:
    """Minimize ``a*z - b*log(z)`` row-wise on the closed simplex."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must have the same two-dimensional shape")
    if not isinstance(bisection_steps, int) or isinstance(bisection_steps, bool):
        raise ValueError("bisection_steps must be a positive integer")
    if bisection_steps < 1:
        raise ValueError("bisection_steps must be a positive integer")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (b < 0).any():
        raise ValueError("Expected finite coefficients and nonnegative counts")
    return _simplex_minimum_core(a, b, bisection_steps)


def _simplex_minimum_from_logs_core(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    bisection_steps: int,
) -> torch.Tensor:
    shift = torch.maximum(
        log_a.amax(dim=1, keepdim=True),
        log_b.amax(dim=1, keepdim=True),
    )
    shift = torch.where(torch.isfinite(shift), shift, 0.0)
    a = torch.exp(log_a - shift)
    b = torch.exp(log_b - shift)
    smallest = torch.nextafter(b.new_tensor(0.0), b.new_tensor(1.0))
    b = torch.where(torch.isfinite(log_b) & (b == 0), smallest, b)
    return _simplex_minimum_core(a, b, bisection_steps)


_compiled_simplex_minimum_from_logs_core = torch.compile(
    _simplex_minimum_from_logs_core,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


@torch.no_grad()
def _simplex_minimum_from_logs(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    bisection_steps: int = 32,
    compile_kernel: bool = True,
) -> torch.Tensor:
    """Solve a simplex block from log coefficients without CUDA sync checks."""
    if log_a.shape != log_b.shape or log_a.ndim != 2:
        raise ValueError("log_a and log_b must have the same two-dimensional shape")
    if not isinstance(bisection_steps, int) or isinstance(bisection_steps, bool):
        raise ValueError("bisection_steps must be a positive integer")
    if bisection_steps < 1:
        raise ValueError("bisection_steps must be a positive integer")
    if not isinstance(compile_kernel, bool):
        raise ValueError("compile_kernel must be a boolean")
    if not log_a.is_cuda:
        for value in (log_a, log_b):
            if torch.isnan(value).any() or torch.isposinf(value).any():
                raise ValueError("Log coefficients may contain finite values and -inf only")

    kernel = (
        _compiled_simplex_minimum_from_logs_core
        if compile_kernel and log_a.is_cuda
        else _simplex_minimum_from_logs_core
    )
    result = kernel(log_a, log_b, bisection_steps)
    return result.clone() if compile_kernel and log_a.is_cuda else result


class DLightSBMMSimplex(DLightSBMM):
    """MM whose alpha and categorical core blocks are optimized on simplexes."""

    def __init__(
        self,
        *args,
        bisection_steps: int = 32,
        compile_simplex_solver: bool = True,
        **kwargs,
    ):
        if not isinstance(bisection_steps, int) or isinstance(bisection_steps, bool):
            raise ValueError("bisection_steps must be a positive integer")
        if bisection_steps < 1:
            raise ValueError("bisection_steps must be a positive integer")
        if not isinstance(compile_simplex_solver, bool):
            raise ValueError("compile_simplex_solver must be a boolean")
        if kwargs.get("regularization_lambda_alpha", 0) != 0 or kwargs.get(
            "regularization_lambda_cores", 0
        ) != 0:
            raise ValueError("Simplex MM currently requires zero regularization")
        if kwargs.get("project_to_simplex_after_step", False):
            raise ValueError("Simplex MM does not need post-step projection")
        if kwargs.get("freeze_beta", False):
            raise ValueError("Simplex MM requires beta updates")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("bisection_steps", "compile_simplex_solver")

    def _update_block(self, log_a, log_b, current_log_z, *, block):
        del current_log_z
        if block not in {"alpha", "cores"}:
            raise ValueError(f"Unknown MM parameter block: {block}")
        squeeze = log_a.ndim == 1
        if squeeze:
            log_a = log_a.unsqueeze(0)
            log_b = log_b.unsqueeze(0)
        result = _simplex_minimum_from_logs(
            log_a,
            log_b,
            self.hparams.bisection_steps,
            self.hparams.compile_simplex_solver,
        ).log()
        return result.squeeze(0) if squeeze else result

    @torch.no_grad()
    def mm_step(self, x0, x1, *, inner_sweeps=None):
        # A loaded baseline or historical checkpoint may use an equivalent
        # unnormalized representation. Enter the feasible parameterization
        # before constructing the constrained surrogate.
        projected_beta, projected_cores = _project_potentials_to_simplex(
            self.log_alpha, self.log_cp_cores
        )
        self.log_alpha.copy_(projected_beta)
        self.log_cp_cores.copy_(projected_cores)
        info = super().mm_step(x0, x1, inner_sweeps=inner_sweeps)
        beta_residual, core_max, core_mean = torch.stack(
            (
                self.log_alpha.exp().sum().sub(1).abs(),
                self.log_cp_cores.exp().sum(dim=1).sub(1).abs().amax(),
                self.log_cp_cores.exp().sum(dim=1).sub(1).abs().mean(),
            )
        ).detach().cpu().tolist()
        return {
            **info,
            "simplex_beta_residual": beta_residual,
            "simplex_core_residual_max": core_max,
            "simplex_core_residual_mean": core_mean,
        }


__all__ = [
    "DLightSBMMSimplex",
    "_simplex_minimum",
    "_simplex_minimum_from_logs",
]
