"""PyTorch backends for log-semiring matrix multiplication."""

import math

import torch
from torch.autograd import Function


def _promoted_operands(a, b):
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), torch.float32)
    return a.to(dtype=dtype), b.to(dtype=dtype)


def _validate_eps(eps: float) -> None:
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")


def cpu_lse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Reference CPU implementation using an explicitly broadcast reduction."""
    if a.is_cuda or b.is_cuda:
        raise ValueError("The 'cpu' LSE backend only accepts CPU tensors.")
    _validate_eps(eps)
    terms = a.unsqueeze(-1) + b.unsqueeze(-3)
    if eps == 0:
        return torch.logsumexp(terms, dim=-2)

    # Include epsilon as an additional finite log-domain term. Unlike applying
    # logaddexp after the reduction, this keeps the all--inf backward finite.
    log_eps = terms.new_full((*terms.shape[:-2], 1, terms.shape[-1]), math.log(eps))
    return torch.logsumexp(torch.cat((terms, log_eps), dim=-2), dim=-2)


def normalized_lse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Fast path for normalized log-probabilities.

    Both operands must be log-probabilities, so their exponentials are bounded
    by one. No normalization checks are performed in this hot path.
    """
    _validate_eps(eps)
    a, b = _promoted_operands(a, b)
    product = torch.matmul(a.exp(), b.exp())
    return torch.log(product + eps)


def _scaled_operands(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = _promoted_operands(a, b)
    a_max = a.amax(dim=-1, keepdim=True)
    b_max = b.amax(dim=-2, keepdim=True)

    # An all--inf row/column has no finite shift. A zero shift preserves
    # exp(-inf) == 0 and therefore the correct -inf output.
    safe_a_max = torch.where(torch.isfinite(a_max), a_max, 0.0)
    safe_b_max = torch.where(torch.isfinite(b_max), b_max, 0.0)
    return (
        torch.exp(a - safe_a_max),
        torch.exp(b - safe_b_max),
        safe_a_max,
        safe_b_max,
    )


class _AnyLogitsLSEMatmul(Function):
    """Memory-aware autograd for stabilized, GEMM-backed LSE matmul."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, eps: float):
        _validate_eps(eps)
        a, b = _promoted_operands(a, b)
        scaled_a, scaled_b, a_max, b_max = _scaled_operands(a, b)
        product = torch.matmul(scaled_a, scaled_b)
        shift = a_max + b_max
        if eps == 0:
            output = torch.log(product) + shift
        else:
            output = torch.logaddexp(
                torch.log(product) + shift,
                shift.new_full((), math.log(eps)),
            )

        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        ctx.eps = eps
        ctx.save_for_backward(a, b)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        scaled_a, scaled_b, _, _ = _scaled_operands(a, b)
        product = torch.matmul(scaled_a, scaled_b)
        if ctx.eps == 0:
            denominator = product
        else:
            _, _, a_max, b_max = _scaled_operands(a, b)
            denominator = product + ctx.eps * torch.exp(-(a_max + b_max))
        normalized_grad = torch.where(
            denominator > 0, grad_out.to(dtype=denominator.dtype) / denominator, 0.0
        )

        d_a = None
        if ctx.needs_input_grad[0]:
            d_a = scaled_a * torch.matmul(
                normalized_grad, scaled_b.transpose(-2, -1)
            )
            d_a = d_a.sum_to_size(ctx.a_shape)

        d_b = None
        if ctx.needs_input_grad[1]:
            d_b = scaled_b * torch.matmul(
                scaled_a.transpose(-2, -1), normalized_grad
            )
            d_b = d_b.sum_to_size(ctx.b_shape)
        return d_a, d_b, None


def any_logits_lse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Stable implementation for arbitrary normalized or raw logits.

    CUDA uses shifted GEMM for throughput. CPU uses the exact reference
    reduction, which avoids probability-domain underflow without sacrificing a
    GPU fast path.
    """
    if not a.is_cuda and not b.is_cuda:
        return cpu_lse_matmul(a, b, eps=eps)
    return _AnyLogitsLSEMatmul.apply(a, b, eps)
