"""Precision regression tests for the actual CUDA kernels, with small test tiles."""

from contextlib import ExitStack
import importlib.util
import itertools
import math
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
    "CUDA and Triton are required",
)
class TritonPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import triton
        from catsbench.lse_matmul.triton import triton_lse_matmul
        from catsbench.lse_matmul.triton_forward import _lse_fwd_kernel
        from catsbench.lse_matmul.triton_backward import (
            _lse_bwd_dA_kernel, _lse_bwd_dB_kernel,
        )

        cls.lse = staticmethod(triton_lse_matmul)
        cls.patches = ExitStack()
        cls.addClassCleanup(cls.patches.close)
        # Exercise the same kernels without a lengthy performance autotuning sweep.
        # Production configurations are restored when the class finishes.
        for kernel in (_lse_fwd_kernel, _lse_bwd_dA_kernel, _lse_bwd_dB_kernel):
            cls.patches.enter_context(patch.object(kernel, "configs", [
                triton.Config(
                    {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8},
                    num_warps=4, num_stages=2,
                )
            ]))
            cls.patches.enter_context(patch.object(kernel, "cache", {}))

    def test_forward_and_backward_against_cpu_float64(self):
        dtypes = (torch.float32, torch.float64)
        for da, db, exp2, eps, batched in itertools.product(
            dtypes, dtypes, (True, False), (0.0, 1e-8), (False, True)
        ):
            with self.subTest(da=da, db=db, exp2=exp2, eps=eps, batched=batched):
                torch.manual_seed(164)
                ashape, bshape = ((2, 3, 9), (1, 9, 4)) if batched else ((3, 9), (9, 4))
                a = (torch.randn(ashape, dtype=da, device="cuda") * 12).requires_grad_()
                b = (torch.randn(bshape, dtype=db, device="cuda") * 12).requires_grad_()
                ar = a.detach().cpu().double().requires_grad_()
                br = b.detach().cpu().double().requires_grad_()
                terms = ar.unsqueeze(-1) + br.unsqueeze(-3)
                if eps:
                    extra = terms.new_full((*terms.shape[:-2], 1, terms.shape[-1]), math.log(eps))
                    terms = torch.cat((terms, extra), dim=-2)
                reference = terms.logsumexp(dim=-2)
                result = self.lse(a, b, use_exp2=exp2, eps=eps)
                dtype = torch.promote_types(da, db)
                self.assertEqual(result.dtype, dtype)
                tol = 1e-10 if dtype == torch.float64 else 2e-5
                torch.testing.assert_close(result.cpu().double(), reference, atol=tol, rtol=tol)
                # Nonuniform upstream gradients also exercise both reduction paths.
                grad = torch.linspace(-1, 1, result.numel(), device="cuda", dtype=dtype).reshape(result.shape)
                result.backward(grad)
                reference.backward(grad.cpu().double())
                for actual, expected, input_dtype in ((a.grad, ar.grad, da), (b.grad, br.grad, db)):
                    self.assertEqual(actual.dtype, input_dtype)
                    tol = 1e-10 if input_dtype == torch.float64 else 2e-5
                    torch.testing.assert_close(actual.cpu().double(), expected, atol=tol, rtol=tol)

    def test_zero_support_with_epsilon_and_transposed_input(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = torch.full((9, 3), -torch.inf, device="cuda", dtype=dtype).T.requires_grad_()
                b = torch.zeros(9, 4, device="cuda", dtype=dtype, requires_grad=True)
                result = self.lse(a, b, eps=1e-8)
                expected = torch.full_like(result, math.log(1e-8))
                torch.testing.assert_close(result, expected)
                result.sum().backward()
                self.assertTrue(torch.equal(a.grad, torch.zeros_like(a)))
                self.assertTrue(torch.equal(b.grad, torch.zeros_like(b)))


if __name__ == "__main__":
    unittest.main()
