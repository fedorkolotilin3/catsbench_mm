"""Run with: python -m unittest discover -s tests -p test_shared_mm_lse.py"""
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import torch
from catsbench.benchmarks import base as benchmark
from catsbench.lse_matmul import lse_matmul
from src.methods import dlight_sb as baseline
from src.methods import dlight_sb_mm as mm
from src.methods.mm_regularized import DLightSBMMRegularized
import test_dlight_sb_mm as existing


class SharedLSETest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            from catsbench.lse_matmul.triton_forward import _lse_fwd_kernel
            from catsbench.lse_matmul.triton_backward import _lse_bwd_dA_kernel, _lse_bwd_dB_kernel
            # Test the production kernels with one existing tile configuration.
            # Avoid benchmarking/compiling every large tile in correctness tests.
            for kernel in (_lse_fwd_kernel, _lse_bwd_dA_kernel, _lse_bwd_dB_kernel):
                config = min(kernel.configs, key=lambda c: c.kwargs['BLOCK_M'] * c.kwargs['BLOCK_N'])
                context = patch.object(kernel, 'configs', [config])
                context.start()
                self.addCleanup(context.stop)

    def test_shared_entry_point(self):
        self.assertIs(baseline.lse_matmul, lse_matmul)
        self.assertIs(benchmark.lse_matmul, lse_matmul)
        self.assertIs(mm.lse_matmul, lse_matmul)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_forward_backward_and_mm_no_grad(self):
        torch.manual_seed(42)
        with patch.dict(os.environ, {'LSE_BACKEND': 'triton'}):
            # Broadcast batch dimensions and a non-contiguous right operand.
            a = torch.randn(2, 3, 50, device='cuda', requires_grad=True)
            b = torch.randn(1, 16, 50, device='cuda').transpose(-1, -2).requires_grad_()
            ar = a.detach().double().requires_grad_()
            br = b.detach().double().requires_grad_()
            expected = torch.logsumexp(ar.unsqueeze(-1) + br.unsqueeze(-3), dim=-2)
            actual = lse_matmul(a, b)
            torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-6)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(a.grad.double(), ar.grad, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(b.grad.double(), br.grad, atol=2e-5, rtol=2e-5)
            result = mm._logsumexp_matmul(a, b)
            self.assertFalse(result.requires_grad)
            torch.testing.assert_close(result, actual.detach())
            # Former scaled-GEMM failure and exact zero support.
            a = torch.tensor([[0., -100.], [-torch.inf, -torch.inf]], device='cuda')
            b = torch.tensor([[-100., -torch.inf], [0., -torch.inf]], device='cuda')
            expected = torch.logsumexp(a.double().unsqueeze(-1) + b.double().unsqueeze(-3), dim=-2)
            torch.testing.assert_close(mm._logsumexp_matmul(a, b).double(), expected, atol=2e-5, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_regularized_mm_and_baseline(self):
        with patch.dict(os.environ, {'LSE_BACKEND': 'triton'}):
            model = existing.DLightSBMMUpdateTest()._make_model().float().cuda()
            model = DLightSBMMRegularized.from_model(model)
            x = torch.tensor([[0, 0], [1, 1], [2, 2]], device='cuda')
            y = torch.tensor([[0, 0], [1, 1], [0, 1]], device='cuda')
            for _ in range(3):
                info = model.mm_step(x, y)
                self.assertLessEqual(info['loss'], info['loss_before'] + 2e-5)
            loss, _ = model.loss(x, y)
            self.assertAlmostEqual(loss.item(), info['loss'], delta=2e-5)
            plain = baseline.DLightSB(prior=deepcopy(model.prior), dim=2,
                num_categories=3, num_potentials=2, num_timesteps=1,
                optimizer=None, entropy_lambda=0.).float().cuda()
            plain.load_state_dict(model.state_dict())
            plain.loss(x, y)[0].backward()
            self.assertTrue(torch.isfinite(plain.log_cp_cores.grad).all())
            self.assertTrue(torch.isfinite(plain.log_alpha.grad).all())


if __name__ == '__main__':
    unittest.main()
