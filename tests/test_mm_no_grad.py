from copy import deepcopy
import unittest

import torch

import test_dlight_sb_mm as existing
from src.methods.dlight_sb import DLightSB
from src.methods.dlight_sb_mm import (
    DLightSBMM, _MMOptimizer, _logsumexp_matmul,
    _simplex_minimum_from_logs,
)
from src.methods.mm_regularized import (
    DLightSBMMRegularized, _exponential_log_update, _exponential_log_penalty,
)


class MMNoGradTest(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.base = existing.DLightSBMMUpdateTest()._make_model()
        self.x = torch.tensor([[0, 0], [1, 1], [2, 2]])
        self.y = torch.tensor([[0, 0], [1, 1], [0, 1]])

    def assert_graph_free(self, value):
        if isinstance(value, torch.Tensor):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        elif isinstance(value, dict):
            for item in value.values():
                self.assert_graph_free(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.assert_graph_free(item)

    def test_direct_evaluation_and_updates_do_not_save_autograd_tensors(self):
        models = [self.base, DLightSBMMRegularized.from_model(self.base)]
        for model in models:
            with self.subTest(model=type(model).__name__), torch.enable_grad():
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                before = model.log_cp_cores.clone()
                info = model.mm_step(self.x, self.y)
                self.assertLessEqual(info['loss'], info['loss_before'] + 1e-8)
                self.assertFalse(torch.equal(before, model.log_cp_cores))
                saved = []
                def pack(tensor):
                    saved.append(tensor)
                    return tensor
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    results = [
                        model.loss(self.x, self.y), model.get_log_c(self.x),
                        model.get_log_v(self.y),
                        model.get_transition_logits(self.x, torch.zeros(3, dtype=torch.long)),
                        model.sample(self.x),
                        model.sample_trajectory(self.x, return_transitions=True),
                    ]
                self.assert_graph_free(results)
                self.assertEqual(saved, [])
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertTrue(torch.is_grad_enabled())

    def test_helpers_ignore_requires_grad_inputs_and_restore_grad_mode(self):
        with torch.enable_grad():
            a = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
            b = torch.ones_like(a, requires_grad=True)
            results = [
                _logsumexp_matmul(a, b),
                _simplex_minimum_from_logs(a, b),
                _exponential_log_update(a, b, a, .01),
                _exponential_log_penalty(b),
                DLightSBMMRegularized.from_model(self.base)._regularization(a, b),
            ]
            self.assert_graph_free(results)
            self.assertTrue(torch.is_grad_enabled())

    def test_optimizer_closure_has_no_grad_and_baseline_still_backpropagates(self):
        baseline = DLightSB(
            prior=deepcopy(self.base.prior), dim=2, num_categories=3,
            num_potentials=2, num_timesteps=1, optimizer=None, entropy_lambda=0.,
        ).double()
        baseline.load_state_dict(self.base.state_dict())
        for cls in (DLightSBMM, DLightSBMMRegularized):
            solver = cls.from_model(baseline)
            self.assertTrue(all(not p.requires_grad for p in solver.parameters()))
        with torch.enable_grad():
            optimizer = _MMOptimizer(baseline.parameters())
            result = optimizer.step(lambda: baseline.log_alpha.square().sum())
            self.assert_graph_free(result)
            loss, _ = baseline.loss(self.x, self.y)
            self.assertTrue(loss.requires_grad)
            loss.backward()
            self.assertIsNotNone(baseline.log_alpha.grad)
            self.assertIsNotNone(baseline.log_cp_cores.grad)


if __name__ == '__main__':
    unittest.main()
