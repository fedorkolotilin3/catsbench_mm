import unittest

import torch

from src.data.prior import Prior
from src.methods.dlight_sb_mm import (
    DLightSBMM,
    _logsumexp_matmul,
    _simplex_newton_step_from_logs,
)


class CompiledLogsumexpTest(unittest.TestCase):
    def test_extreme_logits_and_exact_zeros(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype), torch.no_grad():
                # The old scaled GEMM clamped subnormals or lost terms here.
                a = torch.tensor([[0., -100.], [0., -104.],
                                  [-torch.inf, -torch.inf]], dtype=dtype)
                b = torch.tensor([[-100., -103., -torch.inf],
                                  [0., 0., -torch.inf]], dtype=dtype)
                actual = _logsumexp_matmul(a, b)
                expected = torch.logsumexp(
                    a.double().unsqueeze(-1) + b.double().unsqueeze(-3), dim=-2
                )
                torch.testing.assert_close(actual, expected.to(dtype))
                self.assertTrue(torch.isneginf(actual[-1]).all())
                self.assertTrue(torch.isneginf(actual[:, -1]).all())

    def test_transposed_inputs_and_changing_batch_size(self):
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for batch_size in (7, 11):
                a = 100 * torch.randn(3, batch_size, generator=generator,
                                      dtype=torch.float64)
                b = 100 * torch.randn(5, batch_size, generator=generator,
                                      dtype=torch.float64)
                actual = _logsumexp_matmul(a, b.T)
                expected = torch.stack([
                    torch.stack([torch.logsumexp(row + col, dim=0) for col in b])
                    for row in a
                ])
                torch.testing.assert_close(actual, expected)


class SimplexNewtonStepTest(unittest.TestCase):
    def _step(self, a, b, z, previous_lambda, initialized=True):
        return _simplex_newton_step_from_logs(
            a.log(),
            b.log(),
            z.log(),
            previous_lambda,
            torch.full_like(previous_lambda, initialized, dtype=torch.bool),
        )

    def test_fixed_point_matches_closed_form_kkt_solution(self):
        a = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float64)
        expected_z = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
        physical_lambda = 2.0
        b = expected_z * (a + physical_lambda)
        scale = torch.maximum(a.amax(dim=1, keepdim=True), b.amax(dim=1, keepdim=True))
        scaled_lambda = torch.full((1, 1), physical_lambda, dtype=a.dtype) / scale

        result = self._step(a, b, expected_z, scaled_lambda)

        torch.testing.assert_close(result["log_z"].exp(), expected_z)
        torch.testing.assert_close(result["newton_correction"], torch.zeros_like(scaled_lambda))

    def test_one_step_is_the_explicit_newton_formula(self):
        a = torch.tensor([[0.4, 0.8, 1.0]], dtype=torch.float64)
        b = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64)
        initial_z = torch.full_like(a, 1 / 3)
        lambda_old = torch.tensor([[0.2]], dtype=a.dtype)
        mass = (b / (a + lambda_old)).sum(dim=1, keepdim=True)
        curvature = (b / (a + lambda_old).square()).sum(dim=1, keepdim=True)
        expected_lambda = lambda_old + (mass - 1) / curvature

        result = self._step(a, b, initial_z, lambda_old)

        torch.testing.assert_close(result["lambda"], expected_lambda)
        torch.testing.assert_close(
            result["log_z"].exp(),
            b / (a + expected_lambda),
        )

    def test_common_coefficient_scale_does_not_change_update(self):
        a = torch.tensor([[0.4, 0.8, 1.0]], dtype=torch.float64)
        b = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64)
        initial_z = torch.full_like(a, 1 / 3)
        previous_lambda = torch.tensor([[0.2]], dtype=a.dtype)

        reference = self._step(a, b, initial_z, previous_lambda)
        scaled = _simplex_newton_step_from_logs(
            a.log() + 100,
            b.log() + 100,
            initial_z.log(),
            previous_lambda,
            torch.ones_like(previous_lambda, dtype=torch.bool),
        )

        torch.testing.assert_close(scaled["lambda"], reference["lambda"])
        torch.testing.assert_close(scaled["log_z"], reference["log_z"])

    def test_bootstrap_lambda_is_derived_from_current_z(self):
        a = torch.tensor([[0.4, 0.8, 1.0]], dtype=torch.float64)
        b = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64)
        initial_z = torch.full_like(a, 1 / 3)
        ignored_lambda = torch.tensor([[123.0]], dtype=a.dtype)
        bootstrap = b.sum(dim=1, keepdim=True) - (a * initial_z).sum(
            dim=1,
            keepdim=True,
        )
        mass = (b / (a + bootstrap)).sum(dim=1, keepdim=True)
        curvature = (b / (a + bootstrap).square()).sum(dim=1, keepdim=True)
        expected_lambda = bootstrap + (mass - 1) / curvature

        result = self._step(
            a,
            b,
            initial_z,
            ignored_lambda,
            initialized=False,
        )

        torch.testing.assert_close(result["lambda"], expected_lambda)


class DLightSBMMUpdateTest(unittest.TestCase):
    def _make_model(self) -> DLightSBMM:
        prior = Prior(
            alpha=0.1,
            num_categories=3,
            num_timesteps=1,
            num_skip_steps=1,
            eps=0.0,
            prior_type="uniform",
            dtype=torch.float64,
        )
        model = DLightSBMM(
            prior=prior,
            dim=2,
            num_categories=3,
            num_potentials=2,
            num_timesteps=1,
            optimizer=None,
            scheduler=None,
            distr_init="uniform",
            entropy_lambda=0.0,
            entropy_warmup_steps=1,
            sample_prob=0.9,
            tau=1.0,
        )
        model.init_weights()
        return model

    def test_mm_step_is_monotone_and_reports_simplex_residual(self):
        torch.manual_seed(0)
        model = self._make_model()
        with torch.no_grad():
            # The step must first map this equivalent unnormalized
            # representation back to the simplex parameterization.
            model.log_alpha.add_(10.0)

        x0 = torch.tensor(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
             [1, 2], [2, 0], [2, 1], [2, 2]],
            dtype=torch.long,
        )
        x1 = torch.tensor(
            [[2, 2], [2, 1], [2, 0], [1, 2], [1, 1],
             [1, 0], [0, 2], [0, 1], [0, 0]],
            dtype=torch.long,
        )

        for _ in range(5):
            info = model.mm_step(x0, x1)
            self.assertLessEqual(info["loss"], info["loss_before"] + 1e-10)
            self.assertGreaterEqual(info["surrogate_decrease"], -1e-10)
            self.assertIn("simplex_beta_residual", info)
            self.assertIn("simplex_core_residual_max", info)
            self.assertIn("simplex_newton_correction_max", info)

        self.assertFalse(torch.isnan(model.log_alpha).any())
        self.assertFalse(torch.isposinf(model.log_alpha).any())
        self.assertTrue(model.simplex_beta_initialized.all())
        self.assertTrue(model.simplex_core_initialized.all())
        self.assertAlmostEqual(
            abs(model.log_alpha.exp().sum().item() - 1),
            info["simplex_beta_residual"],
            places=12,
        )
        self.assertAlmostEqual(
            (model.log_cp_cores.exp().sum(dim=1) - 1).abs().amax().item(),
            info["simplex_core_residual_max"],
            places=12,
        )

    def test_state_dict_stays_compatible_and_checkpoint_keeps_newton_state(self):
        model = self._make_model()
        self.assertFalse(any(key.startswith("simplex_") for key in model.state_dict()))
        model.simplex_beta_lambda.fill_(0.25)
        model.simplex_beta_initialized.fill_(True)
        model.simplex_core_lambda.fill_(-0.125)
        model.simplex_core_initialized.fill_(True)
        checkpoint = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(checkpoint)
        restored = self._make_model()

        restored.load_state_dict(checkpoint["state_dict"])
        restored.on_load_checkpoint(checkpoint)

        torch.testing.assert_close(
            restored.simplex_beta_lambda,
            model.simplex_beta_lambda,
        )
        torch.testing.assert_close(
            restored.simplex_core_lambda,
            model.simplex_core_lambda,
        )
        self.assertTrue(restored.simplex_beta_initialized.all())
        self.assertTrue(restored.simplex_core_initialized.all())

    def test_unobserved_target_category_preserves_support(self):
        torch.manual_seed(0)
        model = self._make_model()
        previous = model.log_cp_cores[:, 2].detach().clone()
        x0 = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        x1 = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.long)

        model.mm_step(x0, x1)

        self.assertFalse(torch.equal(model.log_cp_cores[:, 2], previous))
        self.assertTrue(torch.isfinite(model.log_cp_cores).all())

    def test_category_missing_then_observed_keeps_objective_finite(self):
        torch.manual_seed(0)
        model = self._make_model()

        first_x0 = torch.tensor(
            [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.long
        )
        first_x1 = first_x0.clone()
        model.mm_step(first_x0, first_x1)

        second_x0 = torch.tensor(
            [[2, 2], [0, 0], [1, 1]], dtype=torch.long
        )
        second_x1 = second_x0.clone()
        info = model.mm_step(second_x0, second_x1)

        self.assertTrue(torch.isfinite(torch.tensor(info["loss_before"])))
        self.assertTrue(torch.isfinite(torch.tensor(info["loss"])))
        self.assertTrue(torch.isfinite(model.log_cp_cores).all())


if __name__ == "__main__":
    unittest.main()
