import unittest

import torch

from src.data.prior import Prior
from src.methods.dlight_sb_mm import (
    DLightSBMM,
    _logsumexp_matmul,
    _support_preserving_update_from_logs,
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


class NonnegativeBlockMinimumTest(unittest.TestCase):
    def test_explicit_update_and_zero_cases(self):
        log_a = torch.log(torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64))
        log_b = torch.tensor([torch.log(torch.tensor(8.0)), -torch.inf, -torch.inf])
        current = torch.log(torch.tensor([1.0, 7.0, 5.0], dtype=torch.float64))

        result = _support_preserving_update_from_logs(log_a, log_b, current)

        self.assertAlmostEqual(result[0].exp().item(), 4.0)
        self.assertAlmostEqual(result[1].exp().item(), 7.0)
        self.assertAlmostEqual(result[2].exp().item(), 5.0)

    def test_zero_over_zero_preserves_current_value(self):
        result = _support_preserving_update_from_logs(
            torch.tensor([-torch.inf], dtype=torch.float64),
            torch.tensor([-torch.inf], dtype=torch.float64),
            torch.log(torch.tensor([7.0], dtype=torch.float64)),
        )
        self.assertAlmostEqual(result.exp().item(), 7.0)

    def test_all_zero_coefficients_preserve_complete_block(self):
        log_a = torch.full((3,), -torch.inf, dtype=torch.float64)
        log_b = torch.full((3,), -torch.inf, dtype=torch.float64)
        current = torch.log(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))

        result = _support_preserving_update_from_logs(log_a, log_b, current)

        self.assertTrue(torch.equal(result, current))

    def test_positive_b_with_zero_a_is_unbounded(self):
        with self.assertRaisesRegex(FloatingPointError, "unbounded"):
            _support_preserving_update_from_logs(
                torch.tensor([-torch.inf], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
            )


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

    def test_mm_step_is_monotone_without_simplex_projection(self):
        torch.manual_seed(0)
        model = self._make_model()
        with torch.no_grad():
            # A global scale is a valid unnormalized representation. The old
            # simplex implementation erased it by forcing sum(beta) == 1.
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

        self.assertFalse(torch.isnan(model.log_alpha).any())
        self.assertFalse(torch.isposinf(model.log_alpha).any())
        self.assertGreater(torch.logsumexp(model.log_alpha, dim=0).item(), 1.0)

    def test_unobserved_target_category_preserves_support(self):
        torch.manual_seed(0)
        model = self._make_model()
        previous = model.log_cp_cores[:, 2].detach().clone()
        x0 = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        x1 = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.long)

        model.mm_step(x0, x1)

        self.assertTrue(torch.equal(model.log_cp_cores[:, 2], previous))
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
