import unittest

import torch

from src.data.prior import Prior
from src.methods.dlight_sb_mm import (
    DLightSBMM,
    _nonnegative_minimum_from_logs,
)


class NonnegativeBlockMinimumTest(unittest.TestCase):
    def test_explicit_update_and_zero_cases(self):
        log_a = torch.log(torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64))
        log_b = torch.tensor([torch.log(torch.tensor(8.0)), -torch.inf, -torch.inf])
        current = torch.log(torch.tensor([1.0, 7.0, 5.0], dtype=torch.float64))

        result = _nonnegative_minimum_from_logs(log_a, log_b, current)

        self.assertAlmostEqual(result[0].exp().item(), 4.0)
        self.assertTrue(torch.isneginf(result[1]))
        self.assertTrue(torch.isneginf(result[2]))

    def test_zero_over_zero_preserves_current_value(self):
        result = _nonnegative_minimum_from_logs(
            torch.tensor([-torch.inf], dtype=torch.float64),
            torch.tensor([-torch.inf], dtype=torch.float64),
            torch.log(torch.tensor([7.0], dtype=torch.float64)),
        )
        self.assertAlmostEqual(result.exp().item(), 7.0)

    def test_positive_b_with_zero_a_is_unbounded(self):
        with self.assertRaisesRegex(FloatingPointError, "unbounded"):
            _nonnegative_minimum_from_logs(
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

    def test_unobserved_target_category_gets_exact_zero(self):
        torch.manual_seed(0)
        model = self._make_model()
        x0 = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        x1 = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.long)

        model.mm_step(x0, x1)

        self.assertTrue(torch.isneginf(model.log_cp_cores[:, 2]).all())


if __name__ == "__main__":
    unittest.main()
