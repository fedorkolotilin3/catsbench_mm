import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from scipy.optimize import brentq
from torch.utils.data import DataLoader, TensorDataset

import test_dlight_sb_mm as existing
from src.methods.mm_regularized import (
    DLightSBMMRegularized, _exponential_log_update, _exponential_log_penalty,
)


class ExponentialLogBlockTest(unittest.TestCase):
    def test_penalty_is_nonnegative_and_accurate_near_zero(self):
        for dtype in (torch.float32, torch.float64):
            u = torch.tensor([-1e-8, 0., 1e-8], dtype=dtype)
            actual = _exponential_log_penalty(u)
            torch.testing.assert_close(actual, u.square() / 2, atol=1e-24, rtol=1e-6)
            self.assertTrue((actual >= 0).all())

    def test_roots_against_independent_scalar_solver(self):
        for dtype in (torch.float32, torch.float64):
            log_a = torch.tensor([-1000., -20., 0., 20., 1000., 0., -torch.inf,
                                  -torch.inf], dtype=dtype)
            log_b = torch.tensor([0., -5., -1., 0., 0., -torch.inf, -1.,
                                  -torch.inf], dtype=dtype)
            strength = 0.03
            actual = _exponential_log_update(log_a, log_b, torch.zeros_like(log_a), strength)
            expected = []
            for la, lb in zip(log_a.double().tolist(), log_b.double().tolist()):
                # Independent ordinary derivative; clipping only avoids
                # overflowing far away from the root in the reference solver.
                import math
                def derivative(u):
                    return math.exp(min(la + u, 700)) + strength * math.exp(min(u, 700)) - math.exp(lb) - strength
                expected.append(brentq(derivative, -2000., 2000., xtol=1e-12))
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype),
                                       atol=2e-4 if dtype == torch.float32 else 1e-10,
                                       rtol=2e-6 if dtype == torch.float32 else 1e-10)


class RegularizedMMTest(unittest.TestCase):
    def setUp(self):
        # Tests cover multiple shapes, strides and dtypes in one process.
        torch._dynamo.reset()

    def make_model(self, strength=0.0001):
        base = existing.DLightSBMMUpdateTest()._make_model()
        return DLightSBMMRegularized.from_model(base, regularization_lambda=strength)

    def data(self):
        return (torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long),
                torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.long))

    def test_zero_strength_matches_original_including_missing_categories(self):
        base = existing.DLightSBMMUpdateTest()._make_model()
        model = DLightSBMMRegularized.from_model(base, regularization_lambda=0.)
        for _ in range(3):
            original = base.mm_step(*self.data())
            actual = model.mm_step(*self.data())
            self.assertEqual(original["loss"], actual["loss"])
            self.assertTrue(torch.equal(base.log_alpha, model.log_alpha))
            self.assertTrue(torch.equal(base.log_cp_cores, model.log_cp_cores))

    def test_regularized_objective_decreases_and_matches_public_loss(self):
        for dtype in (torch.float32, torch.float64):
            torch._dynamo.reset()
            model = self.make_model().to(dtype=dtype)
            previous = model.log_cp_cores[:, 2].detach().clone()
            for _ in range(4):
                info = model.mm_step(*self.data(), inner_sweeps=2)
                self.assertLessEqual(info["loss"], info["loss_before"] + 1e-5)
                self.assertGreaterEqual(info["surrogate_decrease"], -1e-5)
                # Base public loss evaluates the potential independently of
                # the cached coefficients used by mm_step.
                data_loss, _ = existing.DLightSBMM.loss(model, *self.data())
                penalty = .0001 * ((model.log_alpha.exp() - model.log_alpha - 1).sum()
                                   + (model.log_cp_cores.exp() - model.log_cp_cores - 1).sum())
                self.assertAlmostEqual(info["data_loss"], data_loss.item(), places=5)
                self.assertAlmostEqual(info["loss"], (data_loss + penalty).item(), places=5)
                public_loss, terms = model.loss(*self.data())
                self.assertAlmostEqual(public_loss.item(), info["loss"], places=5)
                self.assertAlmostEqual(terms["regularization"].item(), penalty.item(), places=5)
            self.assertTrue(torch.isfinite(model.log_cp_cores).all())
            self.assertFalse(torch.equal(previous, model.log_cp_cores[:, 2]))

    def test_invalid_strength_and_exact_zero_parameters(self):
        for strength in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, "regularization_lambda"):
                self.make_model(strength)
        model = self.make_model()
        with torch.no_grad():
            model.log_alpha[0] = -torch.inf
        original = model.log_cp_cores.detach().clone()
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            model.mm_step(*self.data())
        self.assertTrue(torch.equal(original, model.log_cp_cores))

    def test_lightning_checkpoint_preserves_penalty_and_resume(self):
        model = self.make_model(.02)
        loader = DataLoader(TensorDataset(*self.data()), batch_size=3)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(accelerator="cpu", max_steps=2, max_epochs=2,
                              logger=False, enable_checkpointing=False,
                              enable_progress_bar=False, enable_model_summary=False,
                              num_sanity_val_steps=0, precision="64-true",
                              default_root_dir=directory)
            trainer.fit(model, train_dataloaders=loader)
            self.assertEqual(trainer.global_step, 2)
            checkpoint = str(Path(directory) / "model.ckpt")
            trainer.save_checkpoint(checkpoint)
            restored = DLightSBMMRegularized.load_from_checkpoint(
                checkpoint, prior=model.prior, optimizer=None, weights_only=False
            )
            self.assertEqual(restored.hparams.regularization_lambda, .02)
            torch.testing.assert_close(restored.log_alpha, model.log_alpha)
            info = restored.mm_step(*self.data())
            self.assertLessEqual(info["loss"], info["loss_before"] + 1e-10)

    def test_experiment_configs_inherit_mm_settings(self):
        configs = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=configs, version_base=None):
            for dimension in (2, 16, 64):
                name = f"benchmark_hd/d{dimension}_g002"
                base = compose(config_name="config", overrides=[f"experiment=dlight_sb_mm/{name}"])
                cfg = compose(config_name="config", overrides=[f"experiment=mm_regularized/{name}"])
                self.assertEqual(cfg.method._target_, "src.methods.mm_regularized.DLightSBMMRegularized")
                self.assertEqual(cfg.method.regularization_lambda, .0001)
                self.assertEqual(cfg.data, base.data)
                self.assertEqual(cfg.trainer, base.trainer)


if __name__ == "__main__":
    unittest.main()
