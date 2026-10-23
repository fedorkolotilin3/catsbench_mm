from copy import deepcopy
from functools import partial
import unittest

import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from src.data.prior import Prior
from src.methods.dlight_sb import DLightSB


class FixedBetaGradientTest(unittest.TestCase):
    def make_model(self):
        prior = Prior(
            alpha=0.1,
            num_categories=3,
            num_timesteps=1,
            num_skip_steps=1,
            eps=0.0,
            prior_type="uniform",
            dtype=torch.float64,
        )
        model = DLightSB(
            prior=deepcopy(prior),
            dim=2,
            num_categories=3,
            num_potentials=2,
            num_timesteps=1,
            optimizer=partial(torch.optim.AdamW, lr=1e-2),
            scheduler=None,
            distr_init="uniform",
            entropy_lambda=0.0,
            entropy_warmup_steps=1,
            sample_prob=0.9,
            tau=1.0,
            freeze_beta=True,
        )
        model.init_weights()
        return model

    def test_default_adamw_updates_cores_but_not_beta(self):
        model = self.make_model()
        optimizer = model.configure_optimizers()["optimizer"]
        beta_before = model.log_alpha.detach().clone()
        cores_before = model.log_cp_cores.detach().clone()
        x0 = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        x1 = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(x0, x1)
        loss.backward()
        optimizer.step()

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertFalse(model.log_alpha.requires_grad)
        self.assertIsNone(model.log_alpha.grad)
        self.assertTrue(torch.equal(model.log_alpha, beta_before))
        self.assertFalse(torch.equal(model.log_cp_cores, cores_before))

    def test_baseline_experiments_enable_fixed_beta(self):
        configs = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=configs, version_base=None):
            for dimension in (2, 16, 64):
                cfg = compose(
                    config_name="config",
                    overrides=[
                        f"experiment=dlight_sb/benchmark_hd/d{dimension}_g002"
                    ],
                )
                self.assertTrue(cfg.method.freeze_beta)
                self.assertEqual(
                    cfg.method.optimizer._target_,
                    "torch.optim.AdamW",
                )
                self.assertEqual(cfg.data.num_train_batches, 20000)
                self.assertEqual(cfg.trainer.max_epochs, 1)
                self.assertEqual(cfg.trainer.max_steps, 20000)
                self.assertEqual(cfg.trainer.log_every_n_steps, 1)
                self.assertEqual(cfg.trainer.val_check_interval, 333)
                self.assertEqual(cfg.logger.csv.flush_logs_every_n_steps, 1)


if __name__ == "__main__":
    unittest.main()
