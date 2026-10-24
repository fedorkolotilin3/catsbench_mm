import importlib.util
from functools import partial
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import torch
from hydra import compose, initialize_config_dir

from src.data.prior import Prior
from src.methods.dlight_sb import DLightSB
from src.methods.dlight_sb_mm import DLightSBMM, _project_potentials_to_simplex
from src.methods.dlight_sb_mm_simplex import (
    DLightSBMMSimplex,
    _simplex_minimum,
    _simplex_minimum_from_logs,
)
from src.methods.mm_regularized import DLightSBMMRegularized


ROOT = Path(__file__).resolve().parents[1]


def make_model(cls=DLightSBMM, **kwargs):
    prior = Prior(
        alpha=0.1,
        num_categories=3,
        num_timesteps=1,
        num_skip_steps=1,
        eps=0.0,
        prior_type="uniform",
        dtype=torch.float64,
    )
    optimizer = kwargs.pop("optimizer", None)
    model = cls(
        prior=prior,
        dim=2,
        num_categories=3,
        num_potentials=2,
        num_timesteps=1,
        optimizer=optimizer,
        scheduler=None,
        distr_init="uniform",
        entropy_lambda=0.0,
        entropy_warmup_steps=1,
        sample_prob=0.9,
        tau=1.0,
        **kwargs,
    )
    model.init_weights()
    return model


def data():
    return (
        torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long),
        torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.long),
    )


class UnifiedOrdinaryMMTest(unittest.TestCase):
    def test_regularization_is_a_parameter_not_a_separate_implementation(self):
        base = make_model()
        unified = DLightSBMM.from_model(
            base,
            regularization_lambda_alpha=0.01,
            regularization_lambda_cores=0.5,
            clip_updates=False,
        )
        compatibility = DLightSBMMRegularized.from_model(base)

        unified_info = unified.mm_step(*data())
        compatibility_info = compatibility.mm_step(*data())

        self.assertEqual(unified_info, compatibility_info)
        torch.testing.assert_close(unified.log_alpha, compatibility.log_alpha)
        torch.testing.assert_close(
            unified.log_cp_cores, compatibility.log_cp_cores
        )

    def test_optional_projection_normalizes_and_preserves_data_loss(self):
        model = make_model(project_to_simplex_after_step=True)
        info = model.mm_step(*data())

        self.assertLessEqual(abs(info["projection_loss_delta"]), 1e-10)
        torch.testing.assert_close(
            model.log_alpha.exp().sum(), model.log_alpha.new_tensor(1.0)
        )
        torch.testing.assert_close(
            model.log_cp_cores.exp().sum(dim=1),
            torch.ones_like(model.log_cp_cores[:, 0]),
        )

    def test_projection_rejects_nonzero_regularization(self):
        with self.assertRaisesRegex(ValueError, "zero regularization"):
            make_model(
                regularization_lambda_alpha=0.01,
                project_to_simplex_after_step=True,
            )

    def test_projection_helper_preserves_component_scores_up_to_global_shift(self):
        generator = torch.Generator().manual_seed(7)
        beta = torch.randn(3, generator=generator, dtype=torch.float64)
        cores = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64)
        projected_beta, projected_cores = _project_potentials_to_simplex(
            beta, cores
        )
        before = beta + cores[:, 0].sum(dim=0)
        after = projected_beta + projected_cores[:, 0].sum(dim=0)
        shift = after - before
        torch.testing.assert_close(shift, shift[0].expand_as(shift))

    def test_fixed_beta_is_a_parameter_of_ordinary_mm(self):
        model = make_model(freeze_beta=True)
        beta_before = model.log_alpha.detach().clone()
        cores_before = model.log_cp_cores.detach().clone()
        model.mm_step(*data())
        torch.testing.assert_close(model.log_alpha, beta_before)
        self.assertFalse(torch.equal(model.log_cp_cores, cores_before))


class SimplexMMTest(unittest.TestCase):
    def test_simplex_rejects_fixed_beta(self):
        with self.assertRaisesRegex(ValueError, "requires beta updates"):
            make_model(DLightSBMMSimplex, freeze_beta=True)

    def test_log_bisection_matches_direct_solver(self):
        a = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([[8.0, 2.0, 1.0]], dtype=torch.float64)
        expected = _simplex_minimum(a, b, bisection_steps=100)
        actual = _simplex_minimum_from_logs(
            a.log(), b.log(), bisection_steps=32, compile_kernel=False
        )
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(1, dtype=a.dtype))

    def test_simplex_method_is_monotone_and_feasible(self):
        model = make_model(DLightSBMMSimplex, compile_simplex_solver=False)
        for _ in range(3):
            info = model.mm_step(*data())
            self.assertLessEqual(info["loss"], info["loss_before"] + 1e-10)
            self.assertLessEqual(info["simplex_beta_residual"], 1e-12)
            self.assertLessEqual(info["simplex_core_residual_max"], 1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_compiled_cuda_simplex_path(self):
        model = make_model(
            DLightSBMMSimplex,
            bisection_steps=4,
        ).float().cuda()
        x0, x1 = (value.cuda() for value in data())
        info = model.mm_step(x0, x1)
        self.assertTrue(torch.isfinite(model.log_alpha).all())
        self.assertTrue(torch.isfinite(model.log_cp_cores).all())
        self.assertLessEqual(info["simplex_beta_residual"], 2e-5)
        self.assertLessEqual(info["simplex_core_residual_max"], 2e-5)


class UnifiedConfigurationTest(unittest.TestCase):
    def test_all_method_families_compose(self):
        configs = str(ROOT / "configs")
        families = {
            "dlight_sb": "src.methods.dlight_sb.DLightSB",
            "dlight_sb_sgd": "src.methods.dlight_sb.DLightSB",
            "dlight_sb_fixed_beta": "src.methods.dlight_sb.DLightSB",
            "dlight_sb_mm": "src.methods.dlight_sb_mm.DLightSBMM",
            "mm_fixed_beta": "src.methods.dlight_sb_mm.DLightSBMM",
            "mm_regularized": "src.methods.dlight_sb_mm.DLightSBMM",
            "mm_projected": "src.methods.dlight_sb_mm.DLightSBMM",
            "mm_simplex": "src.methods.dlight_sb_mm_simplex.DLightSBMMSimplex",
        }
        with initialize_config_dir(config_dir=configs, version_base=None):
            for family, target in families.items():
                for dimension in (2, 16, 64):
                    experiment = f"{family}/benchmark_hd/d{dimension}_g002"
                    with self.subTest(experiment=experiment):
                        config = compose(
                            config_name="config",
                            overrides=[f"experiment={experiment}"],
                        )
                        self.assertEqual(config.method._target_, target)
                        self.assertTrue(config.experiment_description)
                        if "mm" in experiment:
                            self.assertLessEqual(
                                config.trainer.check_val_every_n_epoch, 5
                            )
                            self.assertEqual(
                                config.callbacks.periodic_checkpoint.num_checkpoints,
                                20,
                            )

    def test_sgd_experiment_only_changes_optimizer_and_step_budget(self):
        configs = str(ROOT / "configs")
        with initialize_config_dir(config_dir=configs, version_base=None):
            adam = compose(
                config_name="config",
                overrides=["experiment=dlight_sb/benchmark_hd/d16_g002"],
            )
            sgd = compose(
                config_name="config",
                overrides=["experiment=dlight_sb_sgd/benchmark_hd/d16_g002"],
            )

        self.assertEqual(sgd.method.optimizer._target_, "torch.optim.SGD")
        self.assertEqual(sgd.method.optimizer.lr, 0.1)
        self.assertEqual(sgd.method.optimizer.momentum, 0.0)
        self.assertEqual(sgd.method.optimizer.weight_decay, 0.0)
        self.assertEqual(sgd.method.entropy_lambda, 0)
        self.assertEqual(sgd.data.num_train_batches, 50000)
        self.assertEqual(sgd.trainer.max_steps, 50000)
        self.assertGreater(sgd.data.num_train_batches, adam.data.num_train_batches)

        for key in (
            "dim",
            "num_categories",
            "num_potentials",
            "num_timesteps",
            "distr_init",
            "entropy_warmup_steps",
            "entropy_lambda",
            "sample_prob",
            "tau",
            "freeze_beta",
        ):
            with self.subTest(key=key):
                self.assertEqual(sgd.method[key], adam.method[key])

    def test_gradient_baseline_can_freeze_beta(self):
        model = make_model(
            DLightSB,
            optimizer=partial(torch.optim.AdamW, lr=1e-2),
            freeze_beta=True,
        )
        optimizer = model.configure_optimizers()["optimizer"]
        beta_before = model.log_alpha.detach().clone()
        cores_before = model.log_cp_cores.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(*data())
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(model.log_alpha, beta_before)
        self.assertFalse(torch.equal(model.log_cp_cores, cores_before))

    def test_runner_dry_run_is_notebook_independent(self):
        for experiment in (
            "mm_simplex/benchmark_hd/d16_g002",
            "dlight_sb_sgd/benchmark_hd/d16_g002",
        ):
            with self.subTest(experiment=experiment):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_experiment.py"),
                        experiment,
                        "--archive",
                        "--dry-run",
                    ],
                    cwd=ROOT,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=True,
                )
                self.assertIn("-m src.run", result.stdout)
                self.assertIn("task_name=test", result.stdout)
                self.assertIn(experiment, result.stdout)

    def test_archive_contains_manifest_config_logs_and_checkpoint(self):
        script = ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("run_experiment", script)
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(runner)

        temp_parent = ROOT / ".pytest_cache"
        temp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            project_root = Path(directory)
            run_dir = project_root / "logs" / "run"
            required = {
                "checkpoints/last.ckpt": b"checkpoint",
                ".hydra/config.yaml": b"seed: 42\n",
                ".hydra/overrides.yaml": b"[]\n",
                "train.log": b"train\n",
                "test.log": b"test\n",
                "csv/train/metrics.csv": b"step,loss\n1,0.5\n",
            }
            for relative, contents in required.items():
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)

            with (
                patch.object(runner, "PROJECT_ROOT", project_root),
                patch.object(
                    runner,
                    "_git",
                    side_effect=lambda *args: {
                        ("rev-parse", "HEAD"): "1234567890abcdef",
                        ("branch", "--show-current"): "master",
                        ("status", "--porcelain"): "",
                    }[args],
                ),
            ):
                archive = runner._archive_run(
                    run_dir,
                    "mm_simplex/benchmark_hd/d16_g002",
                    42,
                    "32-true",
                    "2026-10-24_04-00-00",
                )

            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                manifest = json.loads(bundle.read("manifest.json"))
            self.assertIn("run/checkpoints/last.ckpt", names)
            self.assertIn("run/.hydra/config.yaml", names)
            self.assertIn("run/csv/train/metrics.csv", names)
            self.assertEqual(manifest["git_branch"], "master")
            self.assertEqual(manifest["git_commit"], "1234567890abcdef")
            self.assertEqual(
                set(manifest["files"]),
                set(required),
            )


if __name__ == "__main__":
    unittest.main()
