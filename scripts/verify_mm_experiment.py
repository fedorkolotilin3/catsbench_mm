"""Numerical and Lightning integration checks for the fixed MM experiment."""

import os
os.environ["LSE_BACKEND"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
sys.stdout.reconfigure(encoding="utf-8")

import tempfile
from copy import deepcopy
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.optimize import minimize

from src.data.prior import Prior
from src.methods.dlight_sb import DLightSB
from src.methods.dlight_sb_mm import DLightSBMM, _simplex_minimum
from scripts.prepare_mm_experiment import prepare


def main():
    torch.set_num_threads(4)
    torch.manual_seed(19)
    for exponent in (20, 100, 300):
        z = _simplex_minimum(torch.tensor([[0., 2.]], dtype=torch.float64),
                             torch.tensor([[10. ** -exponent, 1.]], dtype=torch.float64))
        torch.testing.assert_close(z, torch.tensor([[.5, .5]], dtype=torch.float64), atol=1e-12, rtol=0)
    for i in range(12):
        a = torch.rand(1, 7, dtype=torch.float64) * 5
        b = torch.rand_like(a)
        b[torch.rand_like(b) < .4] = 0
        if i == 0:
            b.zero_()
        z = _simplex_minimum(a, b)[0].numpy()
        an, bn = a[0].numpy(), b[0].numpy()
        positive = bn > 0

        def objective(v):
            if (v[positive] <= 0).any():
                return 1e100
            return an @ v - bn[positive] @ np.log(v[positive])

        result = minimize(objective, np.ones(7)/7, method="SLSQP", bounds=[(0, 1)]*7,
                          constraints=[{"type": "eq", "fun": lambda v: v.sum()-1}],
                          options={"ftol": 1e-11, "maxiter": 1000})
        assert result.success, result.message
        assert abs(objective(z) - result.fun) < 1e-7
        assert abs(z.sum()-1) < 1e-12
    print("Simplex: tiny counts, boundaries and independent SciPy solutions PASS")

    prior = Prior(alpha=.2, num_categories=5, num_timesteps=2, num_skip_steps=1,
                  prior_type="uniform", dtype="float64", eps=0)
    base = DLightSB(prior=prior, dim=2, num_categories=5, num_potentials=3,
                    num_timesteps=2, optimizer=partial(torch.optim.AdamW, lr=.03),
                    distr_init="gaussian", entropy_lambda=0).double()
    base.init_weights()
    x, y = torch.randint(5, (83, 2)), torch.randint(5, (71, 2))
    direct = DLightSBMM.from_model(base)
    history = direct.fit_mm(x, y, max_iter=3, tol=0)
    assert max(np.diff(history)) < 1e-10
    torch.testing.assert_close(direct.loss(x, y)[0], torch.tensor(history[-1], dtype=torch.float64))
    assert all(p.grad is None for p in direct.parameters())

    # Check descent of the actual touching bound, independently from mm_step diagnostics.
    old = DLightSBMM.from_model(base)
    with torch.no_grad():
        z = old.log_cp_cores.logsumexp(1, keepdim=True)
        old.log_cp_cores.sub_(z)
        old.log_alpha.add_(z.squeeze(1).sum(0))
        old.log_alpha.sub_(old.log_alpha.logsumexp(0))
    updated = DLightSBMM.from_model(old)
    updated.mm_step(x, y)
    with torch.no_grad():
        idx = y[:, :, None, None].expand(-1, -1, 1, 3)
        def logits(m):
            r = torch.gather(m.log_cp_cores[None].expand(len(y), -1, -1, -1), 2, idx)
            return m.log_alpha + r.squeeze(2).sum(1)
        gamma = logits(old).softmax(1)
        old_c = old.get_log_c(x)
        def bound(m):
            first = (old_c + torch.expm1(m.get_log_c(x)-old_c)).mean()
            second = -(gamma * logits(m)).sum(1).mean() + torch.xlogy(gamma, gamma).sum(1).mean()
            return first + second
        torch.testing.assert_close(bound(old), old.loss(x, y)[0], atol=1e-12, rtol=0)
        assert updated.loss(x, y)[0] <= bound(updated) + 1e-12
        assert bound(updated) <= bound(old) + 1e-12
    print("Touching majorant, descent and inherited loss agreement PASS")

    options = dict(accelerator="cpu", devices=1, precision="64-true", logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
    trained = DLightSBMM.from_model(base)
    loader = DataLoader([(x, y)], batch_size=None)
    trainer = Trainer(**options, max_steps=3, max_epochs=10)
    trainer.fit(trained, train_dataloaders=loader)
    assert trainer.global_step == 3
    for key in direct.state_dict():
        torch.testing.assert_close(trained.state_dict()[key], direct.state_dict()[key], atol=1e-12, rtol=0)
    with tempfile.TemporaryDirectory(dir=ROOT / "data") as tmp:
        path = Path(tmp) / "resume.ckpt"
        trainer.save_checkpoint(path)
        resumed = DLightSBMM.from_model(base)
        trainer2 = Trainer(**options, max_steps=5, max_epochs=10)
        trainer2.fit(resumed, train_dataloaders=loader, ckpt_path=path)
        direct.fit_mm(x, y, max_iter=2, tol=0)
        assert trainer2.global_step == 5
        for key in direct.state_dict():
            torch.testing.assert_close(resumed.state_dict()[key], direct.state_dict()[key], atol=1e-12, rtol=0)
    print("Lightning update == fit_mm; max_steps and checkpoint resume PASS")

    # The Lightning stopping decision must agree with the standalone solver.
    tolerance = 0.1
    reference = DLightSBMM.from_model(base)
    expected = reference.fit_mm(x, y, max_iter=100, tol=tolerance)
    stopping = DLightSBMM.from_model(base, tol=tolerance)
    stop_trainer = Trainer(**options, max_steps=100, max_epochs=100)
    stop_trainer.fit(stopping, train_dataloaders=loader)
    assert stop_trainer.global_step == len(expected) - 1 < 100
    assert stop_trainer.should_stop
    assert stop_trainer.callback_metrics["train/converged"].item() == 1
    for key in reference.state_dict():
        torch.testing.assert_close(stopping.state_dict()[key], reference.state_dict()[key], atol=1e-12, rtol=0)
    for invalid_tol in (-1, float("nan"), float("inf")):
        try:
            DLightSBMM.from_model(base, tol=invalid_tol)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid tolerance accepted")
    print("Lightning tolerance stopping matches fit_mm; invalid tolerances rejected PASS")

    restricted = DLightSBMM.from_model(base)
    restricted.fit_mm(x, y % 2, max_iter=3, tol=0)
    assert torch.isneginf(restricted.log_cp_cores[:, 2:]).all()
    assert restricted.sample(x).max() < 2
    snapshot = deepcopy(restricted.state_dict())
    try:
        restricted.mm_step(x[:0], y)
    except ValueError:
        pass
    else:
        raise AssertionError("Empty data accepted")
    assert all(torch.equal(snapshot[k], v) for k, v in restricted.state_dict().items())
    print("Zero-count support, inherited sampler, failed update leaves weights unchanged PASS")

    prepare()
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs")):
        for method in ("dlight_sb", "dlight_sb_mm"):
            cfg = compose(config_name="config", overrides=[f"experiment={method}/fixed_cpu"])
            cfg.paths.root_dir = str(ROOT)
            cfg.paths.output_dir = str(ROOT / "logs" / "fixed_mm_smoke" / method)
            cfg.trainer.max_steps = 2
            cfg.trainer.max_epochs = 2
            dm = instantiate(cfg.data)
            model = instantiate(cfg.method)
            callback = instantiate(cfg.callbacks.fixed_experiment)
            t = instantiate(cfg.trainer, callbacks=[callback], logger=False)
            t.fit(model, datamodule=dm)
            assert t.global_step == 2
            assert len(callback.rows) == 3
            assert callback.rows[-1]["update_seconds"] > 0
    print("Both Hydra configurations, fixed data, timing and result artifacts PASS")


if __name__ == "__main__":
    main()
