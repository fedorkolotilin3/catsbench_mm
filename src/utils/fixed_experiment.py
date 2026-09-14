"""Shared initialization, full empirical loss and CPU timing for both solvers."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
from lightning import Callback
from catsbench.metrics import ShapeScore, TrendScore


class FixedExperiment(Callback):
    def __init__(self, num_threads=4, eval_batch_size=256):
        self.num_threads = num_threads
        self.eval_batch_size = eval_batch_size

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        self.started = perf_counter()
        if trainer.strategy.root_device.type != "cpu" or trainer.world_size != 1:
            raise ValueError("This experiment requires a single CPU process")
        if pl_module.hparams.entropy_lambda != 0:
            raise ValueError("Compare the unregularized loss: entropy_lambda=0")
        torch.set_num_threads(self.num_threads)
        pl_module.double()
        self.data = trainer.datamodule.payload
        batch_size = trainer.datamodule.hparams.batch_size
        n = len(self.data["train_x"])
        if batch_size is None:
            if trainer.accumulate_grad_batches != 1:
                raise ValueError("A full batch requires accumulate_grad_batches=1")
        elif (batch_size < 1 or n != len(self.data["train_y"]) or n % batch_size
              or trainer.accumulate_grad_batches != n // batch_size
              or trainer.limit_train_batches != 1.0 or not pl_module.automatic_optimization):
            raise ValueError("Use equal microbatches and accumulate one complete dataset per AdamW step")
        if pl_module.hparams.num_potentials != self.data["metadata"]["num_potentials"]:
            raise ValueError("num_potentials differs from the saved initialization")
        # Callback.setup runs before DLightSB.setup, preventing random reinitialization.
        for name, expected in self.data["prior_buffers"].items():
            actual = dict(pl_module.prior.named_buffers())[name]
            if not torch.equal(actual, expected):
                raise ValueError(f"Configured prior differs from the fixed experiment: {name}")
        initial = self.data["initial_state"]
        current = pl_module.state_dict()
        if current.keys() != initial.keys() or any(current[k].shape != initial[k].shape for k in initial):
            raise ValueError("Model structure differs from the fixed initialization")
        pl_module.load_state_dict(initial)
        pl_module._did_weight_init = True
        self.rows = []
        self.update_seconds = 0.0
        self.pending_step_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.out = Path(trainer.default_root_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # Compression changes neither empirical weights nor the inherited loss.
        self.empirical = {}
        for name in ("train_x", "train_y", "test_x", "test_y"):
            unique, counts = torch.unique(self.data[name], dim=0, return_counts=True)
            self.empirical[name] = (unique, counts.double() / len(self.data[name]))

    @torch.no_grad()
    def empirical_loss(self, model, split):
        terms = []
        for suffix, fn in (("x", model.get_log_c), ("y", model.get_log_v)):
            x, weights = self.empirical[f"{split}_{suffix}"]
            values = torch.cat([fn(b) for b in x.split(self.eval_batch_size)])
            terms.append(weights @ values)
        return (terms[0] - terms[1]).item()

    def record(self, trainer, model, step_seconds):
        start = perf_counter()
        loss = self.empirical_loss(model, "train")
        self.evaluation_seconds += perf_counter() - start
        row = dict(step=trainer.global_step, train_loss=loss,
                   step_seconds=step_seconds, update_seconds=self.update_seconds,
                   elapsed_seconds=perf_counter() - self.started,
                   evaluation_seconds=self.evaluation_seconds)
        self.rows.append(row)
        for logger in trainer.loggers:
            logger.log_metrics({f"experiment/{k}": v for k, v in row.items() if k != "step"},
                               step=trainer.global_step)

    def on_fit_start(self, trainer, pl_module):
        self.record(trainer, pl_module, 0.0)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_started = perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # CPU operations are synchronous. Stop before shared evaluation and disk I/O.
        seconds = perf_counter() - self.batch_started
        self.update_seconds += seconds
        self.pending_step_seconds += seconds
        if trainer.global_step == self.rows[-1]["step"]:
            return
        self.record(trainer, pl_module, self.pending_step_seconds)
        self.pending_step_seconds = 0.0
        if hasattr(pl_module, "mm_step"):
            previous, current = self.rows[-2]["train_loss"], self.rows[-1]["train_loss"]
            if current > previous + 1e-9 * max(1.0, abs(previous)):
                raise FloatingPointError("Inherited full train loss increased after an MM step")
        if trainer.global_step % 50 == 0:
            print(f"step={trainer.global_step} loss={self.rows[-1]['train_loss']:.6f} "
                  f"updates={self.update_seconds:.2f}s", flush=True)

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module):
        training_wall_seconds = perf_counter() - self.started
        eval_started = perf_counter()
        pl_module.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.data["metadata"]["sampling_seed"])

            def predict(x):
                return torch.cat([pl_module.sample(b) for b in x.split(self.eval_batch_size)])

            pred = predict(self.data["test_x"])
            scores = {}
            for cls in (ShapeScore, TrendScore):
                metric = cls(dim=2, num_categories=50)
                metric.update(self.data["test_y"], pred)
                scores[cls.__name__] = metric.compute().item()
            metric = TrendScore(dim=2, num_categories=50, conditional=True)
            for x, truth in zip(self.data["conditions"], self.data["conditional_truth"]):
                metric.update(truth, predict(x[None].repeat(len(truth), 1)))
            scores["ConditionalTrendScore"] = metric.compute().item()
        test_loss = self.empirical_loss(pl_module, "test")
        summary = dict(
            method=type(pl_module).__name__, steps=trainer.global_step,
            stop_reason=("tolerance" if bool(trainer.callback_metrics.get("train/converged", 0))
                         else "limit"),
            train_loss=self.rows[-1]["train_loss"],
            test_loss=test_loss if torch.isfinite(torch.tensor(test_loss)) else str(test_loss),
            update_seconds=self.update_seconds,
            training_wall_seconds=training_wall_seconds,
            loss_evaluation_seconds=self.evaluation_seconds,
            final_evaluation_seconds=perf_counter() - eval_started,
            file_sha256=trainer.datamodule.file_sha256,
            num_threads=torch.get_num_threads(), dtype=str(pl_module.dtype),
            torch=str(torch.__version__), python=platform.python_version(),
            platform=platform.platform(), **scores,
        )
        with (self.out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        torch.save(pred, self.out / "predictions.pt")
        trainer.save_checkpoint(self.out / "final.ckpt")
        summary["total_seconds"] = perf_counter() - self.started
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
