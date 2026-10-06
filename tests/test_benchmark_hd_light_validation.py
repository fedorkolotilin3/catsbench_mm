import unittest
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock, patch

import torch

transformers = ModuleType("transformers")
transformers.CLIPImageProcessor = Mock()
transformers.CLIPVisionModelWithProjection = Mock()
sys.modules.setdefault("transformers", transformers)

from src.data.batch import Batch
from src.metrics.benchmark_hd import BenchmarkHDMetricsCallback


class _Benchmark:
    reverse = False

    def __init__(self) -> None:
        self.sample = Mock(side_effect=lambda x: torch.zeros_like(x))
        self.sample_trajectory = Mock(
            return_value=(
                torch.zeros(3, 2, 2, dtype=torch.long),
                torch.zeros(2, 2, 2, 3),
            )
        )
        self.get_transition_logits = Mock(
            side_effect=lambda x, t: torch.zeros(len(x), 2, 3)
        )


def _method() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        sample=Mock(side_effect=lambda x: torch.zeros_like(x)),
        sample_trajectory=Mock(
            return_value=(
                torch.zeros(3, 2, 2, dtype=torch.long),
                torch.zeros(2, 2, 2, 3),
            )
        ),
        get_transition_logits=Mock(
            side_effect=lambda x, t: torch.zeros(len(x), 2, 3)
        ),
        log=Mock(),
        log_dict=Mock(),
    )


class BenchmarkHDLightValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.callback = BenchmarkHDMetricsCallback(
            dim=2,
            num_categories=3,
            num_cond_samples=1000,
            validation_num_cond_samples=100,
            num_timesteps=2,
        )
        self.callback.benchmark = _Benchmark()
        self.callback.metrics = Mock()
        self.callback.cond_metrics = Mock()
        self.callback.forward_kl_div = Mock()
        self.callback.reverse_kl_div = Mock()
        self.batch = Batch(
            encoded=(
                torch.zeros(2, 2, dtype=torch.long),
                torch.zeros(2, 2, dtype=torch.long),
            )
        )

    def test_validation_uses_small_conditional_sample_without_trajectories(self) -> None:
        method = _method()

        with patch("src.metrics.benchmark_hd.BenchmarkHD", _Benchmark):
            self.callback._update_metrics(None, method, self.batch, 0, stage="val")

        conditional_input = self.callback.benchmark.sample.call_args.args[0]
        self.assertEqual(len(conditional_input), 100)
        self.callback.benchmark.sample_trajectory.assert_not_called()
        method.sample_trajectory.assert_not_called()
        self.callback.forward_kl_div.update.assert_not_called()
        self.callback.reverse_kl_div.update.assert_not_called()

    def test_test_uses_full_conditional_sample_and_trajectory_kl(self) -> None:
        method = _method()

        with patch("src.metrics.benchmark_hd.BenchmarkHD", _Benchmark):
            self.callback._update_metrics(None, method, self.batch, 0, stage="test")

        conditional_input = self.callback.benchmark.sample.call_args.args[0]
        self.assertEqual(len(conditional_input), 1000)
        self.callback.benchmark.sample_trajectory.assert_called_once()
        method.sample_trajectory.assert_called_once()
        self.callback.forward_kl_div.update.assert_called_once()
        self.callback.reverse_kl_div.update.assert_called_once()

    def test_validation_does_not_log_trajectory_kl(self) -> None:
        method = _method()
        self.callback.metrics.compute.return_value = {"shape_score": torch.tensor(1.0)}
        self.callback.cond_metrics.compute.return_value = {
            "cond_shape_score": torch.tensor(1.0)
        }

        with patch("src.metrics.benchmark_hd.BenchmarkHD", _Benchmark):
            self.callback._compute_and_log_metrics(None, method, stage="val")

        self.callback.forward_kl_div.compute.assert_not_called()
        self.callback.reverse_kl_div.compute.assert_not_called()

    def test_rejects_invalid_validation_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkHDMetricsCallback(
                dim=2,
                num_categories=3,
                num_cond_samples=1000,
                validation_num_cond_samples=0,
                num_timesteps=2,
            )


if __name__ == "__main__":
    unittest.main()
