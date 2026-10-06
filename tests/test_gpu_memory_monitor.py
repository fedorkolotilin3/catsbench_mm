import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.callbacks import GpuMemoryMonitor


class GpuMemoryMonitorTest(unittest.TestCase):
    def test_rejects_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            GpuMemoryMonitor(every_n_steps=0)

    def test_logs_two_peak_metrics_at_configured_interval(self) -> None:
        trainer = SimpleNamespace(
            strategy=SimpleNamespace(root_device=torch.device("cuda")),
            global_step=10,
        )
        module = SimpleNamespace(log=Mock())
        monitor = GpuMemoryMonitor(every_n_steps=10)

        with (
            patch("torch.cuda.max_memory_allocated", return_value=1024),
            patch("torch.cuda.max_memory_reserved", return_value=2048),
        ):
            monitor.on_train_batch_end(trainer, module, None, None, 9)

        calls = module.log.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, ("gpu/allocated_bytes_peak", 1024.0))
        self.assertEqual(calls[1].args, ("gpu/reserved_bytes_peak", 2048.0))

    def test_skips_cpu_and_non_logging_steps(self) -> None:
        module = SimpleNamespace(log=Mock())
        monitor = GpuMemoryMonitor(every_n_steps=10)

        cpu_trainer = SimpleNamespace(
            strategy=SimpleNamespace(root_device=torch.device("cpu")),
            global_step=10,
        )
        monitor.on_train_batch_end(cpu_trainer, module, None, None, 9)

        cuda_trainer = SimpleNamespace(
            strategy=SimpleNamespace(root_device=torch.device("cuda")),
            global_step=9,
        )
        monitor.on_train_batch_end(cuda_trainer, module, None, None, 8)

        module.log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
