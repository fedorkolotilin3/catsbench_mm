from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.callbacks.periodic_checkpoint import PeriodicCheckpoint


class PeriodicCheckpointTest(unittest.TestCase):
    def test_evenly_spaced_steps(self):
        self.assertEqual(
            PeriodicCheckpoint.checkpoint_steps(1000, 20),
            tuple(range(50, 1001, 50)),
        )
        self.assertEqual(
            PeriodicCheckpoint.checkpoint_steps(3, 20),
            (1, 2, 3),
        )

    def test_saves_once_when_threshold_is_reached(self):
        callback = PeriodicCheckpoint(num_checkpoints=4)
        trainer = SimpleNamespace(
            max_steps=100,
            estimated_stepping_batches=100,
            global_step=0,
            checkpoint_callback=SimpleNamespace(dirpath="run/checkpoints"),
            default_root_dir="run",
            save_checkpoint=Mock(),
        )
        callback.on_fit_start(trainer, None)
        trainer.global_step = 25
        with patch("pathlib.Path.mkdir"):
            callback.on_train_batch_end(trainer, None, None, None, 0)
            callback.on_train_batch_end(trainer, None, None, None, 0)
        trainer.save_checkpoint.assert_called_once()
        path = trainer.save_checkpoint.call_args.args[0]
        self.assertEqual(path.as_posix(), "run/checkpoints/history/step_00000025.ckpt")
        self.assertFalse(trainer.save_checkpoint.call_args.kwargs["weights_only"])


if __name__ == "__main__":
    unittest.main()
