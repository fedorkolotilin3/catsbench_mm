import unittest
from pathlib import Path
from unittest.mock import Mock, sentinel

from hydra import compose, initialize_config_dir

from src.methods.workflows.fit import FitWorkflow


class TestTrainingResume(unittest.TestCase):
    def test_fit_loads_full_trusted_checkpoint(self):
        workflow = FitWorkflow(
            trainer_config=None,
            callbacks_config=None,
            logger_config=None,
        )
        trainer = Mock()
        workflow.make_trainer = Mock(return_value=trainer)

        result = workflow.train(
            method=sentinel.method,
            datamodule=sentinel.datamodule,
            ckpt_path="last.ckpt",
            hparams={},
        )

        trainer.fit.assert_called_once_with(
            model=sentinel.method,
            datamodule=sentinel.datamodule,
            ckpt_path="last.ckpt",
            weights_only=False,
        )
        self.assertIs(result, trainer)

    def test_mm_epoch_and_step_limits_match(self):
        config_dir = str((Path(__file__).parents[1] / "configs").resolve())
        expected_limits = {
            "dlight_sb_mm/benchmark_hd/d2_g002": 1000,
            "dlight_sb_mm/benchmark_hd/d16_g002": 2000,
            "dlight_sb_mm/benchmark_hd/d64_g002": 1000,
        }

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            for experiment, expected_limit in expected_limits.items():
                with self.subTest(experiment=experiment):
                    config = compose(
                        config_name="config",
                        overrides=[f"experiment={experiment}"],
                    )
                    self.assertEqual(config.trainer.max_epochs, expected_limit)
                    self.assertEqual(config.trainer.max_steps, expected_limit)


if __name__ == "__main__":
    unittest.main()
