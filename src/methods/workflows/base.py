from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig

from src.methods.base import BaseMethod
from src.utils import instantiate_callbacks, instantiate_loggers
from src.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class BaseWorkflow(ABC):
    def __init__(
        self,
        trainer_config: DictConfig,
        callbacks_config: Optional[DictConfig],
        logger_config: Optional[DictConfig],
    ) -> None:
        self.trainer_config = trainer_config
        self.callbacks_config = callbacks_config
        self.logger_config = logger_config
        self.loggers = None

    def make_trainer(
        self,
        hparams: Mapping[str, Any],
        **overrides,
    ) -> Trainer:
        """Instantiate a fresh Trainer and callbacks, reusing workflow loggers."""
        log.info("Instantiating callbacks...")
        callbacks = instantiate_callbacks(self.callbacks_config)

        if self.loggers is None:
            log.info("Instantiating loggers...")
            self.loggers = instantiate_loggers(
                self.logger_config,
                self.trainer_config.get("default_root_dir"),
            )
            for logger in self.loggers:
                logger.log_hyperparams(hparams)

        log.info(f"Instantiating trainer <{self.trainer_config._target_}>...")
        return instantiate(
            self.trainer_config,
            callbacks=callbacks,
            logger=self.loggers,
            **overrides,
        )

    @abstractmethod
    def train(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: Optional[str],
        hparams: Mapping[str, Any],
    ) -> Optional[Trainer]:
        """Run the training procedure and return its Trainer."""

    def test(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: str,
        hparams: Mapping[str, Any],
    ) -> Trainer:
        trainer = self.make_trainer(hparams)
        trainer.test(
            model=method,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        return trainer
