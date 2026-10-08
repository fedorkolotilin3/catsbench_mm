from typing import Any, Mapping, Optional

from lightning import LightningDataModule, Trainer

from src.methods.base import BaseMethod

from .base import BaseWorkflow


class FitWorkflow(BaseWorkflow):
    """Run a single standard Lightning fit."""

    def train(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: Optional[str],
        hparams: Mapping[str, Any],
    ) -> Trainer:
        trainer = self.make_trainer(hparams)
        trainer.fit(
            model=method,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        return trainer
