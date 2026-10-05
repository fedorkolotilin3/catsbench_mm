import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import torch
try:
    import torch_npu # type: ignore
    from torch_npu.contrib import transfer_to_npu # type: ignore
    torch.cuda.get_device_capability = lambda x: (7, None)
except ImportError:
    pass
import lightning as L
from lightning import LightningDataModule

from .utils.ranked_logger import RankedLogger
from .methods import BaseMethod
from .methods.workflows import BaseWorkflow
from .utils import get_run_directory_from_checkpoint


if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major >= 8: 
        torch.set_float32_matmul_precision("high")

log = RankedLogger(__name__, rank_zero_only=True)

OmegaConf.register_new_resolver(
    "get_run_directory_from_checkpoint", 
    get_run_directory_from_checkpoint, 
    replace=True
)


@hydra.main(version_base='1.1', config_path='../configs', config_name='config.yaml')
def main(config: DictConfig):
    if config.get('seed'):
        L.seed_everything(config.seed, workers=True)
    
    if config.data.num_workers > 0 and config.get('trainer.strategy') is None:
        try:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # NOTE: hydra will instantiate all subobjects of the object recursively
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    log.info(f'Instantiating datamodule <{config.data._target_}>...')
    datamodule: LightningDataModule = instantiate(config.data)
    
    log.info(f'Instantiating method <{config.method._target_}>...')
    method: BaseMethod = instantiate(config.method)

    ckpt_path = config.get('ckpt_path')
    if ckpt_path == 'auto':
        ckpt_path = os.path.join(config.paths.output_dir, 'checkpoints', 'last.ckpt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f'No last checkpoint found for this experiment at {ckpt_path}. '
                'Pass ckpt_path=/path/to/checkpoint.ckpt explicitly.'
            )
    
    log.info(f'Instantiating workflow <{config.workflow._target_}>...')
    workflow: BaseWorkflow = instantiate(config.workflow)
    hparams = OmegaConf.to_container(config)

    if config.task_name == 'train':
        log.info('Starting training!')
        workflow.train(
            method=method,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            hparams=hparams,
        )

    elif config.task_name == 'test':
        assert ckpt_path is not None, 'The `ckpt_path` must be provided for testing!'
        log.info(f'Starting testing with ckpt_path: {ckpt_path}.')
        workflow.test(
            method=method,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            hparams=hparams,
        )
    else:
        raise ValueError(f'Unknown task name: {config.task_name}!')

if __name__ == '__main__':
    main()
