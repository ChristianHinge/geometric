from datetime import datetime
from typing import Union

import hydra
import torch
from omegaconf import DictConfig

from src.models import train_model, test_model
from src.models.optimiser import Optimiser
from src.settings.configurations import Training, Evaluation


def run_train(cfg: Union[Training, DictConfig]):
    train_model.train(name=datetime.strftime(datetime.now(), '%d-%H-%M'), **cfg)


def run_eval(cfg: Union[Evaluation, DictConfig]):
    test_model.eval(**cfg)


def run_optimise(cfg: DictConfig):
    cfg = cfg['optimise']
    Optimiser().optimise(cfg['config'], cfg['counts'])


@hydra.main(config_path='config', config_name='run_mode')
def main(cfg: DictConfig):
    torch.manual_seed(cfg['seed'])

    if 'train' in cfg:
        run_train(cfg['train'])

    if 'evaluate' in cfg:
        run_eval(cfg['evaluate'])

    if 'optimise' in cfg:
        run_optimise(cfg)


if __name__ == '__main__':
    main()
