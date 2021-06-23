from datetime import datetime
from typing import Union

import hydra
from omegaconf import DictConfig

from src.models import train_model, test_model
from src.models.optimiser import Optimiser
from src.settings.configurations import Training, Evaluation, dict_


def run_train(cfg: Union[Training, DictConfig], seed):
    train_model.train(seed, name=datetime.strftime(datetime.now(), '%d-%H-%M'), **cfg)


def run_eval(cfg: Union[Evaluation, DictConfig], seed):
    test_model.eval(seed, **cfg)


def run_optimise(cfg: DictConfig, seed):
    cfg = cfg['optimise']
    Optimiser(seed).optimise(dict_(cfg['config']), cfg['counts'])


@hydra.main(config_path='config', config_name='run_mode')
def main(cfg: DictConfig):
    seed = cfg['seed']
    cfg = cfg['mode']

    if 'train' in cfg:
        run_train(cfg['train'], seed)

    if 'evaluate' in cfg:
        run_eval(cfg['evaluate'], seed)

    if 'optimise' in cfg:
        run_optimise(cfg, seed)


if __name__ == '__main__':
    main()
