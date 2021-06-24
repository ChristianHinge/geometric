from datetime import datetime
from typing import Union

import hydra
from omegaconf import DictConfig

from src.models import test_model, train_model
from src.models.optimiser import Optimiser
from src.settings.configurations import Evaluation, Training


def run_train(cfg: Union[Training, DictConfig], seed, azure):
    train_model.train(
        seed, name=datetime.strftime(datetime.now(), "%d-%H-%M"), **cfg, azure=azure
    )


def run_eval(cfg: Union[Evaluation, DictConfig], seed):
    test_model.eval(seed, **cfg)


def run_optimise(cfg: DictConfig, seed):
    Optimiser(seed).optimise(cfg)


@hydra.main(config_path="config", config_name="run_mode")
def main(cfg: DictConfig):
    seed = cfg["seed"]
    azure = cfg["azure"]
    cfg = cfg["mode"]

    if "train" in cfg:
        run_train(cfg["train"], seed, azure)

    if "evaluate" in cfg:
        run_eval(cfg["evaluate"], seed)

    if "optimise" in cfg:
        run_optimise(cfg["optimise"], seed)


if __name__ == "__main__":
    main()
