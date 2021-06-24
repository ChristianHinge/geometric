import logging
import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import MUTANGDataModule
from src.models.model import GCN
from src.settings.configurations import dict_


class Optimiser:
    def __init__(self, seed: int):
        self.seed = seed

    def optimise(self, configuration):
        counts = configuration["counts"]
        self.gpu = configuration["gpu"]
        config = configuration["config"]

        wandb.login(key=os.getenv("WANDB_KEY"))
        sweep_id = wandb.sweep(
            dict_(config), project="Geometric", entity="classy_geometric"
        )
        wandb.agent(
            sweep_id,
            function=self.train,
            count=counts,
            project="Geometric",
            entity="classy_geometric",
        )

    def train(self):
        log = logging.getLogger(__name__)
        with wandb.init():
            wandb_logger = WandbLogger(
                project="geometric_hyp_opt", entity="classy_geometric"
            )

            dm = MUTANGDataModule(batch_size=wandb.config.batch_size, seed=self.seed)
            dataset = dm.prepare_data()
            torch.manual_seed(self.seed)
            model = GCN(
                dataset.num_node_features,
                dataset.num_classes,
                hidden_channels=wandb.config.layers,
                lr=wandb.config.lr,
                p=wandb.config.p,
            )
            log.debug("Model structure:")
            log.debug(model)
            log.info(wandb.config.layers)

            if self.gpu:
                kwargs = {"gpus": -1, "precision": 16}
            else:
                kwargs = {"gpus": None, "precision": 32}

            trainer = pl.Trainer(
                logger=wandb_logger,  # W&B integration
                max_epochs=wandb.config.epochs,  # number of epochs
                deterministic=True,  # keep it deterministic
                **kwargs,
            )

            trainer.fit(model, dm)
