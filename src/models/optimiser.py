import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import MUTANGDataModule
from src.models.model import GCN


class Optimiser:

    def optimise(self, configuration, counts):
        wandb.login(key=os.getenv("WANDB_KEY"))
        sweep_id = wandb.sweep(
            configuration, project="geometric_hyp_opt", entity="classy_geometric")
        wandb.agent(sweep_id, function=self.train, count=counts, project="geometric_hyp_opt",
                    entity="classy_geometric")

    @staticmethod
    def train():
        with wandb.init():
            wandb_logger = WandbLogger(project="geometric_hyp_opt", entity="classy_geometric")

            dm = MUTANGDataModule(batch_size=wandb.config.batch_size)
            dataset = dm.prepare_data()

            model = GCN(
                dataset.num_node_features,
                dataset.num_classes,
                hidden_channels=wandb.config.layers,
                lr=wandb.config.lr,
                p=wandb.config.p
            )
            print(model)
            print(wandb.config.layers)

            if wandb.config.GPU:
                kwargs = {"gpus": -1, "precision": 16}
            else:
                kwargs = {"gpus": None, "precision": 32}

            trainer = pl.Trainer(
                logger=wandb_logger,  # W&B integration
                max_epochs=wandb.config.epochs,  # number of epochs
                deterministic=True,  # keep it deterministic
                # default_root_dir=os.path.join(CHECKPOINT_PATH, name + ".") ** kwargs,
            )

            trainer.fit(model, dm)
