import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.datamodule import MUTANGDataModule
import os

def train(lr: float, epochs: float, batch_size: int,layers: int,GPU: bool,p: float):

    # Initialise wandb logger
    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb_logger = WandbLogger(project="Geometric", entity='classy_geometric')

    dm = MUTANGDataModule(batch_size=batch_size)
    dataset = dm.prepare_data()

    model = GCN(dataset.num_node_features, dataset.num_classes,hidden_channels=layers,lr=lr,p=p)

    if GPU:
        kwargs = {'gpus':-1,'precision':16}
    else:
        kwargs = {'gpus':None,'precision':32}

    trainer = pl.Trainer(
    logger=wandb_logger,    # W&B integration
    max_epochs=epochs,           # number of epochs
    deterministic=True,     # keep it deterministic
    **kwargs
    )

    trainer.fit(model,dm)


