import pytorch_lightning as pl
import wandb
import os 
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.datamodule import MUTANGDataModule
#from dotenv import load_dotenv, find_dotenv

def hyp_opt_iter():

    #dotenv_path = find_dotenv()
    #load_dotenv(dotenv_path)

    #setup wandb logger
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
