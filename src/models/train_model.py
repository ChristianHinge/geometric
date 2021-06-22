import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.model import GCN
from src.data.datamodule import MUTANGDataModule
from src.settings import CHECKPOINT_PATH
import os
from dotenv import load_dotenv, find_dotenv


def train(
    lr: float,
    epochs: float,
    batch_size: int,
    layers: int,
    GPU: bool,
    p: float,
    name: str,
):

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Initialise wandb logger
    wandb.login(key=os.getenv("WANDB_KEY"))
    wandb_logger = WandbLogger(
        name=name, project="Geometric", entity="classy_geometric"
    )

    dm = MUTANGDataModule(batch_size=batch_size)
    dataset = dm.prepare_data()

    model = GCN(
        dataset.num_node_features,
        dataset.num_classes,
        hidden_channels=layers,
        lr=lr,
        p=p,
    )

    filename = f"{name}"
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/accuracy",
        dirpath=CHECKPOINT_PATH,
        filename=filename,
        save_top_k=1,
        mode="max",
    )

    if GPU:
        kwargs = {"gpus": -1, "precision": 16}
    else:
        kwargs = {"gpus": None, "precision": 32}

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        max_epochs=epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        # default_root_dir=os.path.join(CHECKPOINT_PATH, name + ".") ** kwargs,
        callbacks=checkpoint_callback,
    )

    trainer.fit(model, dm)

    # Append wandb run ID to name in order for continue test logging if needed
    best_path = checkpoint_callback.best_model_path
    version = wandb_logger.version
    new_path_name = os.path.join(CHECKPOINT_PATH, filename + "_" + version + ".ckpt")

    os.rename(best_path, new_path_name)
