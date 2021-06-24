import logging
import os

import pytorch_lightning as pl
import torch
import wandb
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import MUTANGDataModule
from src.models.model import GCN
from src.settings.paths import CHECKPOINT_PATH


def train(
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    layers: list,
    gpu: bool,
    dropout_rate: float,
    name: str,
    azure: bool,
):
    log = logging.getLogger(__name__)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Initialise wandb logger
    wandb.login(key=os.getenv("WANDB_KEY"))
    wandb_logger = WandbLogger(
        name=name, project="Geometric", entity="classy_geometric"
    )

    dm = MUTANGDataModule(batch_size=batch_size, seed=seed)
    dataset = dm.prepare_data()

    torch.manual_seed(seed)
    model = GCN(
        dataset.num_node_features,
        dataset.num_classes,
        hidden_channels=layers,
        lr=lr,
        p=dropout_rate,
    )

    filename = f"{name}"
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/accuracy",
        dirpath=CHECKPOINT_PATH,
        filename=filename,
        save_top_k=1,
        mode="max",
    )

    if gpu:
        kwargs = {"gpus": -1, "precision": 16}
    else:
        kwargs = {"gpus": None, "precision": 32}

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        max_epochs=epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        default_root_dir=os.path.join(CHECKPOINT_PATH, name + "."),
        callbacks=checkpoint_callback,
        **kwargs
    )

    trainer.fit(model, dm)

    # Append wandb run ID to name in order for continue test logging if needed
    best_path = checkpoint_callback.best_model_path
    version = wandb_logger.version
    model_name = filename + "_" + version + ".ckpt"
    new_path_name = os.path.join(CHECKPOINT_PATH, model_name)

    os.rename(best_path, new_path_name)

    if azure:
        log.info("-- Registering model in azure workspace --")

        from azureml.core import Run

        run = Run.get_context()

        run.upload_file(
            name=os.path.join("outputs", model_name),
            path_or_stream=os.path.join(CHECKPOINT_PATH, model_name),
        )

        run.register_model(
            model_path=os.path.join("outputs", model_name),
            model_name=model_name,
            tags={"Training context": "Training of GNN model"},
            properties={},
        )
