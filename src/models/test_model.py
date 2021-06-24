import os

import pytorch_lightning as pl
from azureml.core import Workspace
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data.datamodule import MUTANGDataModule
from src.models.model import GCN
from src.settings.paths import CHECKPOINT_PATH, CLOUD_PATH


def eval(seed: int, project: str, filename: str, entity: str, azure_stored: bool):
    wandb.login(key=os.getenv("WANDB_KEY"))

    run_name = filename.split("_")[0]
    run_id = filename.replace(".", "_").split("_")[1]
    wandlogger = WandbLogger(
        name=run_name, project=project, resume=True, version=run_id, entity=entity
    )

    if azure_stored:
        ws = Workspace.from_config(os.path.join(CLOUD_PATH, "config.json"))
        model = ws.models[filename]

        model.download(CHECKPOINT_PATH, exist_ok=True)
    test_path = os.path.join(CHECKPOINT_PATH, filename)

    model = GCN.load_from_checkpoint(test_path)

    dm = MUTANGDataModule(batch_size=64, seed=seed)

    trainer = pl.Trainer(logger=wandlogger)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":

    eval(
        seed=42,
        project="Geometric",
        filename="22-10-12_190dze4r.ckpt",
        entity="classy_geometric",
        azure_stored=True,
    )
