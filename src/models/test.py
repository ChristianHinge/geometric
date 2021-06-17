import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.datamodule import MUTANGDataModule
from src.settings import CHECKPOINT_PATH
import os


wandb.login(key=os.getenv("WANDB_KEY"))
wandlogger = WandbLogger(name='debug3',project='Geometric',resume=True,version='2k4tvipf')

test_path = os.path.join(CHECKPOINT_PATH,"debug2.ckpt")
model = GCN.load_from_checkpoint(test_path)

dm = MUTANGDataModule(batch_size=64)

trainer = pl.Trainer(logger=wandlogger)
trainer.test(model,datamodule=dm)










