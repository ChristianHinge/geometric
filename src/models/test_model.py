import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.datamodule import MUTANGDataModule
from src.settings import CHECKPOINT_PATH
import os



def eval(project: str, filename: str, entity: str):
    wandb.login(key=os.getenv("WANDB_KEY"))

    test_path = os.path.join(CHECKPOINT_PATH,filename)

    run_name = filename.split('_')[0]
    run_id = filename.replace('.','_').split('_')[1]

    wandlogger = WandbLogger(name=run_name,project=project,resume=True,version=run_id,entity=entity)

    model = GCN.load_from_checkpoint(test_path)

    dm = MUTANGDataModule(batch_size=64)

    trainer = pl.Trainer(logger=wandlogger)
    trainer.test(model,datamodule=dm)

if __name__=='__main__':

    eval(project='Geometric',filename='17-14:31_3azam77n.ckpt',entity='classy_geometric')









