import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.make_dataset import get_mutag_data, get_dataloader
from src.data.datamodule import MUTANGDataModule

wandb.login(key='9bb7c1fffbc3ead322b0944221455e8deaaa7111')
wandb_logger = WandbLogger(project="Geometric", entity='classy_geometric')


traindataset = get_mutag_data(train=True, cleaned=False)
trainloader = get_dataloader(traindataset)
testdataset = get_mutag_data(train=False, cleaned=False)
testloader = get_dataloader(testdataset)

dm = MUTANGDataModule(batch_size=64)
dataset = dm.prepare_data()

model = GCN(dataset.num_node_features, dataset.num_classes,lr=0.01,p=0.5)

trainer = pl.Trainer(
   logger=wandb_logger,    # W&B integration
   # log_every_n_steps=50,   # set the logging frequency
   # gpus=-1,                # use all GPUs
   max_epochs=20,           # number of epochs
   deterministic=False,     # keep it deterministic
   )

trainer.fit(model,dm)


