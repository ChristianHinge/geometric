import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.models.model import GCN
from src.data.make_dataset import get_mutag_data, get_dataloader

wandb.login(key='9bb7c1fffbc3ead322b0944221455e8deaaa7111')
wandb_logger = WandbLogger(project="Geometric", entity='classy_geometric')


dataset = get_mutag_data(train=True, cleaned=False)
trainloader = get_dataloader(dataset)

data = next(iter(trainloader))

model = GCN(dataset.num_node_features, dataset.num_classes,lr=1e-3,p=0.5)
model.train()
out = model(data.x, data.edge_index, data.batch)


#trainer = pl.Trainer(
#    logger=wandb_logger,    # W&B integration
#    # log_every_n_steps=50,   # set the logging frequency
    # gpus=-1,                # use all GPUs
#    max_epochs=2,           # number of epochs
#    deterministic=True,     # keep it deterministic
#    )
