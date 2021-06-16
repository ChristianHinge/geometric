import torch
from torch.nn import Linear, ModuleDict
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM

from src.data.make_dataset import get_mutag_data, get_dataloader

class GCN(pl.LightningModule):
    def __init__(
        self,
        input_num_features: int,
        num_classes: int,
        hidden_channels: dict = {"conv1": 64, "conv2": 64, "conv3": 64},
        seed: int = 12345,
        p: float = 0.5,
    ):
        super(GCN, self).__init__()
        torch.manual_seed(seed)

        self.p = p

        # initialize module here    
        acc = pl.metrics.Accuracy()
        # use .clone so that each metric can maintain its own state
        self.train_acc = acc.clone()
        # assign all metrics as attributes of module so they are detected as children
        self.val_acc = acc.clone()
        self.test_acc = acc.clone()

        self.conv_layers = ModuleDict()
        current_dim = input_num_features
        for conv_name, hchannel in hidden_channels.items():
            self.conv_layers[conv_name] = GCNConv(current_dim, hchannel)
            current_dim = hchannel

        self.linear = Linear(current_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for layer in self.conv_layers.values():
            x = F.relu(layer(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)

        return x

    def shared_step(self,batch,batch_idx):

        outputs = self.forward(batch.x, batch.edge_index,batch.batch)
        preds = F.softmax(outputs)
        loss = F.cross_entropy(preds, batch.y)

        return loss, preds, batch.y

    def training_step(self, batch, batch_idx):

        loss, preds, targets = self.shared_step(batch,batch_idx)
        self.log({'train/CE loss': loss})
        return {'loss':loss, 'preds':preds, 'targets':targets}

    def training_step_end(self,outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.train_acc(outs['preds'],outs['targets'])
        self.log({'train/acc_step': self.train_acc})

    def training_epoch_end(self,outs):
        # additional log mean accuracy at the end of the epoch
        self.log("train/acc_epoch", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):

        loss, preds, targets = self.shared_step(batch,batch_idx)
        self.log({'validation/CE loss': loss})

        return {'loss':loss, 'preds':preds, 'targets':targets}

    def validation_step_end(self,outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.val_acc(outs['preds'],outs['targets'])
        self.log({'validation/acc_step': self.val_acc})

    def validation_epoch_end(self,outs):
        # additional log mean accuracy at the end of the epoch
        self.log("validation/acc_epoch", self.val_acc.compute())

    def test_step(self, batch, batch_idx):

        loss, preds, targets = self.shared_step(batch,batch_idx)
        self.log({'test/CE loss': loss})

        return {'loss':loss, 'preds':preds, 'targets':targets}

    def test_step_end(self,outs):
        # log accuracy on each step_end, for compatibility with data-parallel
        self.test_acc(outs['preds'],outs['targets'])
        self.log({'test/acc_step': self.test_acc})

    def test_epoch_end(self,batch,batch_idx):
        # additional log mean accuracy at the end of the epoch
        self.log("test/acc_epoch", self.test_acc.compute())

    


if __name__ == "__main__":
    dataset = get_mutag_data(train=True, cleaned=False)
    trainloader = get_dataloader(dataset)

    data = next(iter(trainloader))

    model = GCN(dataset.num_node_features, dataset.num_classes)
    model.train()
    out = model(data.x, data.edge_index, data.batch)
