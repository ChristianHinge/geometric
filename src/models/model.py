from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(pl.LightningModule):
    def __init__(
        self,
        input_num_features: int = 7,
        num_classes: int = 2,
        hidden_channels=None,
        lr: float = 1e-3,
        p: float = 0.5,
        seed: int = 12345,
    ):

        super(GCN, self).__init__()
        if hidden_channels is None:
            hidden_channels = [64, 64, 64]
        torch.manual_seed(seed)

        self.p = p
        self.lr = lr

        # initialize module here
        acc = pl.metrics.Accuracy()
        # use .clone so that each metric can maintain its own state
        self.train_acc = acc.clone()
        # assign all metrics as attributes of module so they are detected as children
        self.val_acc = acc.clone()
        self.test_acc = acc.clone()

        self.conv_layers = ModuleList()
        current_dim = input_num_features
        for h_channel in hidden_channels:
            self.conv_layers.append(GCNConv(current_dim, h_channel))
            current_dim = h_channel

        self.linear = Linear(current_dim, num_classes)

        self.save_hyperparameters()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for layer in self.conv_layers:
            x = F.relu(layer(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)

        return x

    def shared_step(self, batch, batch_idx):
        outputs = self.forward(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, batch.y)

        return loss, probs, batch.y

    def training_step(self, batch, batch_idx):
        loss, probs, targets = self.shared_step(batch, batch_idx)
        self.train_acc(probs, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "accuracy": self.train_acc}

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self.shared_step(batch, batch_idx)

        self.val_acc(probs, targets)

        self.log("validation/loss", loss, on_step=False, on_epoch=True)
        self.log("validation/accuracy", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, probs, targets = self.shared_step(batch, batch_idx)
        self.test_acc(probs, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: Optional[int] = None):
        outputs = self.forward(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(outputs, dim=1)
        return probs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
