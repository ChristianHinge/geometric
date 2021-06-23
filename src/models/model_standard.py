import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from src.data.make_dataset import get_mutag_data, get_dataloader


class GCN(torch.nn.Module):
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


if __name__ == "__main__":
    dataset = get_mutag_data(train=True, cleaned=False)
    trainloader = get_dataloader(dataset)

    data = next(iter(trainloader))

    model = GCN(dataset.num_node_features, dataset.num_classes)
    model.train()
    out = model(data.x, data.edge_index, data.batch)
