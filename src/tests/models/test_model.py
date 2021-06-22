import pytest
import torch
from torch_geometric.data import Batch

from src.models.model import GCN


@pytest.fixture
def model():
    return GCN(2, 2)


class TestGCN:

    def test_model_structure(self, model):
        assert model.conv_layers.conv1.in_channels == 2
        assert model.conv_layers.conv1.out_channels == 64
        assert model.conv_layers.conv2.in_channels == 64
        assert model.conv_layers.conv2.out_channels == 64
        assert model.conv_layers.conv3.in_channels == 64
        assert model.conv_layers.conv3.out_channels == 64
        assert model.linear.in_features == 64
        assert model.linear.out_features == 2

    def test_forward_correct(self, model):
        x = torch.zeros([1, 2])
        edge_index = torch.zeros([2, 2], dtype=torch.long)
        batch = torch.zeros([1, 1], dtype=torch.int64)
        data = Batch(x=x, edge_index=edge_index, batch=batch)

        output = model.forward(data.x, data.edge_index, data.batch)

        assert output.shape == torch.Size([1, 2])

    def test_forward_empty(self, model):
        data = Batch()
        with pytest.raises(TypeError):
            model.forward(data.x, data.edge_index, data.batch)

    def test_forward_wrong_shape(self, model):
        x = torch.zeros([1, 2, 3])
        edge_index = torch.zeros([2], dtype=torch.long)
        batch = torch.zeros([1, 1, 1, 1], dtype=torch.int64)
        data = Batch(x=x, edge_index=edge_index, batch=batch)

        with pytest.raises(IndexError):
            model.forward(data.x, data.edge_index, data.batch)


    def test_shared_step_probs_and_y(self, model):
        x = torch.zeros([1, 2])
        edge_index = torch.zeros([2, 2], dtype=torch.long)
        batch = torch.zeros([1, 1], dtype=torch.int64)
        data = Batch(x=x, edge_index=edge_index, batch=batch, y=torch.zeros([1],dtype=torch.long))
        batch_idx = 0
        loss, probs, y = model.shared_step(data,batch_idx)

        assert 0.999 <= torch.sum(probs).item() <= 1.0001
        assert y.shape[0] == 1
        #assert y in [0,1]
        #test bigger batch size

