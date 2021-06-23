import pytest
import torch

from src.data.datamodule import MUTANGDataModule

dm1 = MUTANGDataModule()
dm1.setup()
dm2 = MUTANGDataModule()
dm2.setup()


def test_split_list():
    with pytest.raises(ValueError):
        dm = MUTANGDataModule(split=[0.8, 0.3, 0.2])


def test_data_split():
    assert len(dm1.full_set) == len(dm1.train_set) + len(dm1.val_set) + len(dm1.test_set)


def test_dataloader_test_set():
    tl1 = dm1.test_dataloader()
    tl2 = dm2.test_dataloader()
    assert len(tl1) == len(tl2)
    for b1, b2 in zip(tl1, tl2):
        assert torch.all(torch.eq(b1.batch, b2.batch))
        assert torch.all(torch.eq(b1.edge_attr, b2.edge_attr))
        assert torch.all(torch.eq(b1.edge_index, b2.edge_index))
        assert torch.all(torch.eq(b1.ptr, b2.ptr))
        assert torch.all(torch.eq(b1.x, b2.x))
        assert torch.all(torch.eq(b1.y, b2.y))


def test_test_set():
    assert len(dm1.test_set) == len(dm2.test_set)

    for ii in range(len(dm1.test_set)):
        assert torch.all(torch.eq(dm1.test_set[ii].edge_attr, dm2.test_set[ii].edge_attr))
        assert torch.all(torch.eq(dm1.test_set[ii].edge_index, dm2.test_set[ii].edge_index))
        assert torch.all(torch.eq(dm1.test_set[ii].x, dm2.test_set[ii].x))
        assert torch.all(torch.eq(dm1.test_set[ii].y, dm2.test_set[ii].y))
