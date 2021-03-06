import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric import datasets
from torch_geometric.data import DataLoader

from src.settings.paths import CLEANED_DATA_PATH, NOT_CLEANED_DATA_PATH


class MUTANGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        cleaned: bool = False,
        split=None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        if split is None:
            split = [0.6, 0.2, 0.2]
        self.split = split
        self.batch_size = batch_size
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.cleaned = cleaned
        self.num_workers = num_workers
        self.seed = seed

        if sum(self.split) != 1:
            raise ValueError("Expected split list to sum to 1")

    def prepare_data(self):
        return datasets.TUDataset(
            root=CLEANED_DATA_PATH if self.cleaned else NOT_CLEANED_DATA_PATH,
            name="MUTAG",
            cleaned=self.cleaned,
            pre_transform=None,
        )

    def setup(self, stage: str = None):
        torch.manual_seed(self.seed)
        self.full_set = datasets.TUDataset(
            root=CLEANED_DATA_PATH if self.cleaned else NOT_CLEANED_DATA_PATH,
            name="MUTAG",
            cleaned=self.cleaned,
            pre_transform=None,
        )

        self.train_set, self.val_set, self.test_set = random_split(
            self.full_set,
            [
                round(len(self.full_set) * self.split[0]),
                round(len(self.full_set) * self.split[1]),
                len(self.full_set)
                - round(len(self.full_set) * self.split[0])
                - round(len(self.full_set) * self.split[1]),
            ],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
