# import logging

# from pathlib import Path

# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

import torch
from torch_geometric import datasets
from torch_geometric import data

from src.settings import CLEANED_DATA_PATH, NOTCLEANED_DATA_PATH


def get_mutag_data(
    train: bool = False, cleaned: bool = False, test_size: int = 0.16, seed: int = 12345
):

    dataset = datasets.TUDataset(
        root=CLEANED_DATA_PATH if cleaned else NOTCLEANED_DATA_PATH,
        name="MUTAG",
        cleaned=cleaned,
    )

    """ Get MUTAG dataset from TU and saves the raw files in (../raw) and preprocessed
        in (../preprocessed). If cleaned = True, then the data is saved in a parent folder
        (../cleaned) and if cleaned = False, then it is saved in a parent folder (/../not-cleaned).

        Returns train dataset if train = True and test set otherwise.  
    """

    torch.manual_seed(seed)
    dataset = dataset.shuffle()

    split_idx = int(test_size * len(dataset))

    if train:
        dataset = dataset[split_idx:]
    else:
        dataset = dataset[:split_idx]

    return dataset


def get_dataloader(dataset, batch_size=64, shuffle=True):

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


dataset = get_mutag_data(cleaned=True)

get_dataloader(dataset)


def main():
    """Downloads cleaned and uncleaned MUTAG dataset."""
    get_mutag_data(cleaned=True)
    get_mutag_data(cleaned=False)


if __name__ == "__main__":

    main()
