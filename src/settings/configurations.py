from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig, ListConfig


@dataclass
class Training:
    epochs: int
    lr: float
    batch_size: int
    dropout_rate: float
    layers: List[int]
    seed: int
    gpu: bool
    azure: bool


@dataclass
class Evaluation:
    project: str
    entity: str
    azure_stored: bool


def dict_(cfg):
    d = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            d[key] = dict(dict_(value))
        elif isinstance(value, ListConfig):
            d[key] = list_(value)
        else:
            d[key] = value
    return d


def list_(cfg):
    items = []
    for item in cfg:
        if isinstance(item, ListConfig):
            items.append(list(list_(item)))
        else:
            items.append(item)

    return items
