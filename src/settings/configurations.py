from dataclasses import dataclass
from typing import List


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
