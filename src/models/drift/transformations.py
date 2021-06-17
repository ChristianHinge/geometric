from abc import abstractmethod, ABC

import torch
import torchdrift
from torch_geometric.data import Data


class Transformation(ABC):

    def __call__(self, x: torch.Tensor):
        self.function(x)

    @abstractmethod
    def function(self, x: torch.Tensor):
        pass


class GaussianBlur(Transformation):
    def __init__(self, severity: int = 1):
        self.severity = severity

    def __call__(self, x: Data):
        return self.function(x)

    def function(self, data: Data):
        x = torchdrift.data.functional.gaussian_noise(data.x, severity=self.severity)
        data.x = x
        return data
