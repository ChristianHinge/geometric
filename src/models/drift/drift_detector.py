import matplotlib.pyplot as plt
import sklearn
import torch
import torch.utils.data
import torch_geometric.datasets
import torchdrift
from torch_geometric import transforms

from src.data.make_dataset import get_dataloader
from src.models.drift.transformations import GaussianBlur
from src.models.model import GCN
from src.settings import NOTCLEANED_DATA_PATH


class DriftDetector:
    def __init__(self):
        self.detector = torchdrift.detectors.KernelMMDDriftDetector()

    @staticmethod
    def prepare_data_set():
        transform = transforms.Compose([
            GaussianBlur(2),
        ])

        dataset = torch_geometric.datasets.TUDataset(
            root=NOTCLEANED_DATA_PATH,
            name="MUTAG",
            transform=transform
        )

        mask = []
        index = 0
        for data in dataset:
            if data.x.shape == torch.Size([17, 7]):
                mask.append(index)
            index += 1

        dataset = torch.utils.data.Subset(dataset, mask)

        return dataset

    def detect(self, dataloader, model):
        data = next(iter(dataloader))
        torchdrift.utils.fit(dataloader, model, self.detector)

        features = model(data)
        score = self.detector(features)
        p_val = self.detector.compute_p_value(features)

        mapper = sklearn.manifold.Isomap(n_components=2)
        base_embedded = mapper.fit_transform(self.detector.base_outputs)
        features_embedded = mapper.transform(features)
        plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
        plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
        plt.title(f'score {score:.2f} p-value {p_val:.2f}')


if __name__ == '__main__':
    d = DriftDetector()
    dataset = d.prepare_data_set()
    dataloader = get_dataloader(dataset, 10)
    model = GCN(7, 2)
    d.detect(dataloader, model)
