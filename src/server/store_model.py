import os

import pytorch_lightning as pl

from src.data.datamodule import MUTANGDataModule
from src.models.model import GCN
from src.models.model_saver import ModelSaver
from src.settings.paths import MODEL_STORE_PATH


def store_model():
    saver = ModelSaver()
    data_module = MUTANGDataModule(batch_size=54)

    gcn = GCN()

    kwargs = {'gpus': None, 'precision': 32}

    trainer = pl.Trainer(
        max_epochs=10,
        deterministic=True,
        **kwargs
    )

    trainer.fit(gcn, data_module)
    saver.save_model(gcn, os.path.join(MODEL_STORE_PATH, 'geometric.pt'))


if __name__ == '__main__':
    store_model()
