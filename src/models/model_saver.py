import os
from datetime import datetime
from typing import Union

import pytorch_lightning as pl
import torch

from src.settings.paths import MODEL_STORE_PATH, MODELS_PATH


class ModelSaver:
    def __init__(self):
        pass

    @property
    def model_name(self):
        now = datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S")
        return f"{now}.pth"

    def save_model(
        self, model: Union[torch.nn.Module, pl.LightningModule], model_path: str = ""
    ):
        torch.save(
            model.state_dict(),
            model_path if model_path else os.path.join(MODELS_PATH, self.model_name),
        )

    @staticmethod
    def script_model(model: Union[torch.nn.Module, pl.LightningModule]):
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(MODEL_STORE_PATH, "geometric.pt"))
