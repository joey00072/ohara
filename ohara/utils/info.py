from typing import Any
import lightning as L
import torch.nn as nn
from lightning.pytorch.utilities.model_summary import ModelSummary


class Model(L.LightningModule):
    def __init__(self, model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model


def model_summary(model: nn.Module):
    model = Model(model)
    print(ModelSummary(model))
