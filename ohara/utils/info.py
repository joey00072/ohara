from typing import Any
import lightning as L
import torch.nn as nn
from lightning.pytorch.utilities.model_summary import ModelSummary


class Model(L.LightningModule):
    def __init__(self, model, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model


def convert_int_to_shortened_string(num):
    """
    Converts an integer into a shortened format using suffixes like K, M, B, and T for
    thousands, millions, billions, and trillions, respectively, with one decimal place.

    Args:
    - num (int): The number to convert.

    Returns:
    - str: The converted, shortened number as a string.
    """
    if abs(num) < 1000:
        return str(num)
    elif abs(num) < 1000000:
        return f"{num / 1000:.1f}K"
    elif abs(num) < 1000000000:
        return f"{num / 1000000:.1f}M"
    elif abs(num) < 1000000000000:
        return f"{num / 1000000000:.1f}B"
    else:
        return f"{num / 1000000000000:.1f}T"


def model_summary(model: nn.Module, print_summary=False):
    "conver normal model to lightning model and print model summary"
    model = Model(model)
    summary = ModelSummary(model)
    if print_summary:
        print(summary)
    return {
        "summary": summary,
        "total_parameters": convert_int_to_shortened_string(summary.total_parameters),
        "trainable_parameters": convert_int_to_shortened_string(summary.trainable_parameters),
    }
