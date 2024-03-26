import lightning as L
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel
import ohara.models.phi as phi
import torch


class Transformer(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(**args, **kwargs)

    @classmethod
    def load_phi(cls, model_name="microsoft/phi-2"):
        cfg = phi.PhiConfig()
        model = phi.Phi(cfg)
        hf_model = AutoModel.from_pretrained(model_name)
        hf_model
        mask = torch.full((1, 1, cfg.seq_len, cfg.seq_len), float("-inf"))

        mode
