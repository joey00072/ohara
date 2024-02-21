import lightning as L


class Transformer(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def phi(
        self,
    ):
        return self.model(x)
