from ..ohara.models.llama import LLAMA,Config

cfg = Config()
model = LLAMA(cfg)

print(model)