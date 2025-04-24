import torch

from torch import Tensor



def quantize_int8(tensor: torch.Tensor, dim: int = -1):
    min_val, max_val = tensor.amin(dim, keepdim=True), tensor.amax(dim, keepdim=True)

    # Compute scale factor to map values to INT8 range (-128 to 127)
    scale = (max_val - min_val) / 255  # 255 because int8 range is -128 to 127
    scale.clamp_(min=1e-8)  # Avoid division by zero

    # Perform quantization
    quantized = ((tensor - min_val) / scale - 128).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale, min_val

def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor, min_val: torch.Tensor):
    return (quantized.float() + 128) * scale + min_val


class KVCache:
    def __init__(self, shape, max_seq_length, idx: int | None = None, device=None, dtype=None, int8: bool = False):
        self.idx = idx
        self.int8 = int8
        kv_dtype = torch.int8 if int8 else dtype
        self.key: Tensor = torch.zeros(shape, device=device, dtype=kv_dtype)
        self.value: Tensor = torch.zeros(shape, device=device, dtype=kv_dtype)
        
        if int8:
            self.key_scale = torch.zeros(shape, device=device, dtype=torch.float32).sum(dim=-1, keepdim=True)
            self.key_min = torch.zeros(shape, device=device, dtype=torch.float32).sum(dim=-1, keepdim=True)
            self.value_scale = torch.zeros(shape, device=device, dtype=torch.float32).sum(dim=-1, keepdim=True)
            self.value_min = torch.zeros(shape, device=device, dtype=torch.float32).sum(dim=-1, keepdim=True)
        self.max_seq_length = max_seq_length

    def forward(self, keys: Tensor, values: Tensor, start_pos: Tensor) -> tuple[Tensor, Tensor]:
        bsz, T, _, _ = keys.shape
        if self.int8:
            keys, key_scale, key_min = quantize_int8(keys)
            values, value_scale, value_min = quantize_int8(values)
            self.key_scale[:bsz, start_pos : start_pos + T] = key_scale
            self.key_min[:bsz, start_pos : start_pos + T] = key_min
            self.value_scale[:bsz, start_pos : start_pos + T] = value_scale
            self.value_min[:bsz, start_pos : start_pos + T] = value_min
            self.key[:bsz, start_pos : start_pos + T] = keys 
            self.value[:bsz, start_pos : start_pos + T] = values 
        else:
            self.key[:bsz, start_pos : start_pos + T] = keys 
            self.value[:bsz, start_pos : start_pos + T] = values 
        keys = self.key[:bsz, : start_pos + T]
        values = self.value[:bsz, : start_pos + T]
        return keys, values
