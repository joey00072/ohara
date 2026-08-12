"""A single KV cache implementation shared by every model in this repo.

The cache stores keys and values in ``(batch, seq_len, num_heads, head_dim)``
layout and is written to sequentially: one prefill of the prompt followed by one
position per decode step. Optionally the entries are kept as int8 and
dequantized on read, which trades a little accuracy for ~4x less cache memory.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quantize_int8(tensor: Tensor, dim: int = -1) -> tuple[Tensor, Tensor, Tensor]:
    """Affine-quantize ``tensor`` to int8 along ``dim``.

    Returns the quantized tensor plus the scale and minimum needed to invert it.
    """
    min_val, max_val = tensor.amin(dim, keepdim=True), tensor.amax(dim, keepdim=True)

    # Map the observed range onto the 256 available int8 levels.
    scale = (max_val - min_val) / 255
    scale.clamp_(min=1e-8)  # avoid division by zero on constant rows

    quantized = ((tensor - min_val) / scale - 128).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale, min_val


def dequantize_int8(quantized: Tensor, scale: Tensor, min_val: Tensor) -> Tensor:
    """Invert :func:`quantize_int8`."""
    return (quantized.float() + 128) * scale + min_val


class KVCache:
    """Fixed-size key/value cache for incremental decoding.

    Args:
        shape: Full cache shape, ``(batch, max_seq_length, num_heads, head_dim)``.
        max_seq_length: Longest sequence the cache can hold.
        idx: Optional layer index, useful when debugging a stack of caches.
        device: Device to allocate the cache on.
        dtype: Storage dtype. Ignored when ``int8`` is set.
        int8: Store entries as int8 and dequantize them on read.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        max_seq_length: int,
        idx: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        int8: bool = False,
    ) -> None:
        self.idx = idx
        self.int8 = int8
        self.max_seq_length = max_seq_length
        self.length = 0
        self.batch_size: int | None = None
        self.dtype = dtype

        storage_dtype = torch.int8 if int8 else dtype
        self.key: Tensor = torch.zeros(shape, device=device, dtype=storage_dtype)
        self.value: Tensor = torch.zeros(shape, device=device, dtype=storage_dtype)

        if int8:
            # One scale/min per quantized row, i.e. per (batch, position, head).
            stats_shape = (*shape[:-1], 1)
            self.key_scale = torch.zeros(stats_shape, device=device, dtype=torch.float32)
            self.key_min = torch.zeros(stats_shape, device=device, dtype=torch.float32)
            self.value_scale = torch.zeros(stats_shape, device=device, dtype=torch.float32)
            self.value_min = torch.zeros(stats_shape, device=device, dtype=torch.float32)

    def forward(self, keys: Tensor, values: Tensor, start_pos: int) -> tuple[Tensor, Tensor]:
        """Append ``keys``/``values`` at ``start_pos`` and return the full cache so far."""
        bsz, seq_len, _, _ = keys.shape
        if values.shape != keys.shape:
            raise ValueError("KV cache keys and values must have matching shapes")
        if bsz > self.key.size(0):
            raise ValueError("KV cache batch size is smaller than the input batch")
        if self.batch_size is not None and bsz != self.batch_size:
            raise ValueError("KV cache batch size cannot change during decoding")
        if start_pos != self.length:
            raise ValueError(
                f"KV cache writes must be sequential: expected position {self.length}, "
                f"got {start_pos}"
            )
        if start_pos < 0 or start_pos + seq_len > self.max_seq_length:
            raise ValueError("KV cache position exceeds max_sequence_length")

        end = start_pos + seq_len
        window = slice(start_pos, end)
        if self.int8:
            quantized_keys, key_scale, key_min = quantize_int8(keys)
            quantized_values, value_scale, value_min = quantize_int8(values)
            self.key[:bsz, window] = quantized_keys
            self.value[:bsz, window] = quantized_values
            self.key_scale[:bsz, window] = key_scale
            self.key_min[:bsz, window] = key_min
            self.value_scale[:bsz, window] = value_scale
            self.value_min[:bsz, window] = value_min
        else:
            self.key[:bsz, window] = keys
            self.value[:bsz, window] = values

        self.length = end
        self.batch_size = bsz
        return self._read(bsz, end, keys.dtype)

    def _read(self, bsz: int, end: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        keys = self.key[:bsz, :end]
        values = self.value[:bsz, :end]
        if not self.int8:
            return keys, values
        keys = dequantize_int8(keys, self.key_scale[:bsz, :end], self.key_min[:bsz, :end])
        values = dequantize_int8(values, self.value_scale[:bsz, :end], self.value_min[:bsz, :end])
        return keys.to(dtype), values.to(dtype)
