# Llama

Implementation: [llama.py](../../../ohara/models/llama.py).

The model uses RMSNorm, rotary positions, causal attention, and SwiGLU. Set
`num_key_value_heads` for grouped-query attention; `0` uses one KV head per query
head. Feed-forward layers are dense by default. `moe_num_experts` and
`moe_layer_interval` select MoE layers.

`forward(tokens)` returns vocabulary logits. `build_kv_cache()` supports
incremental decoding; `save_pretrained()` writes config and safetensors weights.
See [pretraining](../../pretrain.md) for the training entrypoint.

Papers: [Llama 1](https://arxiv.org/abs/2302.13971),
[Llama 2](https://arxiv.org/abs/2307.09288).
