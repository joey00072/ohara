import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.attention import select_blocks
from ohara.models.qwen38.distributed import RowShardedEmbedding
from ohara.models.qwen38.ops import delta_reference, sparse_attention_reference


def tiny(**overrides):
    values = dict(hidden_size=32, head_dim=8, rotary_dim=8, linear_key_dim=8,
                  linear_value_dim=8, index_dim=8, ngram_dim=32, residual_rank=4,
                  expert_dim=16, token_budget=8, query_chunk_size=3, backend="torch")
    return Config(**(values | overrides))


def test_official_shape_and_meta_construction():
    cfg = Config.official()
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_experts, cfg.top_k) == (48, 2560, 512, 10)
    with torch.device("meta"):
        model = Qwen38(cfg)
    ngram = model.ngram.embedding.weight.numel()
    backbone = sum(p.numel() for p in model.parameters()) - ngram - sum(p.numel() for p in model.mtp.parameters())
    assert 50e9 < ngram < 52e9
    assert 124e9 < backbone < 127e9


@pytest.mark.parametrize("length", [1, 3, 4, 5, 13])
def test_selection_is_causal_unique_and_keeps_incomplete_tail(length):
    positions = torch.arange(length)
    scores = torch.randn(2, length, length // 4)
    indices, _, _ = select_blocks(scores, positions, 4, 8)
    for batch in range(2):
        for t in range(length):
            picked = indices[batch, t]
            picked = picked[picked >= 0].tolist()
            assert len(picked) == len(set(picked))
            assert all(i <= t for i in picked)
            assert set(range(((t + 1) // 4) * 4, t + 1)) <= set(picked)
            if t < 8:
                assert set(picked) == set(range(t + 1))


def test_sparse_attention_matches_dense_mask_outputs_and_gradients():
    torch.manual_seed(3)
    q = torch.randn(2, 7, 4, 8, requires_grad=True)
    k = torch.randn(2, 7, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices, _, _ = select_blocks(torch.randn(2, 7, 1), torch.arange(7), 4, 4)
    actual = sparse_attention_reference(q, k, v, indices)
    mask = torch.zeros(2, 7, 8, dtype=torch.bool).scatter_(-1, indices.remainder(8), True)[..., :7]
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask[:, None], enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb)


def test_delta_chunked_state_matches_one_pass():
    torch.manual_seed(4)
    q, k, v = [torch.randn(2, 7, 2, 4, requires_grad=True) for _ in range(3)]
    g = -torch.rand(2, 7, 2)
    beta = torch.rand(2, 7, 2)
    expected, expected_state = delta_reference(q, k, v, g, beta)
    first, state = delta_reference(q[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3])
    second, final = delta_reference(q[:, 3:], k[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], state)
    torch.testing.assert_close(torch.cat((first, second), 1), expected)
    torch.testing.assert_close(final, expected_state)
    actual_grads = torch.autograd.grad(first.sum() + second.sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)


def test_causality_ngram_boundaries_and_all_components_train():
    torch.manual_seed(8)
    model = Qwen38(tiny()).eval()
    tokens = torch.randint(1, 256, (2, 13))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(1, 256, (2, 5))
    torch.testing.assert_close(model(tokens).logits[:, :8], model(changed).logits[:, :8])
    one = model.ngram.indices(torch.tensor([[1, 2, 0, 4, 5]]))[:, 3:]
    two = model.ngram.indices(torch.tensor([[9, 7, 0, 4, 5]]))[:, 3:]
    torch.testing.assert_close(one, two)
    model.train()
    result = model(tokens, tokens, loss_chunk_size=3)
    result.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    indexer = model.layers[3].attention.index_qk.weight
    assert indexer.grad.abs().sum() > 0
    assert model.ngram.embedding.weight.grad.abs().sum() > 0


def test_loss_chunking_and_layer_checkpoint_gradients_agree():
    torch.manual_seed(9)
    first = Qwen38(tiny())
    second = copy.deepcopy(first)
    second.config.activation_checkpointing = True
    tokens = torch.randint(0, 256, (1, 7))
    a, b = first(tokens, tokens, loss_chunk_size=2), second(tokens, tokens, loss_chunk_size=100)
    torch.testing.assert_close(a.loss, b.loss)
    a.loss.backward()
    b.loss.backward()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-4)


def test_mtp_teacher_forcing_shifts_and_shared_layer():
    model = Qwen38(tiny(mtp_steps=3))
    tokens = torch.arange(1, 10).unsqueeze(0)
    seen = []
    handle = model.mtp.register_forward_pre_hook(
        lambda module, args: seen.append((args[0].shape[1], args[1].detach().clone()))
    )
    try:
        model(tokens, tokens).loss.backward()
    finally:
        handle.remove()
    assert len(seen) == 3
    for step, (length, embeddings) in enumerate(seen, 1):
        assert length == tokens.size(1) - step
        torch.testing.assert_close(embeddings, model.embedding(tokens[:, step:]))
    assert model.mtp.hidden_proj.weight.grad.abs().sum() > 0


def _sharded_lookup(rank, rendezvous, empty_owner):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous,
                            timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,))
        embedding = RowShardedEmbedding(11, 4, mesh, device="cpu")
        weight = (torch.arange(44).reshape(11, 4).float() / 100).requires_grad_()
        local = embedding.weight.to_local()
        with torch.no_grad():
            local.copy_(weight.detach()[embedding.start:embedding.start + local.size(0)])
        requests = [torch.tensor([[0, 7, 0, 10]]), torch.tensor([[8, 9, 7, 1]])]
        if empty_owner:
            # One rank requests nothing, and every requested row belongs to rank zero.
            requests = [torch.tensor([[0, 0, 1, 1]]), torch.empty((1, 0), dtype=torch.long)]
        actual = embedding(requests[rank])
        torch.testing.assert_close(actual, F.embedding(requests[rank], weight))
        actual.square().sum().backward()
        expected_loss = sum(F.embedding(ids, weight).square().sum() for ids in requests) / 2
        expected_loss.backward()
        torch.testing.assert_close(
            embedding.weight.grad.to_local(), weight.grad[embedding.start:embedding.start + local.size(0)],
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_owner", [False, True])
def test_row_sharded_lookup_outputs_and_global_mean_gradients(tmp_path, empty_owner):
    mp.spawn(_sharded_lookup, args=(f"file://{tmp_path / 'pg'}", empty_owner), nprocs=2)


@pytest.mark.skipif(not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
                    reason="requires CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("length,slots,dim", [(3, 3, 8), (5, 37, 16), (9, 65, 128), (9, 65, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_kernel_forward_backward(length, slots, dim, dtype):
    from ohara.kernels.qwen38_sparse import sparse_attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    q = torch.randn(1, length, 4, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, slots + 1, 2, dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.arange(slots, device=device).expand(1, length, -1).clone()
    indices[:, 0] = -1  # Empty rows have zero outputs and gradients.
    indices[:, 1, :2] = -1
    actual, expected = sparse_attention(q, k, v, indices), sparse_attention_reference(q, k, v, indices)
    tolerance = 0.02 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    a = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FLA")
def test_fla_delta_matches_reference():
    from ohara.models.qwen38.ops import delta

    q, k, v = [torch.randn(1, 65, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3)]
    g = (-torch.rand(1, 65, 2, device="cuda")).requires_grad_()
    beta = torch.rand(1, 65, 2, device="cuda", requires_grad=True)
    actual, expected = delta(q, k, v, g, beta, "cuda"), delta_reference(q, k, v, g, beta)[0]
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    a = torch.autograd.grad(actual.float().square().sum(), (q, k, v, g, beta), retain_graph=True)
    b = torch.autograd.grad(expected.float().square().sum(), (q, k, v, g, beta))
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, atol=0.05, rtol=0.05)
