import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from ohara.models.llama import Config, Llama
from ohara.modules.moe import MoE, apply_qb_update
from ohara.modules.moe_grouped import GroupedMoE
from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.sft import ConversationDataset
from ohara.trainer import Trainer


def make_moe(kind):
    kwargs = dict(dim=2, hidden_dim=4, num_experts=2, num_experts_per_tok=1)
    if kind == "grouped":
        return GroupedMoE(**kwargs, num_shared_experts=0)
    return MoE(**kwargs)


SCORES = torch.tensor([[10., 0.], [9., 0.], [0., 1.], [0., 2.]])


@pytest.mark.parametrize("kind", ["loop", "grouped"])
def test_quantile_is_invariant_to_uneven_microbatches(kind):
    biases = []
    for chunks in ([4], [2, 2], [1, 3], [1, 1, 1, 1]):
        moe = make_moe(kind)
        for scores in SCORES.split(chunks):
            moe._accumulate_qb(scores, scores.min(dim=-1, keepdim=True).values)
        moe.apply_qb_update()
        biases.append(moe.router_bias.clone())
        assert moe.qb_samples.numel() == 0
    for bias in biases:
        torch.testing.assert_close(bias, torch.tensor([-4., 4.]))


@pytest.mark.parametrize("kind", ["loop", "grouped"])
def test_padding_does_not_change_bias_or_load(kind):
    torch.manual_seed(11)
    moe = make_moe(kind)
    padded = copy.deepcopy(moe)
    real = torch.randn(1, 3, 2)
    moe(real)
    padded(torch.cat((real, torch.full((1, 7, 2), 1000.)), dim=1),
           padding_mask=torch.tensor([[False] * 3 + [True] * 7]))
    torch.testing.assert_close(moe.expert_counts, padded.expert_counts)
    moe.apply_qb_update()
    padded.apply_qb_update()
    torch.testing.assert_close(moe.router_bias, padded.router_bias)
    before = padded.router_bias.clone()
    padded(real, padding_mask=torch.ones(1, 3, dtype=torch.bool))
    padded.apply_qb_update()
    torch.testing.assert_close(padded.router_bias, before)


def _distributed_quantile(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        for kind in ("loop", "grouped"):
            for lengths in ((1, 3), (0, 4), (0, 0)):
                moe = make_moe(kind)
                start = 0 if rank == 0 else lengths[0]
                scores = SCORES[start:start + lengths[rank]]
                moe._accumulate_qb(scores, scores.min(dim=-1, keepdim=True).values)
                moe.apply_qb_update()
                expected = torch.zeros(2) if sum(lengths) == 0 else torch.tensor([-4., 4.])
                torch.testing.assert_close(moe.router_bias, expected)
                assert moe.qb_samples.numel() == 0
    finally:
        dist.destroy_process_group()


def test_quantile_across_unequal_and_empty_ranks(tmp_path):
    mp.spawn(_distributed_quantile, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)


class RenderedDataset(ConversationDataset):
    def _rendered(self, epoch):
        # EOS is also PAD. The prompt is real input despite its ignored target.
        yield [1, 0, 2, 0], [0, 0, 1, 1]


def test_dataset_padding_mask_uses_lengths_not_loss_mask_or_token_ids():
    dataset = RenderedDataset([{}], SimpleNamespace(pad_token_id=0), max_length=7,
                              infinite=False, return_padding_mask=True)
    inputs, targets, padding = next(iter(dataset))
    assert inputs.tolist() == [1, 0, 2, 0, 0, 0, 0]
    assert targets[0] == -1
    assert padding.tolist() == [False, False, False, False, True, True, True]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_trainer_passes_padding_through_llama(grouped, chunk_size):
    engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
    engine.launch()
    model = Llama(Config(vocab_size=8, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2, dropout=0,
                        moe_num_experts=2, moe_experts_per_tok=1, moe_grouped=grouped))
    model = engine.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[1, 2, 3, 0, 0]])
    targets = torch.tensor([[-1, 3, 4, -1, -1]])
    padding = torch.tensor([[False, False, False, True, True]])
    loader = DataLoader(TensorDataset(inputs, targets, padding), batch_size=1)
    train_loader, val_loader = engine.prepare_dataloaders(loader, loader)
    trainer = Trainer(engine, model, optimizer, train_loader, val_loader,
                      get_lr=lambda _: 1e-3, micro_batch=1, max_iters=1,
                      eval_iters=1, save_ckpt_iters=0, eval_val_batches=1,
                      loss_chunk_size=chunk_size, apply_router_balancing=apply_qb_update)
    trainer.train()
    assert model.layers[0].ff.expert_counts.sum() == 3
    assert model.layers[0].ff.qb_samples.numel() == 0
