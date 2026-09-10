import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ohara.models.llama import Config, Llama
from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.trainer import Trainer


def small_model(*, weight_tying: bool) -> Llama:
    return Llama(
        Config(
            vocab_size=23,
            max_sequence_length=8,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_hidden_layers=1,
            dropout=0.0,
            weight_tying=weight_tying,
        )
    )


@pytest.mark.parametrize("weight_tying", [False, True])
def test_chunked_loss_matches_full_loss_and_gradients(weight_tying):
    torch.manual_seed(17)
    full = small_model(weight_tying=weight_tying)
    chunked = copy.deepcopy(full)
    inputs = torch.randint(0, 23, (2, 7))
    targets = torch.randint(0, 23, (2, 7))
    targets[0, 1:4] = -1
    targets[1, -1] = -1

    full_loss = F.cross_entropy(
        full(inputs).float().reshape(-1, 23),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="sum",
    )
    chunked_loss = chunked(
        inputs, targets=targets, loss_chunk_size=5, ignore_index=-1
    )
    torch.testing.assert_close(chunked_loss, full_loss)

    full_loss.backward()
    chunked_loss.backward()
    full_grads = dict(full.named_parameters())
    chunked_grads = dict(chunked.named_parameters())
    assert full_grads.keys() == chunked_grads.keys()
    for name in full_grads:
        torch.testing.assert_close(
            chunked_grads[name].grad,
            full_grads[name].grad,
            atol=2e-5,
            rtol=2e-5,
        )
    if weight_tying:
        assert chunked.token_emb.weight is chunked.vocab_proj.weight


def test_chunked_loss_recomputes_bounded_projection_chunks():
    torch.manual_seed(3)
    model = small_model(weight_tying=False)
    inputs = torch.randint(0, 23, (2, 7))
    targets = torch.randint(0, 23, (2, 7))
    projected_shapes = []
    saved_shapes = []

    hook = model.vocab_proj.register_forward_hook(
        lambda _module, _inputs, output: projected_shapes.append(tuple(output.shape))
    )

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = model(inputs, targets=targets, loss_chunk_size=5)
            loss.backward()
    finally:
        hook.remove()

    chunks = math.ceil(inputs.numel() / 5)
    assert len(projected_shapes) >= chunks * 2
    assert all(shape[0] <= 5 and shape[-1] == 23 for shape in projected_shapes)
    assert not any(len(shape) == 2 and shape[-1] == 23 and shape[0] <= 5 for shape in saved_shapes)


def test_chunked_loss_trains_head_when_hidden_states_are_frozen():
    model = small_model(weight_tying=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.vocab_proj.weight.requires_grad_(True)
    inputs = torch.randint(0, 23, (2, 7))
    targets = torch.randint(0, 23, (2, 7))

    model(inputs, targets=targets, loss_chunk_size=5).backward()

    assert model.vocab_proj.weight.grad is not None
    assert torch.count_nonzero(model.vocab_proj.weight.grad) > 0


def test_chunked_loss_details_match_full_logits_and_all_ignored_mask():
    model = small_model(weight_tying=True).eval()
    inputs = torch.randint(0, 23, (2, 7))
    targets = torch.randint(0, 23, (2, 7))
    targets[0, :3] = -1
    with torch.no_grad():
        logits = model(inputs).float().reshape(-1, 23)
        expected = F.cross_entropy(
            logits, targets.reshape(-1), ignore_index=-1, reduction="none"
        )
        loss, token_losses, predictions = model(
            inputs,
            targets=targets,
            loss_chunk_size=5,
            ignore_index=-1,
            return_loss_details=True,
        )
    torch.testing.assert_close(token_losses, expected)
    torch.testing.assert_close(loss, expected.sum())
    assert torch.equal(predictions, logits.argmax(dim=-1))

    ignored = torch.full_like(targets, -1)
    with torch.no_grad():
        loss, token_losses, predictions = model(
            inputs,
            targets=ignored,
            loss_chunk_size=5,
            ignore_index=-1,
            return_loss_details=True,
        )
    assert loss.item() == 0
    assert torch.count_nonzero(token_losses) == 0
    assert predictions.shape == (inputs.numel(),)


def test_chunked_loss_validates_before_stateful_decoder_forward():
    model = small_model(weight_tying=False)
    layer_calls = []
    hook = model.layers[0].register_forward_hook(
        lambda *_args: layer_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="targets must match"):
            model(
                torch.ones(2, 7, dtype=torch.long),
                targets=torch.ones(2, 6, dtype=torch.long),
                loss_chunk_size=5,
            )
    finally:
        hook.remove()
    assert layer_calls == []


def make_trainer(model, inputs, targets, *, loss_chunk_size, micro_batch=1):
    engine = OharaEngine(
        EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32))
    )
    model = engine.prepare(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    optimizer = engine.prepare_optimizers(optimizer)[0]
    loader = engine.prepare_dataloaders(
        DataLoader(TensorDataset(inputs, targets), batch_size=2)
    )
    trainer = Trainer(
        engine=engine,
        model=model,
        optimizer=optimizer,
        train_dataloader=loader,
        val_dataloader=loader,
        get_lr=lambda _: 0.02,
        micro_batch=micro_batch,
        max_iters=1,
        eval_iters=0,
        save_ckpt_iters=0,
        ignore_index=-1,
        print_every=100,
        eval_val_batches=2,
        token_bytes=torch.tensor([0] + [1] * 22),
        loss_chunk_size=loss_chunk_size,
    )
    return trainer


def test_trainer_chunked_eval_matches_loss_accuracy_and_bpb():
    torch.manual_seed(29)
    full_model = small_model(weight_tying=True)
    chunked_model = copy.deepcopy(full_model)
    inputs = torch.randint(1, 23, (4, 7))
    targets = torch.randint(1, 23, (4, 7))
    targets[0, :2] = -1
    targets[2, -1] = -1
    full = make_trainer(full_model, inputs, targets, loss_chunk_size=0)
    chunked = make_trainer(chunked_model, inputs, targets, loss_chunk_size=5)
    try:
        expected = full.evaluate(full.val_dataloader, 2)
        actual = chunked.evaluate(chunked.val_dataloader, 2)
    finally:
        full.close()
        chunked.close()
    for name in ("loss", "ppl", "bits_per_token", "accuracy", "bpb", "tokens", "bytes"):
        assert actual[name] == pytest.approx(expected[name], rel=1e-6, abs=1e-7)


def test_trainer_chunked_microbatch_update_matches_full_logits_with_masks():
    torch.manual_seed(41)
    full_model = small_model(weight_tying=True)
    chunked_model = copy.deepcopy(full_model)
    inputs = torch.randint(0, 23, (4, 7))
    targets = torch.randint(0, 23, (4, 7))
    targets[0, 1:5] = -1
    targets[1, -2:] = -1
    targets[3, 0] = -1
    full = make_trainer(full_model, inputs, targets, loss_chunk_size=0, micro_batch=2)
    chunked = make_trainer(chunked_model, inputs, targets, loss_chunk_size=5, micro_batch=2)
    try:
        full.train()
        chunked.train()
    finally:
        full.close()
        chunked.close()
    for full_parameter, chunked_parameter in zip(
        full_model.parameters(), chunked_model.parameters(), strict=True
    ):
        torch.testing.assert_close(chunked_parameter, full_parameter, atol=2e-6, rtol=2e-5)
