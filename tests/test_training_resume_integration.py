import shutil
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ohara.data_resume import restore_input_state
from ohara.models.llama import Config, Llama
from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.tokenbin import TokenBinDataset, write_token_bin
from ohara.trainer import Trainer
from test_tokenbin import FakeTokenizer


def test_real_training_resume_matches_dropout_optimizer_and_cursor(tmp_path):
    corpus = tmp_path / "train.bin"
    write_token_bin(["abcdefg" * 40], FakeTokenizer(), corpus, log=False)
    checkpoint = tmp_path / "checkpoint.pt"
    midpoint = tmp_path / "midpoint.pt"

    def build():
        torch.manual_seed(731)
        engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
        engine.launch()
        model = engine.prepare(Llama(Config(vocab_size=200, max_sequence_length=4,
                                           hidden_size=16, intermediate_size=32,
                                           num_attention_heads=2, num_hidden_layers=1, dropout=0.3)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        loaders = []
        for seed in (9, 10):
            dataset = TokenBinDataset(corpus, max_length=4, seed=seed)
            dataset.training_recipe = {"max_iters": 4, "dropout": 0.3}
            loaders.append(DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(seed)))
        train_dl, val_dl = engine.prepare_dataloaders(*loaders)
        trainer = Trainer(engine=engine, model=model, optimizer=optimizer, train_dataloader=train_dl,
                          val_dataloader=val_dl, get_lr=lambda idx: 0.002 / idx, micro_batch=2,
                          max_iters=4, eval_iters=1, eval_val_batches=1, eval_train_batches=1,
                          save_ckpt_iters=2, checkpoint_path=checkpoint, print_every=10)
        return engine, model, optimizer, trainer, train_dl

    engine, model, optimizer, trainer, _ = build()
    save = engine.save

    def save_and_keep_middle(path, state):
        save(path, state)
        if state["idx"] == 2:
            shutil.copyfile(path, midpoint)

    with patch.object(engine, "save", side_effect=save_and_keep_middle):
        trainer.train()
    expected_model = {key: value.clone() for key, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_tokens = trainer.train_tokens_seen
    engine.close()

    engine, model, optimizer, trainer, loader = build()
    state = engine.load(midpoint, {"model": model, "optimizer": optimizer})
    restore_input_state(loader, state["input_states"][0], gradient_accumulation_steps=2,
                        data_rank=0, data_world_size=1)
    trainer.train_batches_consumed = state["train_batches_consumed"]
    trainer.train_tokens_seen = state["train_tokens_seen"]
    torch.set_rng_state(state["torch_rng_state"])
    # Match CLI startup validation after restoring the process RNG.
    trainer.evaluate(trainer.val_dataloader, 1)
    trainer.train(start_iter=state["idx"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
    actual_optimizer = optimizer.state_dict()
    for index, values in expected_optimizer["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(actual_optimizer["state"][index][key], value, rtol=0, atol=0)
    assert trainer.train_tokens_seen == expected_tokens
    assert trainer.train_batches_consumed == 8
    engine.close()
