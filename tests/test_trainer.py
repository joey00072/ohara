import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, IterableDataset

from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.trainer import Trainer


class EchoDataset(IterableDataset):
    def __init__(self, vocab_size: int = 16, seq_len: int = 8, ignore_index: int = -1):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.ignore_index = ignore_index

    def __iter__(self):
        g = torch.Generator().manual_seed(123)
        while True:
            x = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
            y = x.clone()
            y[0] = self.ignore_index
            yield x, y


class EchoModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        return F.one_hot(x, num_classes=self.vocab_size).float() * self.scale


class FiniteEchoDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        x = torch.tensor([(index + offset) % 16 for offset in range(8)])
        return x, x.clone()


class TrainerTests(unittest.TestCase):
    def _build_trainer(
        self,
        max_iters: int = 2,
        eval_iters: int = 1,
        token_bytes: torch.Tensor | None = None,
    ) -> Trainer:
        engine = OharaEngine(
            EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32))
        )
        engine.launch()

        train_loader = DataLoader(EchoDataset(), batch_size=2)
        val_loader = DataLoader(EchoDataset(), batch_size=2)
        train_loader, val_loader = engine.prepare_dataloaders(train_loader, val_loader)

        model = EchoModel(vocab_size=16)
        model = engine.prepare(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = engine.prepare_optimizers(optimizer)[0]

        scheduler = lambda _: 1e-3
        return Trainer(
            engine=engine,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            get_lr=scheduler,
            micro_batch=1,
            max_iters=max_iters,
            eval_iters=eval_iters,
            save_ckpt_iters=0,
            ignore_index=-1,
            eval_val_batches=2,
            eval_train_batches=1,
            print_every=1,
            token_bytes=token_bytes,
        )

    def test_evaluate_returns_metrics(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=1)
        metrics = trainer.evaluate(trainer.val_dataloader, num_batches=2)

        self.assertIn("loss", metrics)
        self.assertIn("ppl", metrics)
        self.assertIn("bits_per_token", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.99)

    def test_evaluate_reports_true_bits_per_byte(self):
        token_bytes = torch.ones(16, dtype=torch.int32)
        token_bytes[0] = 0
        trainer = self._build_trainer(
            max_iters=1,
            eval_iters=1,
            token_bytes=token_bytes,
        )

        metrics = trainer.evaluate(trainer.val_dataloader, num_batches=2)

        self.assertIn("bpb", metrics)
        self.assertLess(metrics["bytes"], metrics["tokens"])
        # EchoModel assigns the same cross entropy to every correct class, so
        # excluding the zero-byte special token does not change this ratio.
        self.assertAlmostEqual(metrics["bpb"], metrics["bits_per_token"], places=6)

    def test_evaluate_does_not_advance_persistent_iterator(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=1)
        self.assertIsNone(trainer.train_dataloader._iterator)
        trainer.evaluate(trainer.train_dataloader, num_batches=2)
        self.assertIsNone(trainer.train_dataloader._iterator)

    def test_validation_batches_are_cached(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=1)
        first = trainer.evaluate(trainer.val_dataloader, num_batches=2)
        self.assertEqual(len(trainer._validation_batches), 2)
        second = trainer.evaluate(trainer.val_dataloader, num_batches=2)
        self.assertEqual(first, second)

    def test_train_loop_with_eval_runs(self):
        trainer = self._build_trainer(max_iters=2, eval_iters=1)
        trainer.train()
        self.assertIsNone(trainer.train_dataloader._iterator)

    def test_prepared_dataloader_matches_engine_device(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=0)
        data, target = next(trainer.train_dataloader)
        self.assertEqual(data.device, trainer.engine.device)
        self.assertEqual(target.device, trainer.engine.device)

    def test_true_bf16_casts_model_parameters(self):
        engine = OharaEngine(
            EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.BF16_TRUE))
        )
        model = engine.prepare(nn.Linear(4, 4))
        self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)

    def test_finite_dataloader_cycles_across_epochs(self):
        engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
        model = engine.prepare(EchoModel(vocab_size=16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = engine.prepare_optimizers(optimizer)[0]
        train_loader = engine.prepare_dataloaders(
            DataLoader(FiniteEchoDataset(), batch_size=2, shuffle=True)
        )
        val_loader = engine.prepare_dataloaders(DataLoader(FiniteEchoDataset(), batch_size=2))
        trainer = Trainer(
            engine=engine,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            get_lr=lambda _: 1e-3,
            micro_batch=1,
            max_iters=5,
            eval_iters=0,
            save_ckpt_iters=0,
            print_every=100,
            eval_val_batches=1,
        )
        trainer.train()
        self.assertEqual(trainer.train_dataloader.idx, 2)

    def test_train_scales_group_learning_rates_and_muon_hyperparameters(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=0)
        group = trainer.optimizer.param_groups[0]
        group["lr_scale"] = 3.0
        group["kind"] = "muon"
        group["momentum"] = 0.1
        group["weight_decay"] = 0.0
        trainer.get_lr = lambda _: 0.02
        trainer.get_optimizer_hparams = lambda _: {
            "momentum": 0.91,
            "weight_decay": 0.12,
        }

        trainer.train()

        self.assertEqual(group["lr"], 0.06)
        self.assertEqual(group["momentum"], 0.91)
        self.assertEqual(group["weight_decay"], 0.12)

    def test_final_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp, "model.pt")
            trainer = self._build_trainer(max_iters=2, eval_iters=0)
            trainer.save_ckpt_iters = 1
            trainer.checkpoint_path = checkpoint_path
            trainer.train()

            payload = trainer.engine.load(checkpoint_path)
            self.assertEqual(payload["idx"], 2)
            self.assertIn("optimizer", payload)
            self.assertIn("torch_rng_state", payload)
            self.assertEqual(payload["train_batches_consumed"], 2)
            self.assertEqual(payload["gradient_accumulation_steps"], 1)
            self.assertEqual(list(Path(tmp).glob("*.tmp-*")), [])

            saved_scale = trainer.model.scale.detach().clone()
            with torch.no_grad():
                trainer.model.scale.zero_()
            trainer.engine.load(
                checkpoint_path,
                {"model": trainer.model, "optimizer": trainer.optimizer},
            )
            self.assertTrue(torch.equal(trainer.model.scale, saved_scale))


if __name__ == "__main__":
    unittest.main()
