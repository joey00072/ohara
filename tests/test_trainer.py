import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torch.utils.data import DataLoader, IterableDataset

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


class TrainerTests(unittest.TestCase):
    def _build_trainer(self, max_iters: int = 2, eval_iters: int = 1) -> Trainer:
        fabric = L.Fabric(accelerator="cpu", devices=1, precision="32-true")

        train_loader = DataLoader(EchoDataset(), batch_size=2)
        val_loader = DataLoader(EchoDataset(), batch_size=2)
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

        model = EchoModel(vocab_size=16)
        model = fabric.setup(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = fabric.setup_optimizers(optimizer)

        scheduler = lambda _: 1e-3
        return Trainer(
            fabric=fabric,
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
        )

    def test_evaluate_returns_metrics(self):
        trainer = self._build_trainer(max_iters=1, eval_iters=1)
        metrics = trainer.evaluate(trainer.val_dataloader, num_batches=2)

        self.assertIn("loss", metrics)
        self.assertIn("ppl", metrics)
        self.assertIn("bpb", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.99)

    def test_train_loop_with_eval_runs(self):
        trainer = self._build_trainer(max_iters=2, eval_iters=1)
        trainer.train()


if __name__ == "__main__":
    unittest.main()
