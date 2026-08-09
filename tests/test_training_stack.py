import copy
import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from ohara.lr_scheduler import CosineScheduler
from ohara.models.llama import Config, Llama
from ohara.optimizer import MuonAdamW, build_adamw, build_muon_adamw
from ohara.runtime import EngineConfig, OharaEngine, PrecisionConfig, PrecisionMode
from ohara.trainer import Trainer


class TrainingStackTests(unittest.TestCase):
    def test_scheduler_boundaries_and_validation(self):
        scheduler = CosineScheduler(
            learning_rate=1.0,
            min_lr=0.1,
            warmup_iters=2,
            max_iters=10,
        )
        self.assertEqual(scheduler(0), 0.0)
        self.assertEqual(scheduler(2), 1.0)
        self.assertAlmostEqual(scheduler(10), 0.1)
        self.assertEqual(scheduler(11), 0.1)

        no_warmup = CosineScheduler(
            learning_rate=1.0,
            min_lr=0.0,
            warmup_iters=0,
            max_iters=4,
        )
        self.assertEqual(no_warmup(0), 1.0)
        with self.assertRaises(ValueError):
            CosineScheduler(warmup_iters=4, max_iters=4)
        with self.assertRaises(ValueError):
            scheduler(-1)

    def test_llama_config_validation_and_rope_theta(self):
        first = Llama(
            Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=1,
                max_sequence_length=8,
                rope_theta=10_000,
            )
        )
        second = Llama(
            Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=1,
                max_sequence_length=8,
                rope_theta=100_000,
            )
        )
        self.assertFalse(torch.equal(first.freq_cos, second.freq_cos))
        with self.assertRaises(ValueError):
            Llama(Config(hidden_size=15, num_attention_heads=4))

    def test_nanochat_initialization_zeros_residual_projections(self):
        torch.manual_seed(3)
        model = Llama(
            Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_hidden_layers=2,
                max_sequence_length=8,
                dropout=0.0,
                weight_tying=False,
                init_style="nanochat",
            )
        )
        for block in model.layers:
            self.assertEqual(torch.count_nonzero(block.attn.proj.weight), 0)
            self.assertEqual(torch.count_nonzero(block.ff.down.weight), 0)
            self.assertGreater(torch.count_nonzero(block.attn.query.weight), 0)
        self.assertAlmostEqual(float(model.token_emb.weight.detach().std()), 0.8, delta=0.08)
        self.assertAlmostEqual(
            float(model.vocab_proj.weight.detach().std()), 0.001, delta=0.0001
        )

        with self.assertRaises(ValueError):
            Llama(
                Config(
                    vocab_size=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=4,
                    num_hidden_layers=1,
                    max_sequence_length=8,
                    weight_tying=True,
                    init_style="nanochat",
                )
            )

    def test_llama_kv_cache_matches_full_causal_forward(self):
        torch.manual_seed(11)
        model = Llama(
            Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=2,
                max_sequence_length=8,
                dropout=0.0,
            )
        ).eval()
        tokens = torch.tensor([[1, 4, 7, 2, 9]])
        full_logits = model(tokens)

        cache = model.build_kv_cache()
        prefill_logits = model(tokens[:, :3], cache, 0)
        self.assertTrue(
            torch.allclose(prefill_logits, full_logits[:, :3], atol=1e-5, rtol=1e-4)
        )
        for position in range(3, tokens.size(1)):
            cached_logits = model(tokens[:, position : position + 1], cache, position)
            self.assertTrue(
                torch.allclose(
                    cached_logits[:, -1],
                    full_logits[:, position],
                    atol=1e-5,
                    rtol=1e-4,
                )
            )

    def test_llama_kv_cache_accepts_chunk_and_validates_bounds(self):
        model = Llama(
            Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=1,
                max_sequence_length=4,
                dropout=0.0,
            )
        ).eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        full_logits = model(tokens)
        cache = model.build_kv_cache()
        model(tokens[:, :2], cache, 0)
        chunk_logits = model(tokens[:, 2:], cache, 2)
        self.assertTrue(
            torch.allclose(chunk_logits, full_logits[:, 2:], atol=1e-5, rtol=1e-4)
        )
        with self.assertRaises(ValueError):
            model(torch.tensor([[5]]), cache, 4)

        non_sequential_cache = model.build_kv_cache()
        with self.assertRaises(ValueError):
            model(tokens[:, 1:2], non_sequential_cache, 1)

    def test_optimizer_does_not_decay_norm_parameters(self):
        model = Llama(
            Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=1,
                max_sequence_length=8,
            )
        )
        optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=0.1)
        decay_group, no_decay_group = optimizer.param_groups
        self.assertEqual(decay_group["weight_decay"], 0.1)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        self.assertTrue(all(parameter.ndim >= 2 for parameter in decay_group["params"]))
        self.assertTrue(all(parameter.ndim < 2 for parameter in no_decay_group["params"]))

    def test_muon_optimizer_partition_is_complete_and_shape_grouped(self):
        model = Llama(
            Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_hidden_layers=2,
                max_sequence_length=8,
                weight_tying=False,
            )
        )
        optimizer = build_muon_adamw(model)
        grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]

        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertEqual(
            next(group for group in optimizer.param_groups if group["name"] == "embedding")[
                "kind"
            ],
            "adamw",
        )
        self.assertEqual(
            next(
                group for group in optimizer.param_groups if group["name"] == "unembedding"
            )["kind"],
            "adamw",
        )
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], 0.02 * group["lr_scale"])
            if group["kind"] == "muon":
                self.assertTrue(all(parameter.ndim == 2 for parameter in group["params"]))
                self.assertEqual(len({parameter.shape for parameter in group["params"]}), 1)

    def test_hybrid_adamw_group_matches_torch_adamw(self):
        first = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        second = torch.nn.Parameter(first.detach().clone())
        group = {
            "kind": "adamw",
            "params": [first],
            "lr": 0.03,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.2,
        }
        hybrid = MuonAdamW([group])
        reference = torch.optim.AdamW(
            [second],
            lr=0.03,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.2,
            foreach=False,
            fused=False,
        )

        for gradient in (torch.tensor([0.4, -0.1, 0.2]), torch.tensor([-0.2, 0.3, 0.1])):
            first.grad = gradient.clone()
            second.grad = gradient.clone()
            hybrid.step()
            reference.step()
        self.assertTrue(torch.allclose(first, second, atol=1e-7, rtol=1e-6))

    def test_muon_update_is_finite_and_state_dict_resumes_exactly(self):
        torch.manual_seed(31)
        config = Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_hidden_layers=1,
            max_sequence_length=8,
            dropout=0.0,
            weight_tying=False,
        )
        first = Llama(config)
        first_optimizer = build_muon_adamw(first)
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))

        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(inputs).reshape(-1, config.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()

        update(first, first_optimizer)
        second = Llama(config)
        second.load_state_dict(first.state_dict())
        second_optimizer = build_muon_adamw(second)
        second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

        update(first, first_optimizer)
        update(second, second_optimizer)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))
            self.assertTrue(torch.isfinite(first_parameter).all())

    def test_cpu_end_to_end_loss_decreases(self):
        torch.manual_seed(7)
        vocab_size = 16
        seq_len = 8
        starts = torch.arange(64).unsqueeze(1) % vocab_size
        positions = torch.arange(seq_len + 1).unsqueeze(0)
        blocks = (starts + positions) % vocab_size
        dataset = TensorDataset(blocks[:, :-1], blocks[:, 1:])

        engine = OharaEngine(EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32)))
        model = engine.prepare(
            Llama(
                Config(
                    vocab_size=vocab_size,
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_hidden_layers=1,
                    max_sequence_length=seq_len,
                    dropout=0.0,
                    weight_tying=True,
                )
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)
        optimizer = engine.prepare_optimizers(optimizer)[0]
        train_loader, val_loader = engine.prepare_dataloaders(
            DataLoader(dataset, batch_size=8, shuffle=True),
            DataLoader(dataset, batch_size=8),
        )
        trainer = Trainer(
            engine=engine,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            get_lr=lambda _: 2e-2,
            micro_batch=2,
            max_iters=12,
            eval_iters=0,
            save_ckpt_iters=0,
            eval_val_batches=4,
            grad_clip_norm=1.0,
            print_every=100,
        )
        initial = trainer.evaluate(trainer.val_dataloader, 4)["loss"]
        self.assertAlmostEqual(initial, math.log(vocab_size), delta=0.5)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        trainer.train()
        final = trainer.evaluate(trainer.val_dataloader, 4)["loss"]

        self.assertLess(final, initial * 0.5)
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        )

    def test_gradient_accumulation_matches_single_batch_with_ignored_tokens(self):
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 6)
                self.projection = torch.nn.Linear(6, 8)

            def forward(self, token_ids):
                return self.projection(self.embedding(token_ids))

        inputs = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
            ]
        )
        targets = torch.tensor(
            [
                [1, 2, 3, -1],
                [2, -1, -1, -1],
                [3, 4, 5, 6],
                [4, 5, -1, -1],
            ]
        )

        torch.manual_seed(19)
        accumulated_model = TinyLM()
        single_batch_model = TinyLM()
        single_batch_model.load_state_dict(accumulated_model.state_dict())

        def train_once(model, batch_size, accumulation_steps):
            engine = OharaEngine(
                EngineConfig(precision=PrecisionConfig(mode=PrecisionMode.FP32))
            )
            model = engine.prepare(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer = engine.prepare_optimizers(optimizer)[0]
            loader = engine.prepare_dataloaders(
                DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
            )
            trainer = Trainer(
                engine=engine,
                model=model,
                optimizer=optimizer,
                train_dataloader=loader,
                val_dataloader=loader,
                get_lr=lambda _: 0.1,
                micro_batch=accumulation_steps,
                max_iters=1,
                eval_iters=0,
                save_ckpt_iters=0,
                ignore_index=-1,
                print_every=100,
                eval_val_batches=1,
            )
            trainer.train()
            return model

        accumulated_model = train_once(accumulated_model, batch_size=2, accumulation_steps=2)
        single_batch_model = train_once(single_batch_model, batch_size=4, accumulation_steps=1)
        for accumulated, single in zip(
            accumulated_model.parameters(), single_batch_model.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(accumulated, single, atol=1e-7, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
