import math
import tempfile
import unittest
from pathlib import Path

from ohara.scaling import (
    CosineWeightDecayScheduler,
    MuonMomentumScheduler,
    WarmupStableDecayScheduler,
    analyze_scaling_results,
    append_result_csv,
    fit_isoflop_curves,
    llama_config_for_depth,
    load_result_csv,
    plan_scaling_run,
    write_scaling_svg,
)


class ScalingLawTests(unittest.TestCase):
    def test_depth_is_the_single_model_size_dial(self):
        small = llama_config_for_depth(
            4,
            vocab_size=128,
            sequence_length=32,
            aspect_ratio=32,
            head_dim=64,
            ffn_multiple_of=64,
        )
        large = llama_config_for_depth(
            8,
            vocab_size=128,
            sequence_length=32,
            aspect_ratio=32,
            head_dim=64,
            ffn_multiple_of=64,
        )

        self.assertEqual(small.hidden_size, 128)
        self.assertEqual(large.hidden_size, 256)
        self.assertEqual(small.num_attention_heads, 2)
        self.assertEqual(large.num_attention_heads, 4)
        self.assertGreater(large.intermediate_size, small.intermediate_size)

    def test_fixed_flop_plan_is_self_consistent(self):
        plan = plan_scaling_run(
            4,
            vocab_size=128,
            flops_budget=2e10,
            sequence_length=32,
            device_batch_size=2,
            world_size=1,
            total_batch_size=128,
            aspect_ratio=32,
            head_dim=64,
            ffn_multiple_of=64,
            reference_batch_size=128,
        )

        self.assertEqual(plan.grad_accum_steps, 2)
        self.assertEqual(plan.tokens_trained, plan.total_batch_size * plan.num_iterations)
        self.assertAlmostEqual(
            plan.actual_training_flops,
            plan.flops_per_token * plan.tokens_trained,
        )
        relative_budget_error = abs(plan.actual_training_flops - plan.flops_budget) / plan.flops_budget
        one_step_error = plan.flops_per_token * plan.total_batch_size / plan.flops_budget
        self.assertLessEqual(relative_budget_error, one_step_error)
        self.assertEqual(
            plan.params_effective,
            plan.params_transformer + plan.params_lm_head,
        )

    def test_token_ratio_plan_and_warmup_stable_decay(self):
        plan = plan_scaling_run(
            2,
            vocab_size=64,
            target_param_data_ratio=10,
            sequence_length=16,
            device_batch_size=1,
            total_batch_size=16,
            aspect_ratio=32,
            head_dim=32,
            ffn_multiple_of=32,
            reference_batch_size=16,
        )
        self.assertAlmostEqual(plan.tokens_per_effective_param, 10, delta=0.1)

        scheduler = WarmupStableDecayScheduler(
            learning_rate=1.0,
            max_iters=10,
            warmup_iters=2,
            warmdown_ratio=0.5,
            final_lr_fraction=0.1,
        )
        self.assertEqual(scheduler(0), 0.0)
        self.assertEqual(scheduler(2), 1.0)
        self.assertEqual(scheduler(5), 1.0)
        self.assertAlmostEqual(scheduler(10), 0.28)

        longer_warmup = WarmupStableDecayScheduler(
            learning_rate=1.0,
            max_iters=4,
            warmup_iters=10,
        )
        self.assertEqual(longer_warmup(4), 0.4)

    def test_muon_hyperparameter_schedules_reach_expected_boundaries(self):
        momentum = MuonMomentumScheduler(
            max_iters=100,
            warmdown_ratio=0.5,
            warmup_iters=20,
        )
        decay = CosineWeightDecayScheduler(weight_decay=0.28, max_iters=100)

        self.assertEqual(momentum(1), 0.85)
        self.assertEqual(momentum(21), 0.97)
        self.assertEqual(momentum(51), 0.97)
        self.assertAlmostEqual(momentum(100), 0.9014)
        self.assertEqual(decay(1), 0.28)
        self.assertAlmostEqual(decay(51), 0.14)
        self.assertGreater(decay(100), 0.0)

        short_run = MuonMomentumScheduler(max_iters=40, warmup_iters=400)
        self.assertLess(short_run(40), 0.87)

    def test_csv_roundtrip_and_isoflop_analysis(self):
        rows = []
        for budget, center, scale in [(1e12, 4.0, 1.0), (1e13, 4.5, math.sqrt(10))]:
            for offset in (-1.0, 0.0, 1.0):
                log_params = center + offset
                params = 10**log_params
                tokens_trained = scale * 1e6 * 10**offset
                rows.append(
                    {
                        "flops_budget": budget,
                        "depth": log_params,
                        "params_effective": params,
                        "tokens_trained": tokens_trained,
                        # Exactly consistent with flops_budget = flops_per_token * tokens_trained,
                        # matching the real results.csv schema.
                        "flops_per_token": budget / tokens_trained,
                        "val_bpb": 1.2 + 0.15 * offset**2,
                    }
                )

        optimums = fit_isoflop_curves(rows)
        self.assertEqual(len(optimums), 2)
        self.assertAlmostEqual(math.log10(optimums[0]["params_effective"]), 4.0)
        self.assertAlmostEqual(math.log10(optimums[1]["params_effective"]), 4.5)
        self.assertTrue(all(row["interior_optimum"] == 1.0 for row in optimums))
        # The optimum falls exactly on the offset=0 row, so tokens_trained at
        # the optimum should recover that row's exact value, not an
        # interpolation artifact.
        self.assertLess(abs(optimums[0]["tokens_trained"] - 1.0e6) / 1.0e6, 1e-9)
        self.assertLess(
            abs(optimums[1]["tokens_trained"] - math.sqrt(10) * 1e6) / (math.sqrt(10) * 1e6),
            1e-9,
        )

        analysis = analyze_scaling_results(rows)
        self.assertAlmostEqual(
            analysis["optimal_params_power_law"]["exponent"],
            0.5,
        )
        self.assertIn("optimal_tokens_power_law", analysis)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "results.csv")
            svg_path = Path(directory, "scaling.svg")
            append_result_csv(path, rows[0])
            append_result_csv(path, rows[1])
            loaded = load_result_csv(path)
            write_scaling_svg(svg_path, rows, analysis)
            svg = svg_path.read_text(encoding="utf-8")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["flops_budget"], rows[0]["flops_budget"])
        self.assertIn("<svg", svg)
        self.assertIn("Iso-FLOP curves", svg)


if __name__ == "__main__":
    unittest.main()
