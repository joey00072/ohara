"""Checks for the grouped fine-grained MoE.

The grouped dispatch is an optimisation of a loop that is easy to read and hard
to get wrong. So the load-bearing test is that the two agree: if the sort,
offsets, or scatter-add are subtly off, tokens silently get the wrong expert's
output and training still "works", just worse. Everything else here guards a
property that a shape check would not catch.
"""

import unittest

import torch

from ohara.modules.moe_grouped import GroupedMoE


def build(**overrides):
    options = {
        "dim": 32,
        "hidden_dim": 16,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
    }
    options.update(overrides)
    torch.manual_seed(0)
    return GroupedMoE(**options)


class DispatchEquivalenceTests(unittest.TestCase):
    """The grouped kernel must reproduce the reference loop."""

    def test_reference_and_grouped_agree(self):
        if not torch.cuda.is_available():
            self.skipTest("grouped_mm needs CUDA")
        moe = build().cuda().to(torch.float32).eval()
        x = torch.randn(2, 16, 32, device="cuda")
        flat = x.reshape(-1, 32)
        indices, weights, _, _ = moe._route(flat)
        weights = weights.to(flat.dtype)
        reference = moe._dispatch_reference(flat, indices, weights)
        grouped = moe._dispatch_grouped(flat, indices, weights)
        torch.testing.assert_close(grouped, reference, rtol=1e-4, atol=1e-4)

    def test_reference_dispatch_routes_each_token_to_its_experts(self):
        # Verified without CUDA: build the expected output one token at a time.
        moe = build(num_shared_experts=0).eval()
        x = torch.randn(1, 6, 32)
        flat = x.reshape(-1, 32)
        indices, weights, _, _ = moe._route(flat)
        got = moe._dispatch_reference(flat, indices, weights.to(flat.dtype))

        expected = torch.zeros_like(flat)
        for token in range(flat.size(0)):
            for slot in range(moe.num_experts_per_tok):
                e = int(indices[token, slot])
                xt = flat[token : token + 1]
                hidden = torch.nn.functional.silu(xt @ moe.w_gate[e]) * (xt @ moe.w_up[e])
                expected[token] += (hidden @ moe.w_down[e])[0] * weights[token, slot]
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


class SharedExpertTests(unittest.TestCase):
    def test_shared_expert_applies_to_every_token(self):
        moe = build(num_shared_experts=1).eval()
        with torch.no_grad():
            # Silence the routed path so only the shared expert contributes.
            moe.w_down.zero_()
            moe.shared_down.weight.fill_(0.01)
        x = torch.randn(2, 8, 32)
        out = moe(x)
        # Every position must be non-zero: the shared expert has no routing to skip.
        self.assertTrue(bool((out.abs().sum(dim=-1) > 0).all()))

    def test_no_shared_expert_leaves_module_absent(self):
        moe = build(num_shared_experts=0)
        self.assertFalse(hasattr(moe, "shared_down"))
        self.assertEqual(moe(torch.randn(2, 8, 32)).shape, (2, 8, 32))

    def test_shared_expert_widens_with_count(self):
        self.assertEqual(build(num_shared_experts=2).shared_gate.out_features, 32)
        self.assertEqual(build(num_shared_experts=1).shared_gate.out_features, 16)


class RoutingTests(unittest.TestCase):
    def test_output_shape_is_preserved(self):
        self.assertEqual(build()(torch.randn(3, 7, 32)).shape, (3, 7, 32))

    def test_sigmoid_weights_are_normalised(self):
        moe = build(gate_fn="sigmoid", normalize_weights=True)
        _, weights, _, _ = moe._route(torch.randn(20, 32))
        torch.testing.assert_close(weights.sum(-1), torch.ones(20), rtol=1e-5, atol=1e-5)

    def test_unnormalised_sigmoid_weights_are_free(self):
        moe = build(gate_fn="sigmoid", normalize_weights=False)
        _, weights, _, _ = moe._route(torch.randn(20, 32))
        self.assertFalse(torch.allclose(weights.sum(-1), torch.ones(20), atol=1e-3))

    def test_softmax_weights_sum_to_one(self):
        moe = build(gate_fn="softmax")
        _, weights, _, _ = moe._route(torch.randn(20, 32))
        torch.testing.assert_close(weights.sum(-1), torch.ones(20), rtol=1e-5, atol=1e-5)

    def test_each_token_selects_k_distinct_experts(self):
        moe = build(num_experts_per_tok=4)
        indices, _, _, _ = moe._route(torch.randn(50, 32))
        self.assertEqual(indices.shape, (50, 4))
        for row in indices:
            self.assertEqual(len(set(row.tolist())), 4)

    def test_router_bias_changes_selection_but_not_weights(self):
        moe = build()
        x = torch.randn(40, 32)
        before_idx, before_w, _, _ = moe._route(x)
        with torch.no_grad():
            moe.router_bias[0] = 50.0
        after_idx, after_w, _, _ = moe._route(x)
        # Expert 0 now wins everywhere...
        self.assertTrue(bool((after_idx == 0).any(dim=-1).all()))
        # ...but weights come from unbiased logits, so a token whose selection is
        # unchanged keeps its old weights.
        unchanged = (before_idx == after_idx).all(dim=-1)
        if bool(unchanged.any()):
            torch.testing.assert_close(after_w[unchanged], before_w[unchanged])

    def test_at_init_only_the_output_projection_has_gradient(self):
        """Zero-init w_down makes the routed branch start as an exact no-op.

        Everything downstream of it -- the router, w_gate, w_up -- therefore has
        zero gradient on the very first step, by construction. w_down itself does
        get gradient, which is what lifts the branch off zero so the rest can
        start learning from step two. Documented because it looks like a bug.
        """
        moe = build()
        moe(torch.randn(2, 8, 32)).sum().backward()
        self.assertGreater(float(moe.w_down.grad.abs().sum()), 0)
        self.assertEqual(float(moe.router.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(moe.w_gate.grad.abs().sum()), 0.0)
        # The shared expert is zero-init on its own down projection for the same
        # reason, so it behaves the same way.
        self.assertGreater(float(moe.shared_down.weight.grad.abs().sum()), 0)

    def test_gradient_reaches_router_experts_and_shared_once_trained(self):
        moe = build()
        with torch.no_grad():  # stand in for the state after one optimizer step
            moe.w_down.normal_(0, 0.02)
            moe.shared_down.weight.normal_(0, 0.02)
        moe(torch.randn(2, 8, 32)).sum().backward()
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0)
        self.assertGreater(float(moe.w_gate.grad.abs().sum()), 0)
        self.assertGreater(float(moe.w_up.grad.abs().sum()), 0)
        self.assertGreater(float(moe.shared_gate.weight.grad.abs().sum()), 0)

    def test_router_bias_gets_no_gradient(self):
        # It is solved in closed form; an optimizer must never see it.
        moe = build()
        moe(torch.randn(2, 8, 32)).sum().backward()
        self.assertFalse(moe.router_bias.requires_grad)


class QuantileBalancingTests(unittest.TestCase):
    def test_balancing_moves_the_bias_off_zero(self):
        moe = build().train()
        moe(torch.randn(4, 32, 32))
        moe.apply_qb_update()
        self.assertGreater(float(moe.router_bias.abs().sum()), 0)

    def test_bias_stays_mean_zero(self):
        moe = build().train()
        for _ in range(3):
            moe(torch.randn(4, 32, 32))
            moe.apply_qb_update()
        self.assertAlmostEqual(float(moe.router_bias.mean()), 0.0, places=5)

    def test_balancing_reduces_load_imbalance(self):
        torch.manual_seed(1)
        moe = build(num_experts=8, num_experts_per_tok=2).train()
        with torch.no_grad():  # start badly skewed toward a few experts
            moe.router.weight[0] *= 12.0
            moe.router.weight[1] *= 12.0
        x = torch.randn(8, 64, 32)

        moe(x)
        before = moe.expert_load(reset=True).float()
        for _ in range(30):
            moe(x)
            moe.apply_qb_update()
            moe.expert_load(reset=True)
        moe(x)
        after = moe.expert_load(reset=True).float()

        def maxvio(counts):
            return float((counts.max() - counts.mean()) / counts.mean())

        self.assertLess(maxvio(after), maxvio(before))

    def test_disabled_balancing_leaves_bias_at_zero(self):
        moe = build(quantile_balancing=False).train()
        moe(torch.randn(4, 16, 32))
        moe.apply_qb_update()
        self.assertEqual(float(moe.router_bias.abs().sum()), 0.0)

    def test_eval_mode_does_not_accumulate(self):
        moe = build().eval()
        moe(torch.randn(4, 16, 32))
        self.assertEqual(float(moe.qb_beta_count), 0.0)

    def test_global_load_helper_includes_grouped_moe(self):
        from ohara.modules.moe import expert_load

        moe = build().train()
        moe(torch.randn(2, 10, 32))
        counts = expert_load(moe)
        self.assertEqual(counts.shape, (1, 8))
        self.assertEqual(int(counts.sum()), 2 * 10 * 2)


class ConfigurationTests(unittest.TestCase):
    def test_rejects_top_k_above_expert_count(self):
        with self.assertRaises(ValueError):
            build(num_experts=4, num_experts_per_tok=8)

    def test_quantile_balancing_needs_headroom_for_the_threshold(self):
        with self.assertRaises(ValueError):
            build(num_experts=4, num_experts_per_tok=4, quantile_balancing=True)

    def test_rejects_unknown_gate(self):
        with self.assertRaises(ValueError):
            build(gate_fn="relu")

    def test_llama_rejects_grouped_only_options_without_grouped_moe(self):
        from ohara.models.llama import Config, Llama

        with self.assertRaisesRegex(ValueError, "shared experts"):
            Llama(Config(moe_num_experts=8, moe_num_shared_experts=1))
        with self.assertRaisesRegex(ValueError, "moe_normalize_weights"):
            Llama(Config(moe_num_experts=8, moe_normalize_weights=False))
        with self.assertRaisesRegex(ValueError, "moe_num_experts"):
            Llama(Config(moe_grouped=True))

    def test_expert_weights_are_stacked_not_a_module_list(self):
        # The whole point: one tensor per projection so dispatch is a single GEMM.
        moe = build(num_experts=256, hidden_dim=8, num_experts_per_tok=8)
        self.assertEqual(moe.w_gate.shape, (256, 32, 8))
        self.assertEqual(moe.w_down.shape, (256, 8, 32))

    def test_down_projection_starts_at_zero(self):
        # A fresh expert must be a no-op on the residual stream.
        moe = build()
        self.assertEqual(float(moe.w_down.detach().abs().sum()), 0.0)

    def test_llama_preserves_coordinated_grouped_initialization(self):
        from ohara.models.llama import Config, Llama

        model = Llama(
            Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                init_style="nanochat",
                moe_num_experts=8,
                moe_grouped=True,
                moe_num_shared_experts=1,
            )
        )
        self.assertEqual(float(model.layers[0].ff.w_down.detach().abs().sum()), 0.0)
        self.assertEqual(float(model.layers[0].ff.shared_down.weight.detach().abs().sum()), 0.0)


class OptimizerPartitionTests(unittest.TestCase):
    """Stacked expert weights must reach Muon at the matrix learning rate.

    They are 3-D, and the partitioner's matrix bucket only collects 2-D tensors.
    Before this was handled they fell through to the scalar catch-all and trained
    at the norm-gain rate -- roughly a quarter of the intended one, silently, for
    68% of the model's parameters.
    """

    def _model(self):
        from ohara.models.llama import Config, Llama

        return Llama(
            Config(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=16,
                max_sequence_length=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                dropout=0.0,
                init_style="nanochat",
                moe_num_experts=32,
                moe_experts_per_tok=4,
                moe_grouped=True,
                moe_num_shared_experts=1,
                moe_gate_fn="sigmoid",
            )
        )

    def test_expert_weights_land_in_a_muon_group_at_the_matrix_lr(self):
        from ohara.optimizer import build_muon_adamw

        model = self._model()
        optimizer = build_muon_adamw(model, matrix_learning_rate=0.02, weight_decay=0.1)
        names = {id(p): n for n, p in model.named_parameters()}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if ".ff.w_" in names[id(parameter)]:
                    self.assertEqual(group["kind"], "muon", names[id(parameter)])
                    self.assertAlmostEqual(group["lr"], 0.02)

    def test_every_parameter_is_assigned(self):
        from ohara.optimizer import build_adamh, build_muon_adamw, build_muonh_adamh

        for builder in (build_muon_adamw, build_adamh, build_muonh_adamh):
            model = self._model()
            optimizer = builder(model)
            assigned = {id(p) for g in optimizer.param_groups for p in g["params"]}
            self.assertEqual(assigned, {id(p) for p in model.parameters()}, builder.__name__)

    def test_a_step_moves_expert_weights_and_stays_finite(self):
        from ohara.optimizer import build_muon_adamw

        model = self._model()
        optimizer = build_muon_adamw(model, matrix_learning_rate=0.02, weight_decay=0.1)
        with torch.no_grad():  # lift w_down off zero so gradient reaches the rest
            for block in model.layers:
                block.ff.w_down.normal_(0, 0.02)
        before = model.layers[0].ff.w_gate.detach().clone()
        for _ in range(3):
            model.zero_grad()
            model(torch.randint(0, 256, (2, 8))).sum().backward()
            optimizer.step()
        self.assertGreater(float((model.layers[0].ff.w_gate - before).detach().abs().max()), 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_hyperspherical_optimizers_preserve_each_expert_matrix_norm(self):
        from ohara.optimizer import build_adamh, build_muonh_adamh

        for builder in (build_adamh, build_muonh_adamh):
            model = self._model()
            with torch.no_grad():
                model.layers[0].ff.w_down.normal_(0, 0.02)
                model.layers[1].ff.w_down.normal_(0, 0.02)
            watched = {
                name: torch.linalg.matrix_norm(parameter.detach()).clone()
                for name, parameter in model.named_parameters()
                if parameter.ndim == 3
            }
            optimizer = builder(model)
            model(torch.randint(0, 256, (2, 8))).sum().backward()
            optimizer.step()
            for name, parameter in model.named_parameters():
                if name in watched:
                    torch.testing.assert_close(
                        torch.linalg.matrix_norm(parameter.detach()),
                        watched[name],
                        rtol=1e-5,
                        atol=1e-5,
                    )


class ConfigRecoveryTests(unittest.TestCase):
    """A published checkpoint must rebuild with the routing it was trained with."""

    def _config(self, **overrides):
        from ohara.models.llama import Config

        options = dict(
            vocab_size=320,
            hidden_size=64,
            intermediate_size=28,
            max_sequence_length=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            dropout=0.0,
            moe_num_experts=64,
            moe_experts_per_tok=4,
            moe_grouped=True,
            moe_num_shared_experts=1,
            moe_gate_fn="sigmoid",
        )
        options.update(overrides)
        return Config(**options)

    def test_grouped_moe_config_round_trips(self):
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Llama

        config = self._config()
        recovered = config_from_state_dict(
            Llama(config).state_dict(), moe_experts_per_tok=4, moe_gate_fn="sigmoid"
        )
        self.assertEqual(recovered, config)

    def test_non_shape_routing_options_round_trip_when_supplied(self):
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Llama

        config = self._config(moe_normalize_weights=False)
        recovered = config_from_state_dict(
            Llama(config).state_dict(),
            moe_experts_per_tok=4,
            moe_gate_fn="sigmoid",
            moe_normalize_weights=False,
        )
        self.assertEqual(recovered, config)

    def test_shared_expert_count_is_recovered(self):
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Llama

        for shared in (0, 1, 2):
            config = self._config(moe_num_shared_experts=shared)
            recovered = config_from_state_dict(
                Llama(config).state_dict(), moe_experts_per_tok=4, moe_gate_fn="sigmoid"
            )
            self.assertEqual(recovered.moe_num_shared_experts, shared)

    def test_top_k_defaults_when_not_supplied(self):
        # Nothing in the weights records top-k, so a caller that forgets gets the
        # default rather than an error. This is the sharp edge config.json exists
        # to blunt for published checkpoints.
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Llama

        recovered = config_from_state_dict(Llama(self._config()).state_dict())
        self.assertEqual(recovered.moe_experts_per_tok, 2)  # not the 4 it trained with

    def test_dense_and_loop_moe_still_detected(self):
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Config, Llama

        dense = Config(
            vocab_size=320,
            hidden_size=64,
            intermediate_size=32,
            max_sequence_length=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            dropout=0.0,
        )
        self.assertEqual(config_from_state_dict(Llama(dense).state_dict()).moe_num_experts, 0)

        loop = Config(
            vocab_size=320,
            hidden_size=64,
            intermediate_size=32,
            max_sequence_length=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            dropout=0.0,
            moe_num_experts=8,
            moe_experts_per_tok=2,
        )
        recovered = config_from_state_dict(Llama(loop).state_dict(), moe_gate_fn="sigmoid")
        self.assertEqual(recovered.moe_num_experts, 8)
        self.assertFalse(recovered.moe_grouped)
        self.assertEqual(recovered.moe_gate_fn, "sigmoid")

    def test_single_moe_layer_recovers_a_mixed_dense_layout(self):
        from ohara.chat_engine import config_from_state_dict
        from ohara.models.llama import Llama

        config = self._config(moe_layer_interval=4)
        recovered = config_from_state_dict(
            Llama(config).state_dict(), moe_experts_per_tok=4, moe_gate_fn="sigmoid"
        )
        self.assertEqual(recovered.moe_layer_interval, 4)
        self.assertEqual(recovered, config)


if __name__ == "__main__":
    unittest.main()
