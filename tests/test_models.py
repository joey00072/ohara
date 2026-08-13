"""Every model in ohara.models should construct, run, and reject bad configs.

These are shape-and-contract checks, not quality checks: they exist so that a
refactor cannot silently leave a model unable to run.
"""

from __future__ import annotations

import pytest
import torch

from ohara.models.gemma import Gemma, GemmaConfig
from ohara.models.gpt import GPT
from ohara.models.gpt import Config as GPTConfig
from ohara.models.llama import Config as LlamaConfig
from ohara.models.llama import Llama
from ohara.models.mamba import Mamba, MambaConfig
from ohara.models.phi import Phi, PhiConfig
from ohara.models.retnet import Config as RetNetConfig
from ohara.models.retnet import RetNet
from ohara.models.roformer import Config as RoFormerConfig
from ohara.models.roformer import RoFormer
from ohara.models.transformer import Config as TransformerConfig
from ohara.models.transformer import ModelingLM, Transformer

BATCH, SEQ_LEN, VOCAB = 2, 8, 64


def token_ids(seq_len: int = SEQ_LEN) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB, (BATCH, seq_len), generator=generator)


def small_llama_config(**overrides) -> LlamaConfig:
    defaults = dict(
        vocab_size=VOCAB,
        max_sequence_length=16,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        dropout=0.0,
    )
    return LlamaConfig(**{**defaults, **overrides})


def transformer_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=VOCAB,
        seq_len=16,
        d_model=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    return TransformerConfig(**{**defaults, **overrides})


def build_language_models() -> dict[str, torch.nn.Module]:
    """One small instance of every token-in/logits-out model."""
    return {
        "gpt": GPT(GPTConfig(vocab_size=VOCAB, max_sequence_length=16, dropout=0.0)),
        "gpt-swiglu": GPT(
            GPTConfig(vocab_size=VOCAB, max_sequence_length=16, dropout=0.0, mlp="swiglu")
        ),
        "llama": Llama(small_llama_config()),
        "roformer": RoFormer(RoFormerConfig(vocab_size=VOCAB, seq_len=16, dropout=0.0)),
        "retnet": RetNet(RetNetConfig(vocab_size=VOCAB, seq_len=16, dropout=0.0)),
        "transformer": Transformer(transformer_config()),
        "phi": Phi(
            PhiConfig(
                vocab_size=VOCAB,
                max_sequence_length=16,
                hidden_size=160,
                num_attention_heads=4,
                num_hidden_layers=2,
                multiple_of=2,
            )
        ),
        "gemma": Gemma(
            GemmaConfig(
                vocab_size=VOCAB,
                max_sequence_length=16,
                hidden_size=32,
                num_attention_heads=4,
                num_hidden_layers=2,
            )
        ),
    }


@pytest.mark.parametrize("name", sorted(build_language_models()))
def test_language_model_forward_shape(name: str) -> None:
    model = build_language_models()[name].eval()
    with torch.no_grad():
        logits = model(token_ids())
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB)
    assert torch.isfinite(logits).all()


def test_mamba_forward_shape() -> None:
    model = Mamba(MambaConfig(d_model=32, n_layers=2)).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, SEQ_LEN, 32))
    assert out.shape == (BATCH, SEQ_LEN, 32)
    assert torch.isfinite(out).all()


def test_modeling_lm_wraps_transformer() -> None:
    model = ModelingLM(transformer_config()).eval()
    with torch.no_grad():
        logits = model(token_ids())
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB)


def test_modeling_lm_save_and_load_round_trip(tmp_path) -> None:
    """The HF hub mixin hands the config back as a dict; loading must still work."""
    model = ModelingLM(transformer_config()).eval()
    ids = token_ids()
    with torch.no_grad():
        before = model(ids)

    model.save_pretrained(tmp_path)
    reloaded = ModelingLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(ids)

    assert isinstance(reloaded.config, TransformerConfig)
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize(
    ("model_name", "position"),
    [("llama", 4), ("phi", 4)],
)
def test_cached_decode_matches_full_forward(model_name: str, position: int) -> None:
    """Prefill + one cached step must reproduce the uncached forward pass."""
    model = build_language_models()[model_name].eval()
    ids = token_ids()[:1]

    with torch.no_grad():
        full = model(ids)
        cache = model.build_kv_cache()
        prefill = model(ids[:, :position], cache, 0)
        step = model(ids[:, position : position + 1], cache, position)

    torch.testing.assert_close(prefill[:, -1], full[:, position - 1], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(step[:, -1], full[:, position], rtol=1e-4, atol=1e-4)


def test_gpt_rejects_unknown_mlp() -> None:
    with pytest.raises(ValueError, match="mlp must be one of"):
        GPT(GPTConfig(vocab_size=VOCAB, mlp="not-an-mlp"))


def test_gpt_rejects_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPT(GPTConfig(vocab_size=VOCAB, hidden_size=30, num_attention_heads=4))


def test_gpt_rejects_sequence_longer_than_context() -> None:
    model = GPT(GPTConfig(vocab_size=VOCAB, max_sequence_length=16)).eval()
    with pytest.raises(ValueError, match="max_sequence_length"), torch.no_grad():
        model(token_ids(seq_len=17))


def test_roformer_requires_even_head_dim() -> None:
    # 12 / 4 = 3, an odd head dim, which rotary embeddings cannot split in pairs.
    with pytest.raises(ValueError, match="even"):
        RoFormer(RoFormerConfig(vocab_size=VOCAB, d_model=12, num_heads=4))


def test_retnet_requires_divisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RetNet(RetNetConfig(vocab_size=VOCAB, d_model=30, num_heads=4))


def test_llama_scaling_param_groups_sum_to_total() -> None:
    model = Llama(small_llama_config())
    counts = model.num_scaling_params()
    assert (
        counts["token_embeddings"]
        + counts["lm_head"]
        + counts["transformer_matrices"]
        + counts["norms_and_scalars"]
        == counts["total"]
    )
    assert counts["total"] == sum(p.numel() for p in model.parameters())
