"""Checkpoint publication preserves non-shape config and requested precision."""
import tempfile
from pathlib import Path

import torch
from dataclasses import asdict

from examples.export_safetensors import export
from ohara.models.llama import Config, Llama


def test_export_restores_checkpoint_config_and_casts_weights():
    config = Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, max_sequence_length=16,
                    moe_shared_exclusive=True)
    model = Llama(config).eval()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory, "model.pt")
        output = Path(directory, "export")
        torch.save({"model": model.state_dict(), "model_config": asdict(config)}, checkpoint)
        result = export(checkpoint, output, dtype=torch.bfloat16)
        loaded = Llama.from_pretrained(output, dtype=torch.bfloat16).eval()
        assert loaded.config == config
        assert result["config"]["moe_shared_exclusive"] is True
        assert next(loaded.parameters()).dtype == torch.bfloat16
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(ids).float(), model.to(torch.bfloat16)(ids).float())
