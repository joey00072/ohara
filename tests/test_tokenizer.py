import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ohara.tokenizer import get_token_bytes, load_tokenizer


class DummyTokenizer:
    def __init__(
        self,
        name_or_path: str,
        *,
        pad_token_id: int | None = 0,
        eos_token_id: int | None = 2,
        eos_token: str | None = "<eos>",
    ):
        self.name_or_path = name_or_path
        self.pad_token_id = pad_token_id
        self.pad_token = None if pad_token_id is None else "<pad>"
        self.eos_token_id = eos_token_id
        self.eos_token = eos_token
        self.bos_token_id = 1
        self.padding_side = None
        self.all_special_ids = [0, 1, 2]
        self.all_special_tokens = ["<pad>", "<bos>", "<eos>"]
        self._decode_map = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "A",
            4: "z",
        }

    def __len__(self):
        return len(self._decode_map)

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.all_special_ids]
        return "".join(self._decode_map.get(i, "") for i in ids)


class TokenizerTests(unittest.TestCase):
    def test_load_tokenizer_prefers_hf(self):
        hf_tok = DummyTokenizer("hf-model")
        with patch("ohara.tokenizer.AutoTokenizer.from_pretrained", return_value=hf_tok) as mocked:
            result = load_tokenizer(hf_name="hf-model", tokenizer_dir="./does-not-matter", prefer_hf=True)

        self.assertEqual(result.source, "hf")
        self.assertIs(result.tokenizer, hf_tok)
        mocked.assert_called_once()
        call_args = mocked.call_args[0]
        self.assertEqual(call_args[0], "hf-model")

    def test_load_tokenizer_falls_back_to_local_when_hf_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp)
            local_dir.joinpath("tokenizer.json").write_text("{}")

            local_tok = DummyTokenizer("local")

            def side_effect(identifier, **kwargs):
                if identifier == "hf-model":
                    raise OSError("network unavailable")
                if identifier == str(local_dir):
                    return local_tok
                raise AssertionError(f"Unexpected tokenizer identifier: {identifier}")

            with patch("ohara.tokenizer.AutoTokenizer.from_pretrained", side_effect=side_effect):
                result = load_tokenizer(
                    hf_name="hf-model",
                    tokenizer_dir=local_dir,
                    prefer_hf=True,
                )

            self.assertEqual(result.source, "local")
            self.assertIs(result.tokenizer, local_tok)

    def test_load_tokenizer_local_first_when_prefer_hf_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp)
            local_dir.joinpath("tokenizer_config.json").write_text("{}")

            local_tok = DummyTokenizer("local")
            hf_tok = DummyTokenizer("hf-model")

            call_order = []

            def side_effect(identifier, **kwargs):
                call_order.append(identifier)
                if identifier == str(local_dir):
                    return local_tok
                if identifier == "hf-model":
                    return hf_tok
                raise AssertionError(f"Unexpected tokenizer identifier: {identifier}")

            with patch("ohara.tokenizer.AutoTokenizer.from_pretrained", side_effect=side_effect):
                result = load_tokenizer(
                    hf_name="hf-model",
                    tokenizer_dir=local_dir,
                    prefer_hf=False,
                )

            self.assertEqual(result.source, "local")
            self.assertEqual(call_order[0], str(local_dir))

    def test_get_token_bytes_zeroes_special_tokens(self):
        tokenizer = DummyTokenizer("dummy")
        token_bytes = get_token_bytes(tokenizer)
        self.assertTrue(torch.equal(token_bytes[:3], torch.zeros(3, dtype=torch.int32)))
        self.assertEqual(int(token_bytes[3].item()), 1)
        self.assertEqual(int(token_bytes[4].item()), 1)

    def test_get_token_bytes_cache_roundtrip(self):
        tokenizer = DummyTokenizer("dummy")
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp).joinpath("token_bytes.pt")
            bytes_first = get_token_bytes(tokenizer, cache_path=cache_path)
            self.assertTrue(cache_path.exists())
            bytes_second = get_token_bytes(tokenizer, cache_path=cache_path)
            self.assertTrue(torch.equal(bytes_first.cpu(), bytes_second.cpu()))
            self.assertEqual(list(Path(tmp).glob(".*.tmp-*")), [])
            with self.assertRaises(ValueError):
                get_token_bytes(DummyTokenizer("different"), cache_path=cache_path)


if __name__ == "__main__":
    unittest.main()
