import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from examples.prepare_dataset import RECIPES
from examples.pretokenize_corpus import main
from ohara.pretokenize import DatasetPreprocessor, OpenHermesDatasetPreprocessor


def local_tokenizer():
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "<bos>": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.name_or_path = "local-first"
    return tokenizer


def test_real_chat_template_produces_token_list():
    tokenizer = local_tokenizer()
    source = Dataset.from_list([{"conversations": [{"from": "human", "value": "hello"}]}])
    with patch("ohara.pretokenize.get_tokenizer", return_value=tokenizer), patch(
        "ohara.pretokenize.load_dataset", return_value=source
    ):
        processor = OpenHermesDatasetPreprocessor(min_length=1, num_proc=1)
        processed = processor.load_and_preprocess_dataset("train")
    assert list(processed["input_ids"]) == [[1]]


def test_missing_cpu_count_and_tinystories_minimum():
    with patch("ohara.pretokenize.get_tokenizer", return_value=local_tokenizer()), patch(
        "ohara.pretokenize.os.cpu_count", return_value=None
    ):
        assert DatasetPreprocessor().num_proc == 1
    assert RECIPES["tinystories"].min_length == 2


def test_bin_reuse_rejects_same_size_tokenizer_and_document_limit_changes(tmp_path):
    Path(tmp_path, "train.jsonl").write_text(json.dumps({"text": "hello"}) + "\n")
    tokenizer = local_tokenizer()
    args = SimpleNamespace(corpus=str(tmp_path), splits="train", tokenizer="local-first",
                           tokenizer_local_files_only=True, chat_tokens=False, text_column="text",
                           batch_size=2, max_documents=1, force=False)
    with patch("examples.pretokenize_corpus.parse_args", return_value=args), patch(
        "examples.pretokenize_corpus.get_tokenizer", return_value=tokenizer
    ):
        main()
        main()  # Identical inputs reuse the completed bin.
        tokenizer.name_or_path = "local-second"
        with pytest.raises(FileExistsError):
            main()
        tokenizer.name_or_path = "local-first"
        args.max_documents = None
        with pytest.raises(FileExistsError):
            main()
