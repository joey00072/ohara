import json
import threading
import unittest
import urllib.error
import urllib.request

import torch

from ohara.chat import (
    ASSISTANT_END,
    ASSISTANT_START,
    CHAT_SPECIAL_TOKENS,
    USER_END,
    USER_START,
    add_chat_tokens,
    normalize_messages,
    render_conversation,
    render_for_completion,
    resize_token_embeddings,
    special_token_ids,
    training_pair,
)
from ohara.chat_engine import ChatEngine, SamplingConfig, config_from_state_dict, sample_next_token
from ohara.models.llama import Config, Llama
from ohara.sft import ConversationDataset, render_multiple_choice
from ohara.webui.server import create_server


class FakeTokenizer:
    """A dense character-level vocabulary that speaks enough of the HF API.

    Keeps these tests off the network while still exercising the real vocabulary
    growth path that adding chat tokens depends on.
    """

    def __init__(self):
        self._vocab = {"<bos>": 0}
        for index, code in enumerate(range(32, 127)):
            self._vocab[chr(code)] = index + 1
        self._ids = {value: key for key, value in self._vocab.items()}
        self._special = {"<bos>"}
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, mapping):
        added = 0
        for token in mapping["additional_special_tokens"]:
            if token not in self._vocab:
                index = len(self._vocab)
                self._vocab[token] = index
                self._ids[index] = token
                self._special.add(token)
                added += 1
        return added

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[char] for char in text if char in self._vocab]

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for token_id in ids:
            token = self._ids.get(int(token_id), "")
            if skip_special_tokens and token in self._special:
                continue
            pieces.append(token)
        return "".join(pieces)


def chat_tokenizer():
    tokenizer = FakeTokenizer()
    add_chat_tokens(tokenizer)
    return tokenizer


class ChatRenderingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = chat_tokenizer()
        self.specials = special_token_ids(self.tokenizer)

    def test_add_chat_tokens_is_idempotent(self):
        tokenizer = FakeTokenizer()
        base = len(tokenizer)
        self.assertEqual(add_chat_tokens(tokenizer), len(CHAT_SPECIAL_TOKENS))
        self.assertEqual(len(tokenizer), base + len(CHAT_SPECIAL_TOKENS))
        self.assertEqual(add_chat_tokens(tokenizer), 0)
        self.assertEqual(len(tokenizer), base + len(CHAT_SPECIAL_TOKENS))

    def test_special_token_ids_requires_chat_tokens(self):
        with self.assertRaises(ValueError):
            special_token_ids(FakeTokenizer())

    def test_only_assistant_tokens_are_supervised(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        ids, mask = render_conversation(self.tokenizer, messages)
        self.assertEqual(len(ids), len(mask))
        # bos, <user_start>, h, i, <user_end>, <assistant_start>, y, o, <assistant_end>
        self.assertEqual(ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(mask[0], 0)
        self.assertEqual(ids[1], self.specials[USER_START])
        self.assertEqual(ids[4], self.specials[USER_END])
        self.assertEqual(ids[5], self.specials[ASSISTANT_START])
        self.assertEqual(mask[5], 0, "the assistant-start cue is read, not produced")
        self.assertEqual(mask[6:9], [1, 1, 1])
        self.assertEqual(ids[8], self.specials[ASSISTANT_END])
        self.assertEqual(mask[8], 1, "the model must learn to stop")

    def test_system_message_is_folded_into_first_user_turn(self):
        merged = normalize_messages(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["role"], "user")
        self.assertEqual(merged[0]["content"], "be brief\n\nhi")

    def test_non_alternating_conversation_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_messages(
                [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ]
            )

    def test_normalize_does_not_mutate_input(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        normalize_messages(messages)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"], "u")

    def test_tool_call_parts_supervise_the_call_but_not_its_output(self):
        messages = [
            {"role": "user", "content": "add"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "x"},
                    {"type": "python", "text": "1+1"},
                    {"type": "python_output", "text": "2"},
                ],
            },
        ]
        ids, mask = render_conversation(self.tokenizer, messages)
        supervised = {
            token for token, flag in zip(ids, mask) if flag == 1
        }
        self.assertIn(self.specials["<|python_start|>"], supervised)
        self.assertNotIn(self.specials["<|output_start|>"], supervised)
        # The interpreter's reply "2" is read back, never generated.
        output_index = ids.index(self.specials["<|output_start|>"])
        self.assertEqual(mask[output_index + 1], 0)

    def test_render_truncates_to_max_tokens(self):
        messages = [
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
        ]
        ids, mask = render_conversation(self.tokenizer, messages, max_tokens=64)
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(mask), 64)

    def test_training_pair_shifts_and_masks(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        ids, mask = render_conversation(self.tokenizer, messages)
        inputs, targets = training_pair(ids, mask)
        self.assertEqual(len(inputs), len(ids) - 1)
        self.assertEqual(len(targets), len(ids) - 1)
        # Position i predicts token i+1, supervised where that token is assistant text.
        for index, target in enumerate(targets):
            expected = ids[index + 1] if mask[index + 1] == 1 else -1
            self.assertEqual(target, expected)
        self.assertGreater(sum(1 for value in targets if value != -1), 0)

    def test_render_for_completion_primes_the_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "should be dropped"},
        ]
        ids = render_for_completion(self.tokenizer, messages)
        self.assertEqual(ids[-1], self.specials[ASSISTANT_START])
        self.assertNotIn(self.specials[ASSISTANT_END], ids)


class ResizeEmbeddingTests(unittest.TestCase):
    def _model(self, vocab_size, weight_tying=False):
        return Llama(
            Config(
                vocab_size=vocab_size,
                hidden_size=32,
                intermediate_size=64,
                max_sequence_length=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                dropout=0.0,
                weight_tying=weight_tying,
            )
        )

    def test_resize_preserves_pretrained_rows(self):
        model = self._model(50)
        original_embedding = model.token_emb.weight.detach().clone()
        original_head = model.vocab_proj.weight.detach().clone()
        resize_token_embeddings(model, 58)
        self.assertEqual(model.token_emb.weight.shape[0], 58)
        self.assertEqual(model.vocab_proj.weight.shape[0], 58)
        self.assertEqual(model.config.vocab_size, 58)
        torch.testing.assert_close(model.token_emb.weight[:50], original_embedding)
        torch.testing.assert_close(model.vocab_proj.weight[:50], original_head)

    def test_resize_keeps_weights_tied(self):
        model = self._model(50, weight_tying=True)
        resize_token_embeddings(model, 58)
        self.assertIs(model.vocab_proj.weight, model.token_emb.weight)

    def test_resize_is_a_noop_at_the_same_size(self):
        model = self._model(50)
        embedding = model.token_emb
        resize_token_embeddings(model, 50)
        self.assertIs(model.token_emb, embedding)

    def test_resize_refuses_to_shrink(self):
        with self.assertRaises(ValueError):
            resize_token_embeddings(self._model(50), 40)

    def test_resized_model_still_runs_forward(self):
        model = self._model(50)
        resize_token_embeddings(model, 58)
        logits = model(torch.tensor([[57, 3, 12]]))
        self.assertEqual(logits.shape, (1, 3, 58))


class SFTPackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = chat_tokenizer()
        self.conversations = [
            {
                "messages": [
                    {"role": "user", "content": "q" * (index % 7 + 1)},
                    {"role": "assistant", "content": "a" * (index % 5 + 1)},
                ]
            }
            for index in range(40)
        ]

    def _dataset(self, **kwargs):
        options = {
            "max_length": 64,
            "buffer_size": 8,
            "infinite": False,
            "seed": 0,
        }
        options.update(kwargs)
        return ConversationDataset(self.conversations, self.tokenizer, **options)

    def test_rows_have_the_requested_shape(self):
        for inputs, targets in self._dataset():
            self.assertEqual(inputs.shape, (64,))
            self.assertEqual(targets.shape, (64,))
            self.assertEqual(inputs.dtype, torch.long)
            self.assertEqual(targets.dtype, torch.long)

    def test_every_row_starts_at_a_conversation_boundary(self):
        for inputs, _ in self._dataset():
            self.assertEqual(int(inputs[0]), self.tokenizer.bos_token_id)

    def test_every_row_supervises_something(self):
        # A row with no supervised target would make Trainer raise.
        for _, targets in self._dataset():
            self.assertGreater(int((targets != -1).sum()), 0)

    def test_targets_are_the_shifted_inputs_where_supervised(self):
        inputs, targets = next(iter(self._dataset()))
        supervised = targets != -1
        # targets[i] is the token that follows inputs[i], i.e. inputs[i + 1].
        for index in range(len(inputs) - 1):
            if supervised[index]:
                self.assertEqual(int(targets[index]), int(inputs[index + 1]))

    def test_finite_dataset_terminates(self):
        rows = list(self._dataset())
        self.assertGreater(len(rows), 0)

    def test_infinite_dataset_keeps_yielding(self):
        iterator = iter(self._dataset(infinite=True))
        rows = [next(iterator) for _ in range(50)]
        self.assertEqual(len(rows), 50)

    def test_malformed_conversations_are_skipped(self):
        conversations = [
            {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
            {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "y"}]},
        ]
        dataset = ConversationDataset(
            conversations, self.tokenizer, max_length=32, buffer_size=4, infinite=False
        )
        rows = list(dataset)
        self.assertEqual(len(rows), 1)

    def test_shards_partition_the_data(self):
        common = {"max_length": 64, "buffer_size": 8, "infinite": False, "shuffle": False}
        first = list(
            ConversationDataset(
                self.conversations, self.tokenizer, data_rank=0, data_world_size=2, **common
            )
        )
        second = list(
            ConversationDataset(
                self.conversations, self.tokenizer, data_rank=1, data_world_size=2, **common
            )
        )
        self.assertGreater(len(first), 0)
        self.assertGreater(len(second), 0)
        whole = list(
            ConversationDataset(
                self.conversations, self.tokenizer, data_rank=0, data_world_size=1, **common
            )
        )
        # Two half-size shards should together cover about as much as one full pass.
        self.assertLessEqual(abs((len(first) + len(second)) - len(whole)), 2)

    def test_rejects_empty_conversation_list(self):
        with self.assertRaises(ValueError):
            ConversationDataset([], self.tokenizer, max_length=32)


class MultipleChoiceRenderingTests(unittest.TestCase):
    def test_letter_follows_the_choice_without_leading_space(self):
        prompt = render_multiple_choice("2+2?", ["3", "4", "5", "6"])
        self.assertIn("- 4=B\n", prompt)
        self.assertNotIn("= B", prompt)
        self.assertTrue(prompt.startswith("Multiple Choice question: 2+2?"))


class SamplingTests(unittest.TestCase):
    def test_zero_temperature_is_greedy(self):
        logits = torch.tensor([[[0.1, 5.0, 0.2, 0.3]]])
        token = sample_next_token(logits, SamplingConfig(temperature=0.0))
        self.assertEqual(int(token.item()), 1)

    def test_top_k_restricts_the_candidate_set(self):
        logits = torch.tensor([[[10.0, 9.0, -50.0, -60.0]]])
        config = SamplingConfig(temperature=1.0, top_p=1.0, top_k=2)
        samples = {
            int(sample_next_token(logits, config).item()) for _ in range(200)
        }
        self.assertTrue(samples <= {0, 1}, f"top-k leaked: {samples}")

    def test_top_p_keeps_the_most_likely_token(self):
        # One token holds nearly all the mass; a tiny top_p must still keep it.
        logits = torch.tensor([[[20.0, 0.0, 0.0, 0.0]]])
        config = SamplingConfig(temperature=1.0, top_p=0.01)
        samples = {int(sample_next_token(logits, config).item()) for _ in range(50)}
        self.assertEqual(samples, {0})

    def test_seeded_sampling_is_reproducible(self):
        logits = torch.randn(1, 1, 32)
        config = SamplingConfig(temperature=1.0, top_p=0.9)
        first = sample_next_token(logits, config, torch.Generator().manual_seed(7))
        second = sample_next_token(logits, config, torch.Generator().manual_seed(7))
        self.assertEqual(int(first.item()), int(second.item()))

    def test_invalid_configurations_are_rejected(self):
        for kwargs in (
            {"temperature": -1.0},
            {"top_p": 1.5},
            {"top_k": -1},
            {"max_new_tokens": 0},
        ):
            with self.assertRaises(ValueError):
                SamplingConfig(**kwargs)


def tiny_engine(tokenizer, *, max_sequence_length=128, forced_token=None):
    """A ChatEngine over a tiny random model, optionally forced to one token."""
    config = Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=2,
        num_attention_heads=2,
        dropout=0.0,
    )
    model = Llama(config)
    if forced_token is not None:
        # Drive the output head so the model deterministically emits one token.
        with torch.no_grad():
            model.vocab_proj.weight.zero_()
            model.vocab_proj.weight[forced_token] = 5.0
            model.token_emb.weight.fill_(1.0)
    return ChatEngine(model, tokenizer, device="cpu", dtype=torch.float32)


class ChatEngineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = chat_tokenizer()
        self.specials = special_token_ids(self.tokenizer)

    def test_config_roundtrips_through_a_state_dict(self):
        config = Config(
            vocab_size=len(self.tokenizer),
            hidden_size=64,
            intermediate_size=128,
            max_sequence_length=96,
            num_hidden_layers=3,
            num_attention_heads=2,
            dropout=0.0,
        )
        recovered = config_from_state_dict(Llama(config).state_dict())
        self.assertEqual(recovered, config)

    def test_generation_stops_at_assistant_end(self):
        engine = tiny_engine(self.tokenizer, forced_token=self.specials[ASSISTANT_END])
        deltas = list(
            engine.generate_stream(
                [{"role": "user", "content": "hi"}],
                SamplingConfig(temperature=0.0, max_new_tokens=32),
            )
        )
        self.assertEqual(deltas, [], "the stop token must not be emitted as text")

    def test_generation_respects_max_new_tokens(self):
        letter = self.tokenizer.get_vocab()["x"]
        engine = tiny_engine(self.tokenizer, forced_token=letter)
        text = engine.generate(
            [{"role": "user", "content": "hi"}],
            SamplingConfig(temperature=0.0, max_new_tokens=5),
        )
        self.assertEqual(text, "xxxxx")

    def test_streaming_and_batch_generation_agree(self):
        letter = self.tokenizer.get_vocab()["q"]
        engine = tiny_engine(self.tokenizer, forced_token=letter)
        messages = [{"role": "user", "content": "hi"}]
        config = SamplingConfig(temperature=0.0, max_new_tokens=6)
        streamed = "".join(engine.generate_stream(messages, config))
        self.assertEqual(streamed, engine.generate(messages, config))

    def test_long_history_drops_oldest_turns(self):
        engine = tiny_engine(self.tokenizer, max_sequence_length=64)
        messages = []
        for index in range(20):
            messages.append({"role": "user", "content": "u" * 20})
            messages.append({"role": "assistant", "content": f"a{index}" * 5})
        messages.append({"role": "user", "content": "final question"})
        ids = engine.render_prompt(messages)
        self.assertLess(len(ids), 64)
        self.assertEqual(ids[-1], self.specials[ASSISTANT_START])

    def test_message_that_cannot_fit_raises(self):
        engine = tiny_engine(self.tokenizer, max_sequence_length=32)
        with self.assertRaises(ValueError):
            engine.render_prompt([{"role": "user", "content": "z" * 500}])

    def test_metadata_reports_model_shape(self):
        engine = tiny_engine(self.tokenizer)
        info = engine.metadata()
        self.assertEqual(info["layers"], 2)
        self.assertEqual(info["hidden_size"], 32)
        self.assertEqual(info["context_length"], 128)
        self.assertEqual(info["vocab_size"], len(self.tokenizer))
        self.assertGreater(info["parameters"], 0)


class WebUIServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = chat_tokenizer()
        letter = cls.tokenizer.get_vocab()["h"]
        cls.engine = tiny_engine(cls.tokenizer, forced_token=letter)
        cls.server = create_server(
            cls.engine,
            host="127.0.0.1",
            port=0,
            sampling=SamplingConfig(temperature=0.0, max_new_tokens=4),
        )
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as response:
            return response.status, response.read()

    def post(self, path, payload):
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")

    def test_serves_the_page_and_its_assets(self):
        for path, needle in (
            # Anchor on the elements the chat script binds to, not on cosmetics.
            ("/", b'id="composer-form"'),
            ("/", b'id="messages"'),
            ("/static/app.js", b"/api/chat"),
            ("/static/style.css", b"--accent"),
        ):
            status, body = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn(needle, body)

    def test_info_reports_model_and_defaults(self):
        _, body = self.get("/api/info")
        payload = json.loads(body)
        self.assertEqual(payload["model"]["layers"], 2)
        self.assertEqual(payload["defaults"]["max_new_tokens"], 4)

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_static_path_traversal_is_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/static/../../../../etc/passwd")
        self.assertEqual(caught.exception.code, 404)

    def test_chat_streams_sse_events(self):
        _, body = self.post("/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
        events = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(events)
        self.assertEqual(events[-1], {"done": True})
        text = "".join(event.get("delta", "") for event in events)
        self.assertEqual(text, "hhhh")

    def test_chat_rejects_malformed_requests(self):
        for payload in (
            {"messages": []},
            {"messages": "hi"},
            {"messages": [{"role": "wizard", "content": "hi"}]},
            {"messages": [{"role": "user", "content": 42}]},
        ):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post("/api/chat", payload)
            self.assertEqual(caught.exception.code, 400, payload)


if __name__ == "__main__":
    unittest.main()
