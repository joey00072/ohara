import gc
import weakref
from types import SimpleNamespace

import torch

from examples import train_llama_engine


def test_resume_releases_loaded_checkpoint_payload(monkeypatch):
    payload_tensor = torch.ones(4)
    payload_ref = weakref.ref(payload_tensor)

    class Engine:
        world_size = 1
        global_rank = 0
        data_parallel_rank = 0
        data_parallel_world_size = 1
        is_global_zero = False

        def __init__(self, tensor):
            self.payload = {
                "model": {"weight": tensor},
                "idx": 7,
                "input_states": [{}],
                "train_batches_consumed": 11,
            }

        def load(self, *_args, **_kwargs):
            payload, self.payload = self.payload, None
            return payload

    engine = Engine(payload_tensor)
    trainer = SimpleNamespace(train_batches_consumed=0, train_tokens_seen=0)
    monkeypatch.setattr(train_llama_engine, "restore_input_state", lambda *_args, **_kwargs: None)
    del payload_tensor

    start_iter = train_llama_engine._restore_training_checkpoint(
        engine=engine,
        checkpoint_path="unused.pt",
        model=object(),
        optimizer=object(),
        train_dataloader=object(),
        trainer=trainer,
        gradient_accumulation_steps=1,
    )
    gc.collect()

    assert start_iter == 7
    assert trainer.train_batches_consumed == 11
    assert payload_ref() is None
