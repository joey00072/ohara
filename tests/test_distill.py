"""Loss/gradient equivalence and real cache alignment regressions."""
from types import SimpleNamespace

import pytest
import torch

from ohara.distill import DistillTokenBinDataset, build_teacher_cache, distillation_loss
from ohara.tokenbin import TokenBinDataset, write_token_bin


@pytest.mark.parametrize("logits", [[20., 0.], [12., 0.], [0.1, 0.2]])
@pytest.mark.parametrize("top_k", [1, 2])
def test_distill_matches_exact_teacher_loss_and_gradient(logits, top_k):
    student = torch.tensor([[logits]], requires_grad=True)
    teacher = torch.zeros_like(student)
    values, indices = teacher.topk(top_k, dim=-1)
    actual = distillation_loss(student, indices, values, teacher.logsumexp(-1))
    expected = -(teacher.softmax(-1) * student.log_softmax(-1)).sum()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_distill_other_bucket_and_mask():
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn_like(student)
    values, indices = teacher.topk(2, dim=-1)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    actual = distillation_loss(student, indices, values, teacher.logsumexp(-1), valid_mask=mask)
    tp = teacher.softmax(-1).gather(-1, indices)
    sp = student.softmax(-1).gather(-1, indices)
    expected = (-(tp * sp.log()).sum(-1) - (1-tp.sum(-1)) * (1-sp.sum(-1)).log())[mask].sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, student, retain_graph=True)[0],
                               torch.autograd.grad(expected, student)[0])


class Tokenizer:
    name_or_path = "tiny"
    bos_token_id = 0

    def __len__(self):
        return 8

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(c) % 7 + 1 for c in text] for text in texts]}


class Teacher(torch.nn.Module):
    config = SimpleNamespace(vocab_size=8)

    def forward(self, x):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(x, 8).float() * 2)


def test_cache_alignment_vocabulary_and_corpus_identity(tmp_path):
    path = tmp_path / "train.bin"
    write_token_bin(["abcdefghijklmno"], Tokenizer(), path, log=False)
    prefix = tmp_path / "teacher"
    build_teacher_cache(Teacher(), path, prefix, seq_len=3, top_k=2,
                        num_blocks=3, device="cpu", dtype=torch.float32, log_every=0)
    cached = DistillTokenBinDataset(path, prefix, max_length=3, shuffle=False,
                                   infinite=False, student_vocab_size=8)
    baseline = TokenBinDataset(path, max_length=3, shuffle=False, infinite=False)
    rows = list(cached)
    assert len(rows) == 3
    for (x, y, index, value, lse), (bx, by) in zip(rows, baseline):
        torch.testing.assert_close(x, bx)
        torch.testing.assert_close(y, by)
        expected = Teacher()(x).logits.log_softmax(-1).gather(-1, index)
        torch.testing.assert_close(value-lse[:, None], expected, atol=.001, rtol=.001)
    with pytest.raises(ValueError, match="vocabulary"):
        DistillTokenBinDataset(path, prefix, max_length=3, student_vocab_size=7)
    write_token_bin(["ponmlkjihgfedcb"], Tokenizer(), path, log=False)
    with pytest.raises(ValueError, match="different token bin"):
        DistillTokenBinDataset(path, prefix, max_length=3)


def test_training_accumulation_matches_full_batch(tmp_path, monkeypatch):
    from examples import train_distill
    import sys

    tokenizer = Tokenizer()
    for split in ("train", "validation"):
        write_token_bin(["abcdefghijklmno"], tokenizer, tmp_path / f"{split}.bin", log=False)
    prefix = tmp_path / "teacher"
    build_teacher_cache(Teacher(), tmp_path / "train.bin", prefix, seq_len=3,
                        top_k=2, device="cpu", dtype=torch.float32, log_every=0)
    monkeypatch.setattr(sys, "argv", ["train_distill", "--teacher-cache", str(prefix)])
    args = train_distill.parse_args()
    args.corpus = str(tmp_path)
    args.seq_len = 3
    args.max_iters = 1
    args.hidden_size = 32
    args.intermediate_size = 32
    args.num_layers = 1
    args.num_heads = 2
    args.moe_num_experts = 0
    args.moe_experts_per_tok = 2
    args.num_workers = 0
    args.precision = "fp32"
    args.eval_batches = 1
    args.eval_every = 0
    args.warmup_iters = 0
    args.distill_alpha = .5
    args.logger = "none"
    monkeypatch.setattr(train_distill, "build_muon_adamw",
                        lambda model, **kwargs: torch.optim.SGD(model.parameters(), lr=.02))
    monkeypatch.setattr(train_distill, "parse_args", lambda: args)
    monkeypatch.setattr(train_distill, "get_tokenizer", lambda **kwargs: tokenizer)
    monkeypatch.setattr(train_distill, "get_token_bytes", lambda tokenizer, device: torch.ones(8, device=device))
    monkeypatch.setattr(train_distill.time, "sleep", lambda _: None)
    states = []
    for accum in (1, 4):
        args.batch_size = 4 // accum
        args.grad_accum_steps = accum
        args.checkpoint_path = str(tmp_path / f"student{accum}.pt")
        train_distill.run()
        states.append(torch.load(args.checkpoint_path, weights_only=False)["model"])
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], atol=2e-5, rtol=2e-4)
