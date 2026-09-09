"""Examples resolve caller paths and retain all header fields."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from examples import scaling_laws, train_status


def test_status_reads_params_and_horizon_from_same_header(tmp_path):
    path = tmp_path / "train.log"
    path.write_text("params=1,234 max_iters=500\n")
    result = train_status.parse_log(path)
    assert result["params"] == 1234
    assert result["total_iters"] == 500


def test_sweep_resolves_paths_from_caller_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus").mkdir()
    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setattr(sys, "argv", ["scaling_laws", "run", "--dataset", "corpus",
                                     "--tokenizer", "tokenizer", "--results-dir", "results"])
    args = scaling_laws.parse_args()
    monkeypatch.setattr(scaling_laws, "get_tokenizer", lambda **kwargs: range(args.vocab_size))
    plan = scaling_laws.plan_scaling_run(2, vocab_size=args.vocab_size, flops_budget=1e10,
                                       sequence_length=8, device_batch_size=1, world_size=1,
                                       total_batch_size=8, aspect_ratio=32, head_dim=64,
                                       ffn_multiple_of=64, reference_batch_size=8)
    calls = []

    def training(command, cwd, check):
        def arg(name):
            return command[command.index(name) + 1]
        assert Path(arg("--dataset")) == tmp_path / "corpus"
        assert Path(arg("--tokenizer")) == tmp_path / "tokenizer"
        assert Path(arg("--token-bytes-cache")) == tmp_path / "results/token_bytes.pt"
        output = Path(arg("--result-json"))
        assert output.is_absolute()
        result = {**asdict(plan), "optimizer": args.optimizer, "optimizer_muon": 1.,
                  "initialization_nanochat": 1., "val_bpb": 2.}
        output.write_text(json.dumps(result))
        calls.append(output)

    monkeypatch.setattr(scaling_laws.subprocess, "run", training)
    scaling_laws.run_sweep(args, [plan])
    assert len(calls) == 1
    assert (tmp_path / "results/results.csv").exists()
