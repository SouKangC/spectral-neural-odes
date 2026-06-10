"""Tests for RunLogger checkpoint round-trip and JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lsa_node.utils.logging import RunLogger


def test_runlogger_writes_jsonl_and_hparams(tmp_path: Path) -> None:
    cfg = {"lr": 1e-3, "batch_size": 4, "model": {"name": "lsa_node"}}
    log = RunLogger(tmp_path / "run", hparams=cfg)
    log.scalar("train/loss", 1.5, step=0)
    log.scalar("train/loss", 1.0, step=1)
    log.scalars({"train/lr": 1e-3, "train/acc": 0.5}, step=1)
    log.close()

    hp = json.loads((tmp_path / "run" / "hparams.json").read_text())
    assert hp["lr"] == 1e-3
    assert hp["model"]["name"] == "lsa_node"

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()
    recs = [json.loads(l) for l in lines]
    assert {"step": 0, "tag": "train/loss", "value": 1.5} in recs
    assert any(r["tag"] == "train/lr" and r["step"] == 1 for r in recs)


def test_runlogger_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with RunLogger(tmp_path / "run") as log:
        # take a step so opt has state
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        path = log.save_checkpoint(model, opt, step=42, extra={"epoch": 3})

    model2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    restored = RunLogger.load_checkpoint(path, model2, opt2)

    assert restored["step"] == 42
    assert restored["extra"] == {"epoch": 3}
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1.data, p2.data)
