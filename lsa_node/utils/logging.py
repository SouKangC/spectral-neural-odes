"""Logging + checkpoint utilities.

Light wrapper around ``torch.utils.tensorboard.SummaryWriter`` plus a
plain JSONL fallback (TB is optional). Includes a small checkpoint
helper that atomically writes ``ckpt.pt`` next to the metrics file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------
class RunLogger:
    """Writes scalars to TensorBoard *and* a JSONL fallback, plus a single
    ``hparams.json`` so a run is self-describing on disk.

    Layout:

        <run_dir>/
          ├── hparams.json
          ├── metrics.jsonl       (every step appended)
          ├── checkpoints/
          │     └── ckpt.pt
          └── tb/                 (only if tensorboard is importable)

    Use as:

        log = RunLogger(out_dir, hparams=cfg)
        log.scalar("train/loss", 0.42, step=5)
        log.save_checkpoint(model, optim, step=5, extra={"epoch": 1})
        log.close()
    """

    def __init__(self, run_dir: str | os.PathLike, hparams: Mapping[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.metrics_fp = (self.run_dir / "metrics.jsonl").open("a", buffering=1)

        if hparams is not None:
            (self.run_dir / "hparams.json").write_text(json.dumps(_to_jsonable(hparams), indent=2))

        self._tb: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: WPS433
            self._tb = SummaryWriter(str(self.run_dir / "tb"))
        except Exception:  # noqa: BLE001
            self._tb = None  # TensorBoard not installed — JSONL only.

    # ---- scalars ----
    def scalar(self, tag: str, value: float, step: int) -> None:
        rec = {"step": step, "tag": tag, "value": float(value)}
        self.metrics_fp.write(json.dumps(rec) + "\n")
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

    def scalars(self, mapping: Mapping[str, float], step: int) -> None:
        for k, v in mapping.items():
            self.scalar(k, v, step)

    # ---- checkpoints ----
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically write a checkpoint to ``<run_dir>/checkpoints/ckpt.pt``."""
        target = self.run_dir / "checkpoints" / "ckpt.pt"
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": dict(extra) if extra is not None else None,
        }
        # Write to tmp then rename so a crash mid-write can't corrupt the file.
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".ckpt-", suffix=".pt", delete=False
        ) as fp:
            torch.save(state, fp.name)
            tmp = Path(fp.name)
        shutil.move(str(tmp), str(target))
        return target

    @staticmethod
    def load_checkpoint(
        path: str | os.PathLike,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Restore weights (and optionally optimizer state) in place.
        Returns the saved ``extra`` dict (or ``{}``) plus the ``step``."""
        state = torch.load(str(path), map_location=map_location, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        return {"step": int(state.get("step", 0)), "extra": state.get("extra") or {}}

    # ---- lifecycle ----
    def close(self) -> None:
        self.metrics_fp.close()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return repr(obj)
