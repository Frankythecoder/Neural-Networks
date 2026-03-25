from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import setup_logging

logger = setup_logging()


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs", self.counter, self.patience
            )

        return self.counter >= self.patience


class ModelCheckpoint:
    """Save best and last model checkpoints."""

    def __init__(self, save_dir: str = "checkpoints", mode: str = "max") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best: float | None = None

    def step(
        self,
        model: nn.Module,
        metric: float,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save last checkpoint; save best if metric improved."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
            **(extra or {}),
        }

        # Always save last
        torch.save(state, self.save_dir / "last.pt")

        # Save best
        if self.best is None or (
            metric > self.best if self.mode == "max" else metric < self.best
        ):
            self.best = metric
            torch.save(state, self.save_dir / "best.pt")
            logger.info("Saved best checkpoint (metric=%.4f) at epoch %d", metric, epoch)
