"""Checkpoint helpers for stable two-stage model contracts."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, **payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)
