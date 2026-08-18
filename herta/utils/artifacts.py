"""Helpers for persistent HERTA result artifacts."""

from __future__ import annotations

from pathlib import Path


def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    """Create the stable directories used for reusable HERTA results."""

    output_dir = Path(output_dir)
    paths = {
        "node_embeddings": output_dir / "node_embeddings",
        "edge_scores": output_dir / "edge_scores",
        "grn": output_dir / "grn",
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "embeddings": output_dir / "embeddings",
        "clustering": output_dir / "clustering",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = ["ensure_output_tree"]
