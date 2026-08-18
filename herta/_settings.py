"""Runtime settings following SIMBA's lightweight global config pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HertaConfig:
    """Global runtime configuration for convenience APIs and plotting."""

    workdir: Path = Path("runs")
    save_fig: bool = False
    figdir: Path = Path("figures")
    figure_params: dict[str, Any] = field(default_factory=dict)
    train_params: dict[str, Any] = field(default_factory=dict)

    def set_workdir(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir)

    def set_figure_params(self, save_fig: bool | None = None, figdir: str | Path | None = None, **kwargs: Any) -> None:
        if save_fig is not None:
            self.save_fig = save_fig
        if figdir is not None:
            self.figdir = Path(figdir)
        self.figure_params.update(kwargs)

    def set_train_params(self, config: dict[str, Any]) -> None:
        self.train_params.update(config)


settings = HertaConfig()
