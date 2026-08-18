"""HERTA-ctGRN package with SIMBA-style public namespaces."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from herta._settings import settings


class _LazyNamespace:
    """Lightweight proxy that imports a namespace module on first use."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"<lazy namespace {self._module_name}>"


pp = _LazyNamespace("herta.pp")
tl = _LazyNamespace("herta.tl")
pl = _LazyNamespace("herta.pl")
evaluate = _LazyNamespace("herta.evaluate")

__version__ = "0.1.0"

__all__ = ["settings", "pp", "tl", "pl", "evaluate"]
