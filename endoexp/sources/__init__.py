"""Dataset adapters. Each reads a prepared folder and returns ``Record`` objects.

Adapters never download anything; ``tools/prepare_data.py`` creates the
folders outside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Record:
    source: str
    key: str
    image: str                      # path relative to the data root
    size: tuple[int, int]           # (width, height) of that image
    labels: dict[str, str]
    regions: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.key}"


def _loaders() -> dict[str, Callable[..., tuple[list[Record], dict]]]:
    from endoexp.sources import hyperkvasir, kvasir_vqa
    return {
        "kvasir_vqa": kvasir_vqa.load,
        "hyperkvasir": hyperkvasir.load_labeled,
        "hyperkvasir_seg": hyperkvasir.load_segmented,
    }


SOURCE_NAMES = ("kvasir_vqa", "hyperkvasir", "hyperkvasir_seg")


def load(name: str, data_root: str | Path, limit: int | None = None) -> tuple[list[Record], dict]:
    loaders = _loaders()
    if name not in loaders:
        raise KeyError(f"unknown source {name!r}; known: {sorted(loaders)}")
    return loaders[name](Path(data_root), limit=limit)
