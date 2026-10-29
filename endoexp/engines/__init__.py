"""Model engines. All read the same ``Query`` objects; heavy imports stay inside the engines.

- ``vllm``: letter distributions from next-token log-probabilities and greedy
  generation, with prefix caching so the fields of one image share its prefill;
- ``hf``: exact letter distributions and hidden-state features (for probes);
- ``fake``: deterministic stand-in used by the CPU tests and dry runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Query:
    key: str
    text: str
    images: list = field(default_factory=list)   # PIL images, already resized
    n_options: int = 0                            # multiple choice when > 0
    max_tokens: int = 0                           # generation when > 0


def create(kind: str, model_path: str | Path, **kwargs):
    if kind == "vllm":
        from endoexp.engines.vllm_engine import VLLMEngine
        return VLLMEngine(model_path, **kwargs)
    if kind == "hf":
        from endoexp.engines.hf_engine import HFEngine
        return HFEngine(model_path, **kwargs)
    if kind == "fake":
        from endoexp.engines.fake import FakeEngine
        return FakeEngine(model_path, **kwargs)
    raise ValueError(f"unknown engine {kind!r}")


def model_info(model_path: str | Path) -> dict:
    """Revision and architecture recorded next to a staged checkpoint."""
    import json
    path = Path(model_path)
    info = {"path": str(path), "exists": path.is_dir()}
    revision = path / "REVISION"
    if revision.is_file():
        info["revision"] = revision.read_text().strip()
    config = path / "config.json"
    if config.is_file():
        data = json.loads(config.read_text())
        info["model_type"] = data.get("model_type")
        info["architectures"] = data.get("architectures")
    return info
