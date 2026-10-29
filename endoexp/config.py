"""Suite and model configuration (TOML)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from endoexp.experiments import DEFAULT_CAPS, EXPERIMENTS

ENGINE_DEFAULTS = {
    "vllm": {"max_model_len": 8192, "gpu_memory_utilization": 0.85, "max_logprobs": 20,
             "enforce_eager": True, "batch_size": 256},
    "hf": {},
    "fake": {},
}


@dataclass
class Unit:
    """One model under one engine running a list of experiments, sharded over GPUs."""
    model: str
    engine: str
    experiments: list[str]
    priority: int = 0
    pilot: bool = False
    shards: int | str = "auto"     # "auto": one shard per GPU of the job
    gpus: int = 1                  # GPUs per shard (tensor parallel size for vllm)
    timeout_minutes: float | None = None
    engine_args: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    seed: int
    sources: list[str]
    max_pixels: int
    units: list[Unit]
    caps: dict = field(default_factory=dict)
    pilot: dict = field(default_factory=dict)
    checks: list[str] = field(default_factory=lambda: ["tokenizers", "nccl"])
    report_tokens: int = 256
    chunk: int = 512
    description: str = ""


def load_suite(path: str | Path) -> Suite:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    units = [Unit(**unit) for unit in data.pop("units")]
    for unit in units:
        unknown = set(unit.experiments) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"unit {unit.model}: unknown experiments {sorted(unknown)}")
        if unit.engine not in ENGINE_DEFAULTS:
            raise ValueError(f"unit {unit.model}: unknown engine {unit.engine!r}")
    suite = Suite(units=units, **data)
    unknown_checks = set(suite.checks) - {"tokenizers", "nccl"}
    if unknown_checks:
        raise ValueError(f"unknown checks {sorted(unknown_checks)}")
    suite.caps = {**DEFAULT_CAPS, **suite.caps}
    suite.pilot = {"items_per_experiment": 16, "limit_per_source": 300, "task_timeout_minutes": 8,
                   **suite.pilot}
    return suite


def load_models(path: str | Path) -> dict[str, dict]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    models = data.get("models", {})
    for key, entry in models.items():
        missing = {"repo", "dir"} - set(entry)
        if missing:
            raise ValueError(f"model {key}: missing {sorted(missing)}")
    return models
