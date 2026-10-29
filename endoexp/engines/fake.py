"""Deterministic stand-in engine for CPU tests and dry runs.

Its answers depend on simple image statistics, so image-specific contrasts,
views and probes all have something to measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from endoexp.engines import Query
from endoexp.fields import FIELDS
from endoexp.io import unit_hash


def _stats(images) -> np.ndarray:
    if not images:
        return np.zeros(6, dtype=np.float64)
    arr = np.asarray(images[0].convert("RGB"), dtype=np.float64) / 255.0
    return np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])


class FakeEngine:
    def __init__(self, model_path: str | Path, **_: object):
        self.model_path = str(model_path)
        self.sharpness = 2.0 + 6.0 * unit_hash("sharpness", self.model_path)

    def score(self, queries: list[Query]) -> dict[str, dict]:
        results = {}
        for query in queries:
            n = query.n_options
            stats = _stats(query.images)
            logits = np.array([np.sin(3.1 * (i + 1) * stats.sum() + i) for i in range(n)])
            logits = self.sharpness * logits if query.images else 0.1 * logits
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            results[query.key] = {"probs": probs.tolist(), "coverage": 0.9, "top_token": "A"}
        return results

    def generate(self, queries: list[Query]) -> dict[str, str]:
        results = {}
        for query in queries:
            stats = _stats(query.images)
            lines = []
            for name, field in FIELDS.items():
                if f"{field.label}:" in query.text:
                    index = int(abs(stats.sum() * 1000 + len(name))) % len(field.options)
                    lines.append(f"{field.label}: {field.options[index]}")
            results[query.key] = "\n".join(lines)
        return results

    def features(self, queries: list[Query]) -> dict[str, dict[str, np.ndarray]]:
        results = {}
        for query in queries:
            stats = _stats(query.images)
            noise = np.array([unit_hash(query.key, i) for i in range(10)])
            vector = np.concatenate([stats, stats ** 2, noise]).astype(np.float16)
            results[query.key] = {"L0_image": vector, "L0_last": vector[::-1].copy()}
        return results

    def close(self) -> None:
        pass
