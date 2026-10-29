"""Metrics for closed-set fields (numpy only)."""

from __future__ import annotations

import math

import numpy as np


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    bounds = np.flatnonzero(np.concatenate(([True], sorted_x[1:] != sorted_x[:-1], [True])))
    for start, end in zip(bounds[:-1], bounds[1:]):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return ranks


def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    """One-vs-rest AUC averaged over classes that have positives and negatives."""
    values = [binary_auc(probs[:, c], labels == c) for c in range(probs.shape[1])]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    scores = []
    for c in np.union1d(labels, preds):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def quadratic_kappa(labels: np.ndarray, preds: np.ndarray, k: int) -> float | None:
    if k < 2 or len(labels) == 0:
        return None
    observed = np.zeros((k, k))
    np.add.at(observed, (labels, preds), 1)
    weights = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return None
    return float(1 - (weights * observed).sum() / denominator)


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


def summarize(labels, probs, ordinal: bool = False) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(1)
    p_true = probs[np.arange(len(labels)), labels]
    result = {
        "n": int(len(labels)),
        "acc": float((preds == labels).mean()),
        "macro_f1": macro_f1(labels, preds),
        "auc": macro_auc(labels, probs),
        "p_true": float(p_true.mean()),
        "nll": float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
        "ece": expected_calibration_error(labels, probs),
    }
    if ordinal:
        result["qwk"] = quadratic_kappa(labels, preds, probs.shape[1])
    return result


def clustered_bootstrap(values, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of per-item values with a 95% interval that resamples whole groups."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    keys, index = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=len(keys))
    counts = np.bincount(index, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values))}


def fmt(value, digits: int = 1, percent: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{100 * value:.{digits}f}" if percent else f"{value:.{digits + 2}f}"
