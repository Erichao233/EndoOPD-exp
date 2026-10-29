"""Linear probes (ridge classifier in dual form) on frozen features."""

from __future__ import annotations

import numpy as np

from endoexp.metrics import macro_f1

LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def _fit_predict(train: np.ndarray, targets: np.ndarray, query: np.ndarray, lam: float) -> np.ndarray:
    scale = train.shape[1]
    kernel = train @ train.T / scale
    mean = targets.mean(0)
    alpha = np.linalg.solve(kernel + lam * np.eye(len(train)), targets - mean)
    return query @ train.T / scale @ alpha + mean


def ridge_probe(x_train, y_train, x_test, y_test, k: int, seed: int = 0) -> dict:
    """Standardize on train, pick lambda on a 20% train holdout, refit, score test."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    mean, std = x_train.mean(0), x_train.std(0)
    std[std < 1e-6] = 1.0
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    onehot = np.eye(k)[y_train]

    order = np.random.default_rng(seed).permutation(len(x_train))
    cut = int(0.8 * len(order))
    best_lam, best_acc = LAMBDAS[len(LAMBDAS) // 2], -1.0
    if cut >= 10 and len(order) - cut >= 5:
        fit_idx, val_idx = order[:cut], order[cut:]
        for lam in LAMBDAS:
            pred = _fit_predict(x_train[fit_idx], onehot[fit_idx], x_train[val_idx], lam).argmax(1)
            acc = float((pred == y_train[val_idx]).mean())
            if acc > best_acc:
                best_lam, best_acc = lam, acc
    pred = _fit_predict(x_train, onehot, x_test, best_lam).argmax(1)
    majority = np.bincount(y_train, minlength=k).argmax()
    return {
        "acc": float((pred == y_test).mean()), "macro_f1": macro_f1(y_test, pred),
        "majority_acc": float((y_test == majority).mean()), "lambda": best_lam,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)), "pred": pred,
    }
