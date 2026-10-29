import numpy as np

from endoexp.metrics import (average_ranks, binary_auc, clustered_bootstrap, macro_f1, quadratic_kappa,
                             summarize)
from endoexp.probe import ridge_probe


def test_ranks_and_auc():
    assert average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert binary_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert binary_auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert binary_auc(np.array([0.5, 0.4]), np.array([1, 1])) is None


def test_f1_and_kappa():
    y = np.array([0, 1, 2, 2])
    assert macro_f1(y, y) == 1.0
    assert quadratic_kappa(y, y, 3) == 1.0
    assert quadratic_kappa(y, np.array([2, 1, 0, 0]), 3) < 0


def test_summarize():
    labels = [0, 1, 1, 0]
    probs = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]
    result = summarize(labels, probs, ordinal=True)
    assert result["n"] == 4 and result["acc"] == 0.75 and result["auc"] == 1.0
    assert abs(result["p_true"] - (0.9 + 0.8 + 0.4 + 0.7) / 4) < 1e-9


def test_clustered_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, 400)
    groups = np.repeat(np.arange(100), 4)
    result = clustered_bootstrap(values, groups)
    assert result["lo"] < result["mean"] < result["hi"] and result["n"] == 400
    assert clustered_bootstrap([], [])["mean"] is None


def test_ridge_probe_learns_a_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 40))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    result = ridge_probe(x[:200], y[:200], x[200:], y[200:], 2)
    assert result["acc"] > 0.85 and result["n_test"] == 100
