import numpy as np
import pytest

from edge_ai_bench.metrics import bootstrap_macro_intervals, macro_scores


def test_macro_scores_perfect_ranking():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    auroc, f1 = macro_scores(probabilities, labels)
    assert auroc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_bootstrap_intervals_bracket_point_estimates():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9], [0.8, 0.2], [0.15, 0.85]])
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    summary = bootstrap_macro_intervals(probabilities, labels, resamples=200, seed=7)
    low, high = summary["macro_auroc_ci95"]
    assert low <= summary["macro_auroc"] <= high