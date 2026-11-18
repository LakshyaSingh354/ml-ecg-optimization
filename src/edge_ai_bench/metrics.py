"""Accuracy summaries and resampling intervals for multilabel outputs."""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_scores(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return macro AUROC and macro F1 at a 0.5 decision threshold.

    Every class must contain both labels. Callers that resample rows should
    skip draws that violate that condition so the average stays over a fixed
    class set.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if probabilities.shape != labels.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and labels must share a 2-D shape")
    if len(labels) == 0:
        raise ValueError("at least one sample is required")
    auc_values = []
    for index in range(labels.shape[1]):
        if len(np.unique(labels[:, index])) < 2:
            raise ValueError("each class must contain both labels")
        auc_values.append(roc_auc_score(labels[:, index], probabilities[:, index]))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return (
        float(np.mean(auc_values)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def bootstrap_macro_intervals(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap confidence intervals for macro AUROC and macro F1."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    point_auroc, point_f1 = macro_scores(probabilities, labels)
    generator = np.random.default_rng(seed)
    sample_count = len(labels)
    auroc_draws: list[float] = []
    f1_draws: list[float] = []
    skipped = 0
    for _ in range(resamples):
        index = generator.integers(0, sample_count, size=sample_count)
        try:
            auroc, f1 = macro_scores(probabilities[index], labels[index])
        except ValueError:
            skipped += 1
            continue
        auroc_draws.append(auroc)
        f1_draws.append(f1)
    if not auroc_draws:
        raise ValueError("bootstrap produced no resamples with both labels in every class")
    low_percentile = 100 * (alpha / 2)
    high_percentile = 100 * (1 - alpha / 2)
    auroc_array = np.asarray(auroc_draws)
    f1_array = np.asarray(f1_draws)
    return {
        "method": "percentile_row_bootstrap",
        "alpha": alpha,
        "resamples": resamples,
        "accepted_resamples": len(auroc_draws),
        "skipped_resamples": skipped,
        "seed": seed,
        "macro_auroc": point_auroc,
        "macro_f1": point_f1,
        "macro_auroc_ci95": [
            float(np.percentile(auroc_array, low_percentile)),
            float(np.percentile(auroc_array, high_percentile)),
        ],
        "macro_f1_ci95": [
            float(np.percentile(f1_array, low_percentile)),
            float(np.percentile(f1_array, high_percentile)),
        ],
    }
