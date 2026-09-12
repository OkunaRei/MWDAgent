from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def fit_quality_bounds(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Tukey fences on raw training features only, without labels.

    This statistical anomaly proxy does not establish sensor or field quality.
    Constant training columns require exact equality during scoring.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or 0 in features.shape or not np.isfinite(features).all():
        raise ValueError("training features must be a nonempty finite 2D matrix")
    q1, q3 = np.quantile(features, [.25, .75], axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        width = 1.5 * (q3 - q1)
        lower, upper = q1 - width, q3 + width
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("training feature range overflows finite quality bounds")
    return lower, upper


def quality_scores(features: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return each row's fraction of finite features inside frozen fences."""
    features = np.asarray(features, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("features must be a 2D matrix with at least one column")
    if lower.shape != (features.shape[1],) or upper.shape != lower.shape:
        raise ValueError("quality bounds must be one-dimensional and match feature columns")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
        raise ValueError("quality bounds must be finite and ordered")
    valid = np.isfinite(features) & (features >= lower) & (features <= upper)
    return valid.mean(axis=1)


def _validate_ablation(
    labels: np.ndarray, probabilities: np.ndarray, quality: np.ndarray,
    quality_threshold: float, confidence_threshold: float,
) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a matrix with at least two classes")
    if labels.shape != (len(probabilities),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be aligned one-dimensional integer class indices")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("labels must refer to probability columns")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("each probability row must sum to 1")
    if quality.shape != labels.shape or not np.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
        raise ValueError("quality must be aligned finite scores in [0, 1]")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (quality_threshold, confidence_threshold)):
        raise ValueError("thresholds must be finite values in [0, 1]")


def _selection_metrics(
    labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray, class_count: int,
) -> dict[str, Any]:
    support, accepted = len(labels), int(mask.sum())
    accuracy = float((labels[mask] == predictions[mask]).mean()) if accepted else None
    return {
        "support": support,
        "accepted": accepted,
        "reviewed": support - accepted,
        "coverage": accepted / support if support else None,
        "risk": 1 - accuracy if accuracy is not None else None,
        "accuracy": accuracy,
        "macro_f1": float(f1_score(labels[mask], predictions[mask], labels=np.arange(class_count),
                                   average="macro", zero_division=0)) if accepted else None,
        "class_support": np.bincount(labels, minlength=class_count).tolist(),
        "accepted_class_support": np.bincount(labels[mask], minlength=class_count).tolist(),
    }


def gating_ablation(
    labels: np.ndarray, probabilities: np.ndarray, quality: np.ndarray, *,
    quality_threshold: float = .9, confidence_threshold: float = .8,
) -> dict[str, dict[str, Any]]:
    """Compare frozen binary gates; every unaccepted sample requires review.

    Empty selections have undefined risk, accuracy and F1. Probability columns
    define the fixed class universe, including classes absent from a selection.
    """
    labels = np.asarray(labels)
    probabilities, quality = np.asarray(probabilities, dtype=float), np.asarray(quality, dtype=float)
    _validate_ablation(labels, probabilities, quality, quality_threshold, confidence_threshold)
    predictions = probabilities.argmax(axis=1)
    good_quality = quality >= quality_threshold
    confident = probabilities.max(axis=1) >= confidence_threshold
    selections = {
        "no_gate": np.ones(len(labels), dtype=bool),
        "quality_only": good_quality,
        "confidence_only": confident,
        "dual": good_quality & confident,
    }
    return {name: _selection_metrics(labels, predictions, mask, probabilities.shape[1])
            for name, mask in selections.items()}
