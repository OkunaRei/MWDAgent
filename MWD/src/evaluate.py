from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def prediction_gate(
    *,
    quality: float,
    confidence: float,
    min_quality: float = 0.7,
    min_confidence: float = 0.8,
    reject_quality: float = 0.3,
    reject_confidence: float = 0.5,
) -> str:
    """Apply the quality-confidence dual gate used by downstream decisions."""
    scores = (quality, confidence, min_quality, min_confidence, reject_quality, reject_confidence)
    if any(not np.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("gate scores and thresholds must be finite values in [0, 1]")
    if min_quality < reject_quality or min_confidence < reject_confidence:
        raise ValueError("accept thresholds must be at least reject thresholds")
    if quality >= min_quality and confidence >= min_confidence:
        return "accept"
    if quality < reject_quality and confidence < reject_confidence:
        return "reject"
    return "review"


def evaluate_gating(
    labels: np.ndarray,
    predictions: np.ndarray,
    quality: np.ndarray,
    confidence: np.ndarray,
    *,
    class_names: list[str] | None = None,
    **gate_kwargs: float,
) -> dict[str, Any]:
    """Summarize selective performance after applying the dual gate."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    quality, confidence = np.asarray(quality, dtype=float), np.asarray(confidence, dtype=float)
    size = labels.size
    if labels.ndim != 1 or size == 0 or any(values.shape != labels.shape for values in (predictions, quality, confidence)):
        raise ValueError("labels, predictions, quality, and confidence must be non-empty and aligned")
    if class_names is not None and any(
        not np.issubdtype(values.dtype, np.integer)
        or (values < 0).any() or (values >= len(class_names)).any()
        for values in (labels, predictions)
    ):
        raise ValueError("class_names requires integer labels in [0, number of classes)")
    decisions = np.array(
        [prediction_gate(quality=float(q), confidence=float(c), **gate_kwargs) for q, c in zip(quality, confidence)]
    )
    counts = {name: int((decisions == name).sum()) for name in ("accept", "review", "reject")}
    rates = {name: count / size for name, count in counts.items()}
    accepted = decisions == "accept"
    accepted_metrics: dict[str, Any] = {"support": counts["accept"]}
    if accepted.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            accepted_metrics.update({
                "accuracy": float(accuracy_score(labels[accepted], predictions[accepted])),
                "balanced_accuracy": float(balanced_accuracy_score(labels[accepted], predictions[accepted])),
                "macro_f1": float(f1_score(labels[accepted], predictions[accepted],
                    labels=np.arange(len(class_names)) if class_names is not None else None,
                    average="macro", zero_division=0)),
            })
    else:
        accepted_metrics.update({"accuracy": None, "balanced_accuracy": None, "macro_f1": None})
    return {"counts": counts, "rates": rates, "accepted": accepted_metrics}


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return the confidence calibration gap using fixed-width bins."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0] or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a 2D array aligned with labels")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("labels must be integer probability-column indices")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to 1")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(confidence, edges[1:-1], right=False), n_bins - 1)
    return float(
        sum(
            (mask.sum() / len(labels)) * abs(correct[mask].mean() - confidence[mask].mean())
            for bin_index in range(n_bins)
            if (mask := bins == bin_index).any()
        )
    )


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    class_indices = np.arange(len(class_names))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        result: dict[str, Any] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
            "expected_calibration_error": expected_calibration_error(labels, probabilities),
            "classification_report": _classification_report(labels, predictions, class_indices, class_names),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=class_indices).tolist(),
        }
    if np.unique(labels).size < len(class_names):
        result["roc_auc_ovr"] = None
    else:
        try:
            auc = roc_auc_score(labels, probabilities, multi_class="ovr", labels=class_indices)
            result["roc_auc_ovr"] = float(auc) if np.isfinite(auc) else None
        except ValueError:
            result["roc_auc_ovr"] = None
    return result


def _classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_indices: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return classification_report(
            labels,
            predictions,
            labels=class_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )


def slice_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    transition_zone: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Evaluate ordinary and geologic-transition samples without retraining."""
    result: dict[str, dict[str, Any]] = {}
    for name, mask in {
        "ordinary": ~transition_zone,
        "transition_zone": transition_zone,
    }.items():
        support = int(mask.sum())
        if support == 0:
            raise ValueError(f"Slice {name} has no samples")
        metrics = classification_metrics(labels[mask], predictions[mask], probabilities[mask], class_names)
        metrics["support"] = support
        result[name] = metrics
    return result
