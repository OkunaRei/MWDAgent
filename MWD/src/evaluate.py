from __future__ import annotations

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


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    class_indices = np.arange(len(class_names))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
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
