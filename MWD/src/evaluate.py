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
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": classification_report(
            labels, predictions, target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    try:
        result["roc_auc_ovr"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", labels=np.arange(len(class_names)))
        )
    except ValueError:
        result["roc_auc_ovr"] = None
    return result
