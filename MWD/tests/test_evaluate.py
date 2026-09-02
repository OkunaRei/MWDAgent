import numpy as np

from src.evaluate import classification_metrics, slice_classification_metrics


def test_slice_metrics_preserve_all_classes_when_slice_lacks_a_class() -> None:
    labels = np.array([0, 1, 1])
    predictions = np.array([0, 1, 0])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )

    metrics = classification_metrics(labels, predictions, probabilities, ["A", "B", "C"])

    assert len(metrics["confusion_matrix"]) == 3
    assert "C" in metrics["classification_report"]
    assert metrics["roc_auc_ovr"] is None


def test_slice_metrics_record_support_and_subset_metrics() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.1, 0.9],
        ]
    )
    transition_zone = np.array([False, True, True, False])

    metrics = slice_classification_metrics(
        labels,
        predictions,
        probabilities,
        ["A", "B"],
        transition_zone,
    )

    assert metrics["ordinary"]["support"] == 2
    assert metrics["transition_zone"]["support"] == 2
    assert metrics["transition_zone"]["accuracy"] == 0.5
