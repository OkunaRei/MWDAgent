import numpy as np
import pytest

from src.evaluate import (
    classification_metrics,
    expected_calibration_error,
    prediction_gate,
    evaluate_gating,
    slice_classification_metrics,
)


def test_gate_supports_noncontiguous_labels_without_fake_probabilities():
    result = evaluate_gating(np.array([3, 8]), np.array([3, 3]),
                             np.ones(2), np.ones(2))
    assert result['accepted']['accuracy'] == 0.5


def test_gate_no_accepted_samples_has_undefined_performance():
    result = evaluate_gating(np.array([0]), np.array([0]),
                             np.zeros(1), np.zeros(1))
    assert result['accepted'] == {'support': 0, 'accuracy': None,
                                  'balanced_accuracy': None, 'macro_f1': None}


def test_gate_named_classes_require_encoded_labels():
    with pytest.raises(ValueError, match='class_names'):
        evaluate_gating(np.array([3, 8]), np.array([3, 3]),
                        np.ones(2), np.ones(2), class_names=['a', 'b'])


@pytest.mark.parametrize('labels,probabilities', [
    ([], np.empty((0, 2))), ([0], np.empty((1, 0))),
    ([[0]], [[0.5, 0.5]]), ([2], [[0.5, 0.5]]),
    ([-1], [[0.5, 0.5]]), ([0.5], [[0.5, 0.5]]),
])
def test_ece_rejects_invalid_labels_and_empty_samples(labels, probabilities):
    with pytest.raises(ValueError):
        expected_calibration_error(np.asarray(labels), np.asarray(probabilities))


@pytest.mark.parametrize('n_bins', [0, -1, 1.5, True])
def test_ece_requires_positive_integer_bin_count(n_bins):
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0]), np.array([[1., 0.]]), n_bins=n_bins)


def test_prediction_gate_requires_both_quality_and_confidence() -> None:
    assert prediction_gate(quality=0.9, confidence=0.95) == "accept"
    assert prediction_gate(quality=0.9, confidence=0.65) == "review"
    assert prediction_gate(quality=0.4, confidence=0.95) == "review"
    assert prediction_gate(quality=0.2, confidence=0.4) == "reject"


def test_prediction_gate_rejects_invalid_scores() -> None:
    with np.testing.assert_raises(ValueError):
        prediction_gate(quality=1.1, confidence=0.9)


def test_evaluate_gating_reports_selective_performance() -> None:
    result = evaluate_gating(
        labels=np.array([0, 1, 1, 0]),
        predictions=np.array([0, 0, 1, 1]),
        quality=np.array([0.95, 0.95, 0.9, 0.2]),
        confidence=np.array([0.95, 0.75, 0.9, 0.4]),
    )
    assert result["counts"] == {"accept": 2, "review": 1, "reject": 1}
    assert result["rates"]["accept"] == 0.5
    assert result["accepted"]["accuracy"] == 1.0


def test_expected_calibration_error_uses_confidence_bins() -> None:
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7],
    ])

    # All confidences fall in the single [0.5, 1.0] bin: accuracy .5,
    # mean confidence .75, so the weighted gap is .25.
    assert expected_calibration_error(labels, probabilities, n_bins=2) == 0.25


def test_expected_calibration_error_rejects_invalid_probabilities() -> None:
    with np.testing.assert_raises(ValueError):
        expected_calibration_error(np.array([0]), np.array([[1.2, -0.2]]))


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
    assert "expected_calibration_error" in metrics["ordinary"]
