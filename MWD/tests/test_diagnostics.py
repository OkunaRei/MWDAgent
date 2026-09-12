import json

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import rejection_records


def inputs():
    return dict(
        frame=pd.DataFrame({"a": [0., 2., np.nan], "b": [1., 0., np.inf],
                            "transition_zone": [False, True, False]}, index=[12, 4, 9]),
        columns=["a", "b"], lower=np.array([0., 0.]), upper=np.array([1., 1.]),
        labels=np.array([0, 1, 1]),
        probabilities=np.array([[.8, .2], [.1, .9], [.6, .4]]),
        class_names=["rock_a", "rock_b"], seed=42,
    )


def test_records_identify_correct_extra_rejection_and_nonfinite_features():
    data = inputs()
    original = data["frame"].copy(deep=True)
    arrays = {key: data[key].copy() for key in ("lower", "upper", "labels", "probabilities")}
    records = rejection_records(**data)
    assert records[0] == dict(row_index=12, seed=42, actual_class="rock_a",
        predicted_class="rock_a", correct=True, confidence=.8, quality=1.,
        confidence_accepted=True, dual_accepted=True, extra_rejected=False,
        transition_zone=False, anomalous_features=[])
    assert records[1]["correct"] is True
    assert records[1]["extra_rejected"] is True
    assert records[1]["anomalous_features"] == ["a"]
    assert records[1]["quality"] == .5
    assert records[2]["correct"] is False
    assert records[2]["confidence_accepted"] is False
    assert records[2]["extra_rejected"] is False
    assert records[2]["anomalous_features"] == ["a", "b"]
    json.dumps(records, allow_nan=False)
    pd.testing.assert_frame_equal(data["frame"], original)
    for key, before in arrays.items():
        np.testing.assert_array_equal(data[key], before)


def test_no_confidence_acceptance_and_no_errors_are_distinct():
    data = inputs()
    data["probabilities"] = np.array([[.6, .4], [.4, .6], [.4, .6]])
    records = rejection_records(**data)
    assert all(record["correct"] for record in records)
    assert not any(record["confidence_accepted"] for record in records)
    assert not any(record["extra_rejected"] for record in records)


def test_empty_validation_produces_no_records():
    data = inputs()
    data.update(frame=data["frame"].iloc[:0], labels=np.array([], dtype=int),
                probabilities=np.empty((0, 2)))
    assert rejection_records(**data) == []


@pytest.mark.parametrize("changes", [
    {"labels": [0, 1]}, {"labels": [[0], [1], [1]]}, {"labels": [0., 1., 1.]},
    {"labels": [0, 1, 2]}, {"probabilities": [[.5, .5]]},
    {"probabilities": [[.2, .2]] * 3}, {"probabilities": [[np.nan, .5]] * 3},
    {"probabilities": [[1.1, -.1]] * 3}, {"probabilities": [.5, .5, .5]},
    {"class_names": ["one"]}, {"class_names": ["same", "same"]},
    {"class_names": [["one"], ["two"]]}, {"columns": ["a", "a"]},
    {"columns": ["missing"]}, {"lower": [0.]},
    {"quality_threshold": np.nan}, {"confidence_threshold": 1.1},
    {"seed": 1.5},
])
def test_rejects_invalid_vectors_and_metadata(changes):
    with pytest.raises(ValueError):
        rejection_records(**{**inputs(), **changes})


@pytest.mark.parametrize("indices", [[1, 1, 2], [1.5, 2.5, 3.5], ["a", "b", "c"]])
def test_requires_unique_integer_source_indices(indices):
    data = inputs()
    data["frame"] = data["frame"].set_axis(indices)
    with pytest.raises(ValueError):
        rejection_records(**data)


def test_requires_boolean_transition_labels():
    data = inputs()
    data["frame"] = data["frame"].assign(transition_zone=["False", "True", "False"])
    with pytest.raises(ValueError):
        rejection_records(**data)
