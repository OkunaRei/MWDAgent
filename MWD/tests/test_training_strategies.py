from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import benchmark


@pytest.fixture
def training_frame():
    frame = pd.DataFrame({f"feature_{i}": np.arange(40) + i for i in range(48)})
    return frame.assign(Rock=["A"] * 24 + ["B"] * 16, transition_zone=[False, True] * 20)


class RecordingModel:
    def __init__(self):
        self.params = {}
        self.fit_features = None
        self.fit_labels = None
        self.validation_indices = []

    def set_params(self, **params):
        assert self.fit_features is None
        self.params = {**self.params, **params}
        return self

    def fit(self, features, labels):
        self.fit_features = features.copy(deep=True)
        self.fit_labels = labels.copy(deep=True)
        return self

    def predict(self, features):
        self.validation_indices.append(features.index.tolist())
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        self.validation_indices.append(features.index.tolist())
        return np.tile([0.6, 0.4], (len(features), 1))


@pytest.mark.parametrize("strategy", [None, "paper_smote", "original", "class_weight", "midpoint_smote", "real_resample"])
def test_strategy_changes_training_only_and_records_counts(monkeypatch, training_frame, strategy):
    expected_train, expected_validation = benchmark.split_train_validation(
        training_frame, validation_size=0.25, seed=11
    )
    original_frame = training_frame.copy(deep=True)
    model = RecordingModel()
    sampling_calls = []

    def resample(features, labels, *, random_state):
        sampling_calls.append((features.copy(deep=True), labels.copy(deep=True), random_state))
        minority_index = labels.index[labels.eq(1)][0]
        return (
            pd.concat([features, features.loc[[minority_index]]], ignore_index=True),
            pd.concat([labels, labels.loc[[minority_index]]], ignore_index=True),
        )

    monkeypatch.setattr(benchmark, "rebalance_training_set", resample)
    monkeypatch.setattr(benchmark, "undersample_majority_then_smote", resample)
    monkeypatch.setattr(benchmark, "rebalance_real_samples", resample)
    monkeypatch.setattr(benchmark, "_make_model", lambda name, seed: model)
    kwargs = {} if strategy is None else {"training_strategy": strategy}
    result = benchmark.run_validation_once(
        training_frame, model_name="lightgbm", validation_size=0.25, seed=11, include_diagnostics=True, **kwargs
    )

    effective_strategy = strategy or "paper_smote"
    assert result["training_strategy"] == effective_strategy
    evidence = result['diagnostics']
    assert evidence['classes'] == ['A', 'B']
    assert evidence['validation_indices'] == expected_validation.index.tolist()
    assert np.asarray(evidence['overall']['confusion_matrix']).sum() == len(expected_validation)
    assert evidence['overall']['classification_report']['B']['recall'] == 0
    assert result["original_train_class_counts"] == {"A": 18, "B": 12}
    assert result["fitted_train_class_counts"] == {
        "A": 18, "B": 13 if effective_strategy in ("paper_smote", "midpoint_smote", "real_resample") else 12
    }
    assert result["balanced_train_rows"] == (31 if effective_strategy in ("paper_smote", "midpoint_smote", "real_resample") else 30)
    assert model.validation_indices == [expected_validation.index.tolist()] * 2
    if effective_strategy in ("paper_smote", "midpoint_smote", "real_resample"):
        assert len(sampling_calls) == 1
        features, labels, seed = sampling_calls[0]
        pd.testing.assert_frame_equal(features, expected_train[benchmark.feature_columns(training_frame)])
        assert labels.index.tolist() == expected_train.index.tolist()
        assert set(features.index).isdisjoint(expected_validation.index)
        assert seed == 11
    else:
        assert sampling_calls == []
        pd.testing.assert_frame_equal(
            model.fit_features, expected_train[benchmark.feature_columns(training_frame)]
        )
        assert model.fit_labels.index.tolist() == expected_train.index.tolist()
    assert model.params == ({"class_weight": "balanced"} if strategy == "class_weight" else {})
    pd.testing.assert_frame_equal(training_frame, original_frame)


def test_invalid_strategy_fails_before_split_or_fit(monkeypatch, training_frame):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid strategy must fail before splitting, sampling, or model construction")

    for name in ("split_train_validation", "rebalance_training_set", "_make_model"):
        monkeypatch.setattr(benchmark, name, unexpected_call)
    with pytest.raises(ValueError, match="Unsupported training strategy"):
        benchmark.run_validation_once(
            training_frame, model_name="lightgbm", validation_size=0.25,
            seed=11, training_strategy="unknown",
        )
