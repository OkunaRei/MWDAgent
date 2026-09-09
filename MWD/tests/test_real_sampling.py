import numpy as np
import pandas as pd
import pytest

from src.sampling import (
    rebalance_real_samples,
    rebalance_training_set,
    undersample_majority_then_smote,
)


def sample_data(counts=(8, 4, 2)):
    labels = pd.Series(
        [label for label, count in zip("ABC", counts) for _ in range(count)], name="Rock"
    )
    features = pd.DataFrame({"signal": np.arange(len(labels), dtype=float),
                             "other": np.arange(len(labels), dtype=float) ** 2})
    return features, labels


def test_real_samples_match_paper_counts_and_preserve_actual_pairs():
    features, labels = sample_data()
    features.index = labels.index = pd.Index(range(100, 114))
    x, y = rebalance_real_samples(features, labels, random_state=7)
    _, paper_y = rebalance_training_set(features, labels, random_state=7)
    assert y.value_counts().to_dict() == paper_y.value_counts().to_dict()
    actual_pairs = set(zip(features.signal, features.other, labels))
    assert set(zip(x.signal, x.other, y)) <= actual_pairs
    assert x.index.equals(y.index)
    assert x.columns.equals(features.columns)
    assert y.name == labels.name


@pytest.mark.parametrize("sampler", [rebalance_real_samples, undersample_majority_then_smote])
def test_sampling_is_reproducible_and_does_not_mutate(sampler):
    features, labels = sample_data()
    saved_x, saved_y = features.copy(deep=True), labels.copy(deep=True)
    first_x, first_y = sampler(features, labels, random_state=7)
    second_x, second_y = sampler(features, labels, random_state=7)
    pd.testing.assert_frame_equal(first_x, second_x)
    pd.testing.assert_series_equal(first_y, second_y)
    pd.testing.assert_frame_equal(features, saved_x)
    pd.testing.assert_series_equal(labels, saved_y)


def test_real_samples_allow_singleton_duplication():
    features, labels = sample_data((8, 4, 1))
    x, y = rebalance_real_samples(features, labels, random_state=7)
    assert y.value_counts().to_dict() == {"A": 4, "B": 4, "C": 4}
    assert (x.loc[y == "C", "signal"] == 12).all()


@pytest.mark.parametrize("invalid", ["misaligned", "duplicate", "infinite", "nan", "text",
                                     "empty", "no_columns", "one_class", "missing_label", "array"])
def test_real_samples_reject_invalid_inputs(invalid):
    features, labels = sample_data()
    if invalid == "misaligned":
        labels.index = labels.index[::-1]
    elif invalid == "duplicate":
        features.index = labels.index = pd.Index([0] * len(labels))
    elif invalid == "infinite":
        features.loc[0, "signal"] = np.inf
    elif invalid == "nan":
        features.loc[0, "signal"] = np.nan
    elif invalid == "text":
        features["signal"] = "text"
    elif invalid == "empty":
        features, labels = features.iloc[:0], labels.iloc[:0]
    elif invalid == "no_columns":
        features = features.iloc[:, :0]
    elif invalid == "one_class":
        labels = pd.Series(["A"] * len(labels))
    elif invalid == "missing_label":
        labels.iloc[0] = None
    elif invalid == "array":
        features = features.to_numpy()
    with pytest.raises(ValueError):
        rebalance_real_samples(features, labels, random_state=7)


def test_midpoint_smote_preserves_half_majority_gap():
    features, labels = sample_data()
    _, y = undersample_majority_then_smote(features, labels, random_state=7)
    assert y.value_counts().to_dict() == {"A": 6, "B": 6, "C": 6}


def test_midpoint_smote_explains_singleton_failure():
    features, labels = sample_data((8, 4, 1))
    with pytest.raises(ValueError, match="at least two samples"):
        undersample_majority_then_smote(features, labels, random_state=7)


@pytest.mark.parametrize("fraction", [True, False, np.bool_(True), np.nan, np.inf, -np.inf, 0, -0.1, 1.1])
def test_midpoint_smote_rejects_invalid_fraction(fraction):
    features, labels = sample_data()
    with pytest.raises(ValueError, match="majority_fraction"):
        undersample_majority_then_smote(features, labels, random_state=7, majority_fraction=fraction)
