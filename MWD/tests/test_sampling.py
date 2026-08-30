import pandas as pd

from src.sampling import rebalance_training_set


def test_rebalance_training_set_targets_the_second_largest_class() -> None:
    features = pd.DataFrame({"signal": range(12)})
    labels = pd.Series(
        ["A"] * 6 + ["B"] * 4 + ["C"] * 2,
        name="Rock",
    )

    _, balanced_labels = rebalance_training_set(features, labels, random_state=7)

    counts = balanced_labels.value_counts()
    assert counts.to_dict() == {"A": 4, "B": 4, "C": 4}


def test_rebalance_training_set_does_not_mutate_inputs() -> None:
    features = pd.DataFrame({"signal": range(8)})
    labels = pd.Series(["A", "A", "A", "B", "B", "B", "C", "C"], name="Rock")
    expected_features = features.copy(deep=True)
    expected_labels = labels.copy(deep=True)

    rebalance_training_set(features, labels, random_state=7)

    pd.testing.assert_frame_equal(features, expected_features)
    pd.testing.assert_series_equal(labels, expected_labels)
