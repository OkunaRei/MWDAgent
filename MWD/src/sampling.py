from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def rebalance_training_set(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Match the paper's train-only undersampling followed by SMOTE.

    The largest class is reduced to the size of the second-largest class;
    SMOTE then brings every class to that size. Inputs remain unchanged.
    """
    counts = labels.value_counts().sort_values(ascending=False)
    if len(counts) < 2:
        raise ValueError("At least two classes are required for rebalancing")
    target_size = int(counts.iloc[1])
    under_strategy = {label: target_size for label, count in counts.items() if count > target_size}
    under_sampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    under_features, under_labels = under_sampler.fit_resample(features, labels)

    min_class_count = int(under_labels.value_counts().min())
    if min_class_count < 2:
        raise ValueError("SMOTE requires at least two samples in every class")
    over_sampler = SMOTE(
        sampling_strategy="all",
        random_state=random_state,
        k_neighbors=min(5, min_class_count - 1),
    )
    balanced_features, balanced_labels = over_sampler.fit_resample(under_features, under_labels)
    return (
        pd.DataFrame(balanced_features, columns=features.columns),
        pd.Series(balanced_labels, name=labels.name).reset_index(drop=True),
    )
