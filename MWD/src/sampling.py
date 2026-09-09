from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
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


def undersample_majority_then_smote(
    features: pd.DataFrame, labels: pd.Series, *, random_state: int, majority_fraction: float = 0.5
) -> tuple[pd.DataFrame, pd.Series]:
    """Preserve a fixed fraction of the gap between the two largest classes."""
    if (
        isinstance(majority_fraction, (bool, np.bool_))
        or not isinstance(majority_fraction, Real)
        or not np.isfinite(majority_fraction)
        or not 0 < majority_fraction <= 1
    ):
        raise ValueError("majority_fraction must be in (0, 1]")
    counts = labels.value_counts().sort_values(ascending=False)
    if len(counts) < 2:
        raise ValueError("At least two classes are required")
    target = int(round(counts.iloc[1] + majority_fraction * (counts.iloc[0] - counts.iloc[1])))
    sampler = RandomUnderSampler(
        sampling_strategy={counts.index[0]: target}, random_state=random_state
    )
    reduced_features, reduced_labels = sampler.fit_resample(features, labels)
    min_count = int(reduced_labels.value_counts().min())
    if min_count < 2:
        raise ValueError("SMOTE requires at least two samples in every class")
    over = SMOTE(sampling_strategy="all", random_state=random_state, k_neighbors=min(5, min_count - 1))
    x, y = over.fit_resample(reduced_features, reduced_labels)
    return pd.DataFrame(x, columns=features.columns), pd.Series(y, name=labels.name).reset_index(drop=True)


def rebalance_real_samples(
    features: pd.DataFrame, labels: pd.Series, *, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Match paper class counts using only existing training feature-label pairs.

    Reduce classes above the second-largest count, then duplicate actual rows
    in smaller classes. No interpolated feature vectors are introduced.
    """
    if not isinstance(features, pd.DataFrame) or not isinstance(labels, pd.Series):
        raise ValueError("features must be a DataFrame and labels must be a Series")
    if features.empty or labels.empty:
        raise ValueError("Nonempty features and labels are required")
    if not features.index.is_unique or not labels.index.is_unique:
        raise ValueError("Feature and label indices must be unique")
    if not features.index.equals(labels.index):
        raise ValueError("Feature and label indices must be aligned")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Features must be finite numeric values")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite numeric values")
    if labels.isna().any():
        raise ValueError("Labels must not contain missing values")
    counts = labels.value_counts().sort_values(ascending=False)
    if len(counts) < 2:
        raise ValueError("At least two classes are required for rebalancing")
    target = int(counts.iloc[1])
    under = RandomUnderSampler(
        sampling_strategy={label: target for label, count in counts.items() if count > target},
        random_state=random_state,
    )
    reduced_x, reduced_y = under.fit_resample(features, labels)
    over = RandomOverSampler(sampling_strategy="all", random_state=random_state)
    balanced_x, balanced_y = over.fit_resample(reduced_x, reduced_y)
    return (
        pd.DataFrame(balanced_x, columns=features.columns).reset_index(drop=True),
        pd.Series(balanced_y, name=labels.name).reset_index(drop=True),
    )
