from __future__ import annotations

import numpy as np
import pandas as pd

from src.quality import quality_scores, _validate_ablation


def rejection_records(frame, columns, lower, upper, labels, probabilities,
                      class_names, *, seed, quality_threshold=.9, confidence_threshold=.8):
    """Record frozen-gate evidence; anomalous features are not causal explanations."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    if not frame.index.is_unique or not pd.api.types.is_integer_dtype(frame.index.dtype):
        raise ValueError('source indices must be unique integers')
    if (not columns or len(set(columns)) != len(columns)
            or any(column not in frame for column in columns)):
        raise ValueError('feature columns must be unique and present')
    if 'transition_zone' not in frame or not pd.api.types.is_bool_dtype(frame['transition_zone']):
        raise ValueError('transition_zone must be boolean')
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if (probabilities.ndim != 2 or len(probabilities) != len(frame)
            or len(class_names) != probabilities.shape[1]
            or not all(isinstance(name, str) for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError('class names and probabilities must align')
    features = frame[columns].to_numpy(dtype=float)
    quality = quality_scores(features, lower, upper)
    _validate_ablation(labels, probabilities, quality, quality_threshold, confidence_threshold)
    anomalous = ~np.isfinite(features) | (features < lower) | (features > upper)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return [dict(row_index=int(index), seed=int(seed), actual_class=class_names[int(labels[i])],
                 predicted_class=class_names[int(predicted[i])], correct=bool(labels[i] == predicted[i]),
                 confidence=float(confidence[i]), quality=float(quality[i]),
                 confidence_accepted=bool(confidence[i] >= confidence_threshold),
                 dual_accepted=bool(confidence[i] >= confidence_threshold and quality[i] >= quality_threshold),
                 extra_rejected=bool(confidence[i] >= confidence_threshold and quality[i] < quality_threshold),
                 transition_zone=bool(frame['transition_zone'].iloc[i]),
                 anomalous_features=[name for name, flag in zip(columns, anomalous[i]) if flag])
            for i, index in enumerate(frame.index)]
