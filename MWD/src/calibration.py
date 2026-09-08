from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from src.benchmark import _make_model, split_train_validation
from src.contract import feature_columns
from src.evaluate import expected_calibration_error
from src.sampling import rebalance_training_set


CONFIDENCE_THRESHOLDS = (0, .5, .7, .8, .9, .95)
TEMPERATURE_BOUNDS = (.05, 20.0)


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must be a nonempty matrix with at least two classes")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("each probability row must sum to 1")
    return values


def _labels(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != (len(probabilities),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be aligned one-dimensional integer class indices")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("labels must refer to probability columns")
    return labels


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = _probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return softmax(np.log(np.clip(probabilities, np.finfo(float).tiny, 1)) / temperature, axis=1)


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one bounded scalar using calibration labels only; retain identity if better."""
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, probabilities)
    classes = np.arange(probabilities.shape[1])

    def objective(log_temperature: float) -> float:
        return float(log_loss(labels, temperature_scale(probabilities, np.exp(log_temperature)), labels=classes))

    result = minimize_scalar(objective, bounds=tuple(np.log(TEMPERATURE_BOUNDS)), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x)) if result.fun < objective(0) else 1.0


def split_calibration(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Public training rows only: 60% model fit, 20% calibration, 20% validation."""
    if not frame.index.is_unique:
        raise ValueError("source row indices must be unique")
    development, validation = split_train_validation(frame, validation_size=.2, seed=seed)
    train, calibration = split_train_validation(development, validation_size=.25, seed=seed)
    return train, calibration, validation


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    probabilities = _probabilities(probabilities)
    labels = _labels(labels, probabilities)
    correct = probabilities.argmax(axis=1) == labels
    confidence = probabilities.max(axis=1)
    sweep = []
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = confidence >= threshold
        sweep.append({"threshold": threshold, "accepted": int(accepted.sum()),
                      "coverage": float(accepted.mean()),
                      "risk": float(1 - correct[accepted].mean()) if accepted.any() else None})
    return {"support": len(labels), "accuracy": float(correct.mean()),
            "ece": expected_calibration_error(labels, probabilities, n_bins=10),
            "nll": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
            "confidence_sweep": sweep}


def _split_record(frame: pd.DataFrame) -> dict[str, Any]:
    indices = frame.index.tolist()
    return {"rows": len(indices), "indices": indices,
            "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest()}


def run_calibration_once(frame: pd.DataFrame, *, seed: int, include_quality: bool = False) -> dict[str, Any]:
    train, calibration, validation = split_calibration(frame, seed=seed)
    columns = feature_columns(train)
    encoder = LabelEncoder().fit(train["Rock"])
    y_train = pd.Series(encoder.transform(train["Rock"]), index=train.index, name="Rock")
    features, labels = rebalance_training_set(train[columns], y_train, random_state=seed)
    model = _make_model("lightgbm", seed)
    model.fit(features, labels)
    calibration_probabilities = model.predict_proba(calibration[columns])
    y_calibration = encoder.transform(calibration["Rock"])
    temperature = fit_temperature(y_calibration, calibration_probabilities)
    probabilities = model.predict_proba(validation[columns])
    calibrated = temperature_scale(probabilities, temperature)
    y_validation = encoder.transform(validation["Rock"])
    transition = validation["transition_zone"].to_numpy(dtype=bool)
    slices = {"overall": np.ones(len(validation), dtype=bool),
              "ordinary": ~transition, "transition_zone": transition}
    quality_result = {}
    if include_quality:
        from src.quality import fit_quality_bounds, quality_scores, gating_ablation
        from src.diagnostics import rejection_records

        lower, upper = fit_quality_bounds(train[columns].to_numpy())
        quality = quality_scores(validation[columns].to_numpy(), lower, upper)
        quality_result = {"quality_ablation": {
            "predictions": rejection_records(validation, columns, lower, upper,
                y_validation, calibrated, encoder.classes_.tolist(), seed=seed),
            "bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
            "quality_threshold": .9, "confidence_threshold": .8,
            "slices": {name: gating_ablation(y_validation[mask], calibrated[mask], quality[mask])
                       for name, mask in slices.items()},
        }}
    return {"seed": seed, "temperature": temperature, "model_params": model.get_params(),
            **quality_result,
            "classes": encoder.classes_.tolist(), "features": columns,
            "balanced_train_rows": len(features),
            "splits": {name: _split_record(part) for name, part in
                       (("train", train), ("calibration", calibration), ("validation", validation))},
            "calibration": {"before": probability_metrics(y_calibration, calibration_probabilities),
                            "after": probability_metrics(y_calibration, temperature_scale(calibration_probabilities, temperature))},
            "validation": {name: {"before": probability_metrics(y_validation[mask], probabilities[mask]),
                                  "after": probability_metrics(y_validation[mask], calibrated[mask])}
                           if mask.any() else {"before": None, "after": None}
                           for name, mask in slices.items()}}
