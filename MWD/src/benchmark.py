from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.contract import feature_columns
from src.evaluate import classification_metrics, slice_classification_metrics
from src.sampling import rebalance_training_set


SEED_LIST = (11, 19, 42, 73, 101)
PAPER_PARAMS = {
    "boosting_type": "dart",
    "colsample_bytree": 0.6563099142197473,
    "learning_rate": 0.32031365887407864,
    "max_depth": 49,
    "min_child_samples": 49,
    "min_child_weight": 2.9301413598309467e-05,
    "n_estimators": 130,
    "num_leaves": 236,
    "reg_alpha": 0.020733894378166445,
    "reg_lambda": 2.584873506220451e-05,
    "subsample": 0.7813241713152921,
}


def split_train_validation(
    frame: pd.DataFrame,
    *,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split only the public training frame; preserve original row indices."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if "Rock" not in frame.columns:
        raise ValueError("Training frame must contain the Rock target")
    train_index, validation_index = train_test_split(
        frame.index,
        test_size=validation_size,
        random_state=seed,
        stratify=frame["Rock"],
    )
    return frame.loc[train_index].copy(deep=True), frame.loc[validation_index].copy(deep=True)


def compute_selection_score(
    *,
    balanced_accuracy: float,
    macro_f1: float,
    transition_macro_f1: float,
    balanced_accuracy_weight: float = 0.2,
    macro_f1_weight: float = 0.5,
    transition_macro_f1_weight: float = 0.3,
) -> float:
    """Score used by future Agent optimization; validation-only by design."""
    weights = (balanced_accuracy_weight, macro_f1_weight, transition_macro_f1_weight)
    if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("selection score weights must be non-negative and sum to 1")
    return float(
        balanced_accuracy_weight * balanced_accuracy
        + macro_f1_weight * macro_f1
        + transition_macro_f1_weight * transition_macro_f1
    )


def select_model(summary: dict[str, dict[str, float | int]]) -> str:
    """Select a model from validation aggregates only."""
    if not summary:
        raise ValueError("summary must contain at least one model")
    if any("selection_score_mean" not in metrics for metrics in summary.values()):
        raise ValueError("Every model must have a selection_score_mean")
    return max(summary, key=lambda name: summary[name]["selection_score_mean"])


def choose_test_seed(seeds: Iterable[int], *, reference_seed: int) -> int:
    """Return a pre-registered seed for the one-time public test evaluation."""
    seed_values = tuple(int(seed) for seed in seeds)
    if reference_seed not in seed_values:
        raise ValueError(f"reference_seed {reference_seed} must be included in evaluation seeds")
    return reference_seed


def _make_model(model_name: str, seed: int) -> Any:
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
            **PAPER_PARAMS,
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=1,
            class_weight=None,
        )
    raise ValueError(f"Unsupported benchmark model: {model_name}")


def _encode_labels(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.Series, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded_train = pd.Series(encoder.fit_transform(train["Rock"]), index=train.index, name="Rock")
    encoded_validation = encoder.transform(validation["Rock"])
    return encoded_train, encoded_validation, encoder


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
    }


def run_validation_once(
    train_frame: pd.DataFrame,
    *,
    model_name: str,
    validation_size: float,
    seed: int,
    selection_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
) -> dict[str, Any]:
    """Train and evaluate one model using only a split of the public train file."""
    train, validation = split_train_validation(train_frame, validation_size=validation_size, seed=seed)
    columns = feature_columns(train)
    y_train, y_validation, encoder = _encode_labels(train, validation)
    balanced_features, balanced_labels = rebalance_training_set(
        train[columns], y_train, random_state=seed
    )
    model = _make_model(model_name, seed)
    model.fit(balanced_features, balanced_labels)
    predictions = model.predict(validation[columns]).astype(int)
    probabilities = model.predict_proba(validation[columns])
    classes = list(encoder.classes_)
    overall = classification_metrics(y_validation, predictions, probabilities, classes)
    slices = slice_classification_metrics(
        y_validation,
        predictions,
        probabilities,
        classes,
        validation["transition_zone"].to_numpy(dtype=bool),
    )
    result = {
        "model": model_name,
        "seed": seed,
        "validation_size": validation_size,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "balanced_train_rows": len(balanced_features),
        "feature_count": len(columns),
        "validation": _flat_metrics(overall),
        "ordinary": {"support": slices["ordinary"]["support"], **_flat_metrics(slices["ordinary"])},
        "transition_zone": {
            "support": slices["transition_zone"]["support"],
            **_flat_metrics(slices["transition_zone"]),
        },
    }
    result["selection_score"] = compute_selection_score(
        balanced_accuracy=result["validation"]["balanced_accuracy"],
        macro_f1=result["validation"]["macro_f1"],
        transition_macro_f1=result["transition_zone"]["macro_f1"],
        balanced_accuracy_weight=selection_weights[0],
        macro_f1_weight=selection_weights[1],
        transition_macro_f1_weight=selection_weights[2],
    )
    return result


def summarize_validation_runs(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)
    summary: dict[str, dict[str, float | int]] = {}
    for model_name, model_runs in grouped.items():
        summary[model_name] = {
            "runs": len(model_runs),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [run["validation"][metric] for run in model_runs if metric in run["validation"]]
            if values:
                array = np.asarray(values, dtype=float)
                summary[model_name][f"{metric}_mean"] = float(array.mean())
                summary[model_name][f"{metric}_std"] = float(array.std(ddof=0))
        transition_values = [
            run["transition_zone"]["macro_f1"]
            for run in model_runs
            if "transition_zone" in run and "macro_f1" in run["transition_zone"]
        ]
        if transition_values:
            array = np.asarray(transition_values, dtype=float)
            summary[model_name]["transition_macro_f1_mean"] = float(array.mean())
            summary[model_name]["transition_macro_f1_std"] = float(array.std(ddof=0))
        scores = [run["selection_score"] for run in model_runs if "selection_score" in run]
        if scores:
            array = np.asarray(scores, dtype=float)
            summary[model_name]["selection_score_mean"] = float(array.mean())
            summary[model_name]["selection_score_std"] = float(array.std(ddof=0))
    return summary


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
