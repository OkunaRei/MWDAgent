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
from src.sampling import rebalance_training_set, undersample_majority_then_smote, rebalance_real_samples


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

FEATURE_GROUPS = {
    "all_48": lambda column: True,
    "penetration": lambda column: column.startswith("Penetr"),
    "rotation_pressure": lambda column: column.startswith("RotaPress"),
    "feed_hammer_pressure": lambda column: column.startswith(("FeedPress", "HammerPress")),
    "water_flow": lambda column: column.startswith("WaterFlow"),
    "central_statistics": lambda column: column.endswith(("Mean", "Median")),
    "variability_statistics": lambda column: column.endswith(
        ("Variance", "StandardDeviation", "Skewness", "Kurtosis")
    ),
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


def feature_group_columns(frame: pd.DataFrame, feature_group: str) -> list[str]:
    """Return an allowed subset of the public MWD features."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unsupported feature group: {feature_group}")
    columns = feature_columns(frame)
    selected = [column for column in columns if FEATURE_GROUPS[feature_group](column)]
    if not selected:
        raise ValueError(f"Feature group has no columns: {feature_group}")
    return selected


def select_model(summary: dict[str, dict[str, float | int]]) -> str:
    """Select a model from validation aggregates only."""
    if not summary:
        raise ValueError("summary must contain at least one model")
    if any("selection_score_mean" not in metrics for metrics in summary.values()):
        raise ValueError("Every model must have a selection_score_mean")
    return max(summary, key=lambda name: summary[name]["selection_score_mean"])


def select_candidate(summary: dict[str, dict[str, float | int]]) -> str:
    """Select a feature/model candidate from validation aggregates only."""
    return select_model(summary)


def candidate_key(feature_group: str, model_name: str) -> str:
    """Build a stable key for optimization logs."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unsupported feature group: {feature_group}")
    if model_name not in {"lightgbm", "extratrees"}:
        raise ValueError(f"Unsupported benchmark model: {model_name}")
    return f"{feature_group}__{model_name}"


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
    if model_name == "lightgbm_small":
        return LGBMClassifier(
            objective="multiclass",
            boosting_type="gbdt",
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=63,
            max_depth=12,
            min_child_samples=20,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
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
        "expected_calibration_error": metrics["expected_calibration_error"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
    }


def run_validation_once(
    train_frame: pd.DataFrame,
    *,
    model_name: str,
    validation_size: float,
    seed: int,
    selection_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
    feature_group: str = "all_48",
    training_strategy: str = "paper_smote",
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one model using only a split of the public train file."""
    valid_strategies = {"paper_smote", "original", "class_weight", "midpoint_smote", "real_resample"}
    if training_strategy not in valid_strategies:
        raise ValueError(f"Unsupported training strategy: {training_strategy}")
    train, validation = split_train_validation(train_frame, validation_size=validation_size, seed=seed)
    columns = feature_group_columns(train, feature_group)
    y_train, y_validation, encoder = _encode_labels(train, validation)
    original_counts = {str(label): int(count) for label, count in train["Rock"].value_counts().items()}
    if training_strategy == "paper_smote":
        fitted_features, fitted_labels = rebalance_training_set(train[columns], y_train, random_state=seed)
    elif training_strategy == "midpoint_smote":
        fitted_features, fitted_labels = undersample_majority_then_smote(
            train[columns], y_train, random_state=seed
        )
    elif training_strategy == "real_resample":
        fitted_features, fitted_labels = rebalance_real_samples(train[columns], y_train, random_state=seed)
    else:
        fitted_features, fitted_labels = train[columns].copy(deep=True), y_train.copy(deep=True)
    model = _make_model(model_name, seed)
    if training_strategy == "class_weight":
        model.set_params(class_weight="balanced")
    model.fit(fitted_features, fitted_labels)
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
        "feature_group": feature_group,
        "seed": seed,
        "validation_size": validation_size,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "balanced_train_rows": len(fitted_features),
        "training_strategy": training_strategy,
        "original_train_class_counts": original_counts,
        "fitted_train_class_counts": {
            str(encoder.classes_[int(label)]): int(count)
            for label, count in pd.Series(fitted_labels).value_counts().items()
        },
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
    if include_diagnostics:
        result = {**result, 'diagnostics': {
            'classes': classes, 'validation_indices': validation.index.tolist(),
            'overall': overall, 'ordinary': slices['ordinary'],
            'transition_zone': slices['transition_zone'],
        }}
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


def summarize_candidate_runs(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize validation runs by feature-group/model candidate."""
    run_list = list(runs)
    return summarize_validation_runs(
        {
            **run,
            "model": candidate_key(run["feature_group"], run["model"]),
        }
        for run in run_list
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
