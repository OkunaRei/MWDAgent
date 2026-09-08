from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.benchmark import (
    _make_model,
    choose_test_seed,
    run_validation_once,
    select_model,
    summarize_validation_runs,
    write_jsonl,
)
from src.contract import feature_columns, load_dataset
from src.evaluate import classification_metrics, slice_classification_metrics
from src.sampling import rebalance_training_set
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-Agent MWD benchmark")
    parser.add_argument("--config", type=Path, default=Path("experiment_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/benchmark"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evaluate_selected_configuration(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    columns = feature_columns(train)
    encoder = LabelEncoder()
    y_train = pd.Series(encoder.fit_transform(train["Rock"]), name="Rock")
    y_test = encoder.transform(test["Rock"])
    balanced_features, balanced_labels = rebalance_training_set(train[columns], y_train, random_state=seed)
    model = _make_model(model_name, seed)
    model.fit(balanced_features, balanced_labels)
    predictions = model.predict(test[columns]).astype(int)
    probabilities = model.predict_proba(test[columns])
    classes = list(encoder.classes_)
    overall = classification_metrics(y_test, predictions, probabilities, classes)
    slices = slice_classification_metrics(
        y_test,
        predictions,
        probabilities,
        classes,
        test["transition_zone"].to_numpy(dtype=bool),
    )
    return {
        "model": model_name,
        "seed": seed,
        "test_rows": len(test),
        "validation_not_used": True,
        "test": {
            "accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
            "macro_f1": overall["macro_f1"],
            "weighted_f1": overall["weighted_f1"],
            "expected_calibration_error": overall["expected_calibration_error"],
            "roc_auc_ovr": overall["roc_auc_ovr"],
        },
        "ordinary": {
            "support": slices["ordinary"]["support"],
            "macro_f1": slices["ordinary"]["macro_f1"],
            "expected_calibration_error": slices["ordinary"]["expected_calibration_error"],
            "balanced_accuracy": slices["ordinary"]["balanced_accuracy"],
        },
        "transition_zone": {
            "support": slices["transition_zone"]["support"],
            "macro_f1": slices["transition_zone"]["macro_f1"],
            "expected_calibration_error": slices["transition_zone"]["expected_calibration_error"],
            "balanced_accuracy": slices["transition_zone"]["balanced_accuracy"],
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = args.config.parent
    train = load_dataset(root / config["dataset"]["train"])
    test = load_dataset(root / config["dataset"]["test"])
    selection_weights = (
        float(config["selection_score"]["balanced_accuracy_weight"]),
        float(config["selection_score"]["macro_f1_weight"]),
        float(config["selection_score"]["transition_macro_f1_weight"]),
    )
    validation_runs = []
    for model_name in config["evaluation"]["models"]:
        for seed in config["evaluation"]["seeds"]:
            validation_runs.append(
                run_validation_once(
                    train,
                    model_name=model_name,
                    validation_size=float(config["dataset"]["validation_size"]),
                    seed=int(seed),
                    selection_weights=selection_weights,
                )
            )

    summary = summarize_validation_runs(validation_runs)
    selected_model = select_model(summary)
    selected_seed = choose_test_seed(
        config["evaluation"]["seeds"],
        reference_seed=int(config["evaluation"]["reference_test_seed"]),
    )
    test_result = evaluate_selected_configuration(
        train,
        test,
        model_name=selected_model,
        seed=int(selected_seed),
    )
    result = {
        "protocol": {
            "validation_only_for_selection": True,
            "public_test_evaluated_after_selection": True,
            "test_rows": len(test),
            "validation_size": config["dataset"]["validation_size"],
            "reference_test_seed": selected_seed,
            "selection_weights": {
                "balanced_accuracy": selection_weights[0],
                "macro_f1": selection_weights[1],
                "transition_macro_f1": selection_weights[2],
            },
        },
        "validation_runs": validation_runs,
        "validation_summary": summary,
        "selected_configuration": {"model": selected_model, "seed": selected_seed},
        "test_result": test_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "optimization_log.jsonl", validation_runs)
    pd.DataFrame(
        [
            {"model": model, **metrics}
            for model, metrics in summary.items()
        ]
    ).to_csv(args.output_dir / "validation_summary.csv", index=False)
    print(json.dumps({"selected": result["selected_configuration"], "test": test_result["test"]}, indent=2))


if __name__ == "__main__":
    main()
