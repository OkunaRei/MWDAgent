from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from src.benchmark import (
    _make_model,
    choose_test_seed,
    feature_group_columns,
    run_validation_once,
    select_candidate,
    summarize_candidate_runs,
    write_jsonl,
)
from src.contract import load_dataset
from src.evaluate import classification_metrics, slice_classification_metrics
from src.sampling import rebalance_training_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a finite, validation-only MWD optimizer")
    parser.add_argument("--config", type=Path, default=Path("experiment_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/optimizer"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evaluate_candidate_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_group: str,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    columns = feature_group_columns(train, feature_group)
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
        "feature_group": feature_group,
        "model": model_name,
        "seed": seed,
        "test_rows": len(test),
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "macro_f1": overall["macro_f1"],
        "weighted_f1": overall["weighted_f1"],
        "roc_auc_ovr": overall["roc_auc_ovr"],
        "ordinary_macro_f1": slices["ordinary"]["macro_f1"],
        "transition_macro_f1": slices["transition_zone"]["macro_f1"],
        "transition_support": slices["transition_zone"]["support"],
        "validation_not_used": True,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = args.config.parent
    train = load_dataset(root / config["dataset"]["train"])
    test = load_dataset(root / config["dataset"]["test"])
    seeds = [int(seed) for seed in config["optimization"]["seeds"]]
    selection_weights = (
        float(config["selection_score"]["balanced_accuracy_weight"]),
        float(config["selection_score"]["macro_f1_weight"]),
        float(config["selection_score"]["transition_macro_f1_weight"]),
    )
    validation_runs = [
        run_validation_once(
            train,
            model_name=model_name,
            feature_group=feature_group,
            validation_size=float(config["dataset"]["validation_size"]),
            seed=seed,
            selection_weights=selection_weights,
        )
        for feature_group in config["optimization"]["feature_groups"]
        for model_name in config["optimization"]["models"]
        for seed in seeds
    ]
    summary = summarize_candidate_runs(validation_runs)
    selected_key = select_candidate(summary)
    selected_group, selected_model = selected_key.split("__", maxsplit=1)
    reference_seed = choose_test_seed(
        seeds,
        reference_seed=int(config["evaluation"]["reference_test_seed"]),
    )
    test_result = evaluate_candidate_test(
        train,
        test,
        feature_group=selected_group,
        model_name=selected_model,
        seed=reference_seed,
    )
    result = {
        "protocol": {
            "validation_only_for_selection": True,
            "public_test_evaluated_after_selection": True,
            "reference_test_seed": reference_seed,
            "candidate_count": len(summary),
            "validation_run_count": len(validation_runs),
            "selection_weights": {
                "balanced_accuracy": selection_weights[0],
                "macro_f1": selection_weights[1],
                "transition_macro_f1": selection_weights[2],
            },
        },
        "validation_summary": summary,
        "selected_candidate": {"key": selected_key, "feature_group": selected_group, "model": selected_model},
        "test_result": test_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "optimizer.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "optimization_log.jsonl", validation_runs)
    pd.DataFrame([{"candidate": key, **metrics} for key, metrics in summary.items()]).to_csv(
        args.output_dir / "candidate_summary.csv", index=False
    )
    print(json.dumps({"selected_candidate": result["selected_candidate"], "test": test_result}, indent=2))


if __name__ == "__main__":
    main()
