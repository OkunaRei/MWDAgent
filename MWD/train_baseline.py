from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from src.contract import feature_columns, load_dataset
from src.evaluate import classification_metrics, slice_classification_metrics
from src.sampling import rebalance_training_set


SEED = 42
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the Hansen MWD rock-type baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mwd_rocktype_10358374"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_dataset(args.data_dir / "mwd_rocktype_blastholes_model_ready_train.csv")
    test = load_dataset(args.data_dir / "mwd_rocktype_blastholes_model_ready_test.csv")
    columns = feature_columns(train)

    encoder = LabelEncoder()
    y_train = pd.Series(encoder.fit_transform(train["Rock"]), name="Rock")
    y_test = encoder.transform(test["Rock"])
    X_train, y_train = rebalance_training_set(train[columns], y_train, random_state=SEED)

    model = LGBMClassifier(
        objective="multiclass",
        random_state=SEED,
        verbosity=-1,
        **PAPER_PARAMS,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(test[columns]).astype(int)
    probabilities = model.predict_proba(test[columns])
    metrics = classification_metrics(y_test, predictions, probabilities, list(encoder.classes_))
    metrics["slices"] = slice_classification_metrics(
        y_test,
        predictions,
        probabilities,
        list(encoder.classes_),
        test["transition_zone"].to_numpy(dtype=bool),
    )
    metrics["metadata"] = {
        "seed": SEED,
        "train_rows": len(train),
        "test_rows": len(test),
        "balanced_train_rows": len(X_train),
        "feature_count": len(columns),
        "features": columns,
        "classes": list(encoder.classes_),
        "sampling": "second-largest-class undersampling followed by SMOTE",
        "paper_params": PAPER_PARAMS,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["classification_report"]).T.to_csv(args.output_dir / "classification_report.csv")
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=encoder.classes_, columns=encoder.classes_)
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    print(json.dumps({key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr")}, indent=2))


if __name__ == "__main__":
    main()
