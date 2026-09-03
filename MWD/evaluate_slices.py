from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from src.contract import feature_columns, load_dataset
from src.evaluate import classification_metrics, slice_classification_metrics
from src.sampling import rebalance_training_set
from train_baseline import PAPER_PARAMS, SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Hansen MWD baseline by geological slices")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mwd_rocktype_10358374"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/slices"))
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

    model = LGBMClassifier(objective="multiclass", random_state=SEED, verbosity=-1, **PAPER_PARAMS)
    model.fit(X_train, y_train)
    predictions = model.predict(test[columns]).astype(int)
    probabilities = model.predict_proba(test[columns])
    classes = list(encoder.classes_)

    result = {
        "overall": classification_metrics(y_test, predictions, probabilities, classes),
        "slices": slice_classification_metrics(
            y_test,
            predictions,
            probabilities,
            classes,
            test["transition_zone"].to_numpy(dtype=bool),
        ),
        "metadata": {
            "seed": SEED,
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_count": len(columns),
            "classes": classes,
            "sampling": "second-largest-class undersampling followed by SMOTE",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for slice_name, metrics in result["slices"].items():
        rows.append(
            {
                "slice": slice_name,
                "support": metrics["support"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc_ovr": metrics["roc_auc_ovr"],
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "slice_summary.csv", index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
