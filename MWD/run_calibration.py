from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import SEED_LIST
from src.calibration import CONFIDENCE_THRESHOLDS, TEMPERATURE_BOUNDS, run_calibration_once
from src.contract import load_dataset


def summarize_rows(rows: list[dict]) -> dict:
    summary = {}
    for name in ("overall", "ordinary", "transition_zone"):
        stages = {}
        for stage in ("before", "after"):
            selected = [row for row in rows if row["slice"] == name and row["stage"] == stage]
            stages[stage] = {
                metric: {"mean": float(np.mean(values)) if values else None,
                         "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         "runs": len(values)}
                for metric in ("accuracy", "ece", "nll")
                for values in [[row[metric] for row in selected]]
            }
        summary[name] = stages
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only temperature calibration experiment")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mwd_rocktype_10358374"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quality-ablation", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or Path("reports/quality_ablation" if args.quality_ablation else "reports/calibration")
    source = args.data_dir / "mwd_rocktype_blastholes_model_ready_train.csv"
    frame = load_dataset(source)
    runs = []
    for seed in SEED_LIST:
        options = {"include_quality": True} if args.quality_ablation else {}
        run = run_calibration_once(frame, seed=seed, **options)
        runs.append(run)
        print(f"seed={seed}, temperature={run['temperature']:.6f}", flush=True)
    rows = [{"seed": run["seed"], "temperature": run["temperature"], "slice": name,
             "stage": stage, **{key: metrics[key] for key in ("support", "accuracy", "ece", "nll")}}
            for run in runs for name, stages in run["validation"].items()
            for stage, metrics in stages.items() if metrics is not None]
    summary = summarize_rows(rows)
    metadata = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "seeds": list(SEED_LIST), "split": {"validation_size": .2, "calibration_size_of_remaining": .25},
                "temperature_bounds": TEMPERATURE_BOUNDS, "ece_bins": 10,
                "confidence_thresholds": CONFIDENCE_THRESHOLDS, "python": platform.python_version(),
                "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "imbalanced-learn")},
                "sampling": "second-largest-class undersampling followed by SMOTE on model-fit partition only",
                "quality_proxy": "training-only Tukey 1.5 IQR inlier feature fraction" if args.quality_ablation else None,
                "limitations": "Random row splits; exploratory repeated holdouts, no independent field validation or measured sensor quality; public test not loaded."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(json.dumps({"metadata": metadata, "summary": summary, "runs": runs}, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "validation_summary.csv", index=False)
    if args.quality_ablation:
        ablations = [{"seed": run["seed"], "slice": name, "mode": mode,
                      **{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}}
                     for run in runs for name, modes in run["quality_ablation"]["slices"].items()
                     for mode, metrics in modes.items()]
        pd.DataFrame(ablations).to_csv(output_dir / "gating_summary.csv", index=False)
        predictions = [record for run in runs
                       for record in run["quality_ablation"].get("predictions", [])]
        (output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in predictions), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
