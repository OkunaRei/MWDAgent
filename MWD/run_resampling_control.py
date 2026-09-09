"""Matched class-count control: real-row duplication versus SMOTE interpolation."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.benchmark import SEED_LIST, run_validation_once, summarize_validation_runs
from src.class_impact import compare_class_impact
from src.contract import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'data/mwd_rocktype_10358374/mwd_rocktype_blastholes_model_ready_train.csv'
    frame = load_dataset(source)
    strategies = ('paper_smote', 'real_resample')
    protocol = {'strategies': strategies, 'model': 'lightgbm', 'seeds': list(SEED_LIST),
        'validation_size': .2, 'class_recall_floor': -.05, 'selection_improvement': .001,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('run_resampling_control.py', 'src/sampling.py', 'src/benchmark.py', 'src/class_impact.py')},
        'scope': 'Exploratory matched-count sampling control; not an independent Agent evaluation'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = {}
    for strategy in strategies:
        runs = [run_validation_once(frame, model_name='lightgbm', seed=seed,
            validation_size=.2, training_strategy=strategy, include_diagnostics=True) for seed in SEED_LIST]
        results[strategy] = runs
        (args.output_dir / f'{strategy}.json').write_text(json.dumps(runs, indent=2, allow_nan=False))
        print(f'{strategy}: five seeds completed', flush=True)
    base, candidate = (results[s] for s in strategies)
    if any(a['fitted_train_class_counts'] != b['fitted_train_class_counts'] for a, b in zip(base, candidate)):
        raise ValueError('Class counts differ between controls')
    impact = compare_class_impact(base, candidate)
    for name, rows in impact.items():
        pd.DataFrame(rows).to_csv(args.output_dir / f'{name}.csv', index=False)
    summaries = {s: summarize_validation_runs(runs)['lightgbm'] for s, runs in results.items()}
    a, b = (summaries[s] for s in strategies)
    violations = [row for row in impact['class_rows'] if row['slice'] == 'overall'
        and row['recall_delta'] is not None and row['recall_delta'] < -.05 - 1e-12]
    aggregate_pass = (b['selection_score_mean'] > a['selection_score_mean'] + .001
        and b['macro_f1_mean'] >= a['macro_f1_mean']
        and b['transition_macro_f1_mean'] >= a['transition_macro_f1_mean'])
    report = {'summaries': summaries, 'aggregate_pass': aggregate_pass,
        'class_violations': violations, 'eligible': aggregate_pass and not violations,
        'new_model_fits': 10, 'new_llm_trials': 0}
    (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
