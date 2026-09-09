"""Frozen six-candidate training-strategy study, using public training data only."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.benchmark import SEED_LIST, run_validation_once, summarize_validation_runs
from src.contract import load_dataset


CATALOG = tuple((model, strategy) for model in ('lightgbm', 'extratrees')
                for strategy in ('paper_smote', 'original', 'class_weight', 'midpoint_smote'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'data/mwd_rocktype_10358374/mwd_rocktype_blastholes_model_ready_train.csv'
    source_bytes = source.read_bytes()
    frame = load_dataset(source)
    protocol = {'seeds': list(SEED_LIST), 'validation_size': .2, 'feature_group': 'all_48',
        'selection_weights': [.2, .5, .3], 'catalog': [list(pair) for pair in CATALOG],
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ('run_sampling_study.py', 'src/benchmark.py', 'src/sampling.py', 'src/evaluate.py')},
        'packages': {name: version(name) for name in ('lightgbm', 'scikit-learn', 'imbalanced-learn', 'numpy', 'pandas')},
        'purpose': 'Numerical action-space study, not independent Agent policy comparison',
        'limitations': 'Repeated random validation rows, previously explored data, no test data access or field generalization claim'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for model, strategy in CATALOG:
        key = f'all_48__{model}__{strategy}'
        runs = [run_validation_once(frame, model_name=model, training_strategy=strategy,
                validation_size=.2, seed=seed) for seed in SEED_LIST]
        summary = summarize_validation_runs(runs)[model]
        item = {'candidate': key, 'runs': runs, 'summary': summary}
        (args.output_dir / f'{key}.json').write_text(json.dumps(item, indent=2, allow_nan=False))
        results.append(item)
        print(json.dumps({'candidate': key, 'summary': summary}), flush=True)
    baseline = results[0]['summary']
    rows = []
    for item in results:
        summary = item['summary']
        rows.append({'candidate': item['candidate'], **summary,
            'promotable_vs_baseline': summary['selection_score_mean'] > baseline['selection_score_mean'] + .001
            and summary['macro_f1_mean'] >= baseline['macro_f1_mean']
            and summary['transition_macro_f1_mean'] >= baseline['transition_macro_f1_mean']})
    pd.DataFrame(rows).to_csv(args.output_dir / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
