"""Compare class-level effects of two frozen training strategies on paired validation splits."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.benchmark import SEED_LIST, run_validation_once
from src.class_impact import compare_class_impact
from src.contract import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'data/mwd_rocktype_10358374/mwd_rocktype_blastholes_model_ready_train.csv'
    frame = load_dataset(source)
    protocol = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'seeds': list(SEED_LIST), 'validation_size': .2, 'model': 'lightgbm',
                'strategies': ['paper_smote', 'class_weight'], 'features': 'all_48',
                'code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('analyze_strategy_impact.py', 'src/benchmark.py', 'src/class_impact.py')},
                'purpose': 'Paired retrospective class-impact analysis, no selection rule changes'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    runs = {}
    for strategy in protocol['strategies']:
        runs[strategy] = [run_validation_once(frame, model_name='lightgbm',
            training_strategy=strategy, validation_size=.2, seed=seed, include_diagnostics=True)
            for seed in SEED_LIST]
        (args.output_dir / f'{strategy}.json').write_text(json.dumps(runs[strategy], indent=2, allow_nan=False))
        print(f'{strategy}: completed {len(runs[strategy])} seeds', flush=True)
    result = compare_class_impact(runs['paper_smote'], runs['class_weight'])
    for name, rows in result.items():
        pd.DataFrame(rows).to_csv(args.output_dir / f'{name}.csv', index=False)
    (args.output_dir / 'impact.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
