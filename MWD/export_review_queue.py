"""Create an exclusive, source-verified review snapshot; never overwrite annotations."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.contract import load_dataset
from src.review_queue import build_review_queue


def verified_records(report, source_path):
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash != report['metadata']['source_sha256']:
        raise ValueError('Source SHA-256 mismatch')
    frame = load_dataset(source_path)
    records = []
    seeds = []
    for run in report['runs']:
        seeds.append(run['seed'])
        partitions = []
        for name in ('train', 'calibration', 'validation'):
            split = run['splits'][name]
            indices = split['indices']
            digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
            if digest != split['indices_sha256'] or len(indices) != split['rows'] or len(set(indices)) != len(indices):
                raise ValueError('Invalid split provenance')
            partitions.append(set(indices))
        if set.union(*partitions) != set(frame.index) or any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
            raise ValueError('Partitions must cover source exactly without overlap')
        predictions = run['quality_ablation']['predictions']
        if len(predictions) != len(partitions[2]) or {row['row_index'] for row in predictions} != partitions[2]:
            raise ValueError('Predictions must cover validation partition exactly')
        for row in predictions:
            if row['seed'] != run['seed'] or row['predicted_class'] not in run['classes']:
                raise ValueError('Prediction metadata mismatch')
            if not set(row['anomalous_features']).issubset(run['features']):
                raise ValueError('Unknown anomalous features')
            confident = row['confidence'] >= run['quality_ablation']['confidence_threshold']
            dual = confident and row['quality'] >= run['quality_ablation']['quality_threshold']
            if row['confidence_accepted'] != confident or row['dual_accepted'] != dual:
                raise ValueError('Gate flags do not match recorded thresholds')
        records.extend(predictions)
    if len(set(seeds)) != len(seeds) or seeds != report['metadata']['seeds']:
        raise ValueError('Seed provenance mismatch')
    return frame, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/quality_ablation/calibration.json'))
    parser.add_argument('--source', type=Path, default=Path('data/mwd_rocktype_10358374/mwd_rocktype_blastholes_model_ready_train.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    source, records = verified_records(report, args.source)
    queue = build_review_queue(records, source)
    metadata = {'source_sha256': report['metadata']['source_sha256'],
                'experiment_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
                'source': str(args.source), 'experiment': str(args.report),
                'validation_occurrences': len(records),
                'unique_validation_rows': len({row['row_index'] for row in records}),
                'review_rows': len(queue), 'row_index_semantics': 'zero-based CSV data row, not leading original ID'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'queue.json').write_text(json.dumps({'metadata': metadata, 'items': queue}, indent=2, allow_nan=False))
    csv_rows = [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()} for row in queue]
    pd.DataFrame(csv_rows).to_csv(args.output_dir / 'queue.csv', index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
