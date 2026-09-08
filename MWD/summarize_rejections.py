"""Summarize retained/reviewed validation occurrences, without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/quality_ablation/predictions.jsonl'))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('No prediction records')
    frame = pd.DataFrame(records)
    rows = []
    for label, group in frame.groupby('actual_class'):
        extra = group[group['extra_rejected']]
        rows.append({'class': label, 'occurrences': len(group),
                     'unique_rows': group['row_index'].nunique(),
                     'confidence_accepted': int(group['confidence_accepted'].sum()),
                     'dual_accepted': int(group['dual_accepted'].sum()),
                     'extra_reviewed': len(extra),
                     'extra_reviewed_correct': int(extra['correct'].sum())})
    pd.DataFrame(rows).to_csv(args.input.parent / 'rejection_classes.csv', index=False)
    extra = frame[frame['extra_rejected']]
    features = [{'class': row.actual_class, 'feature': feature}
                for row in extra.itertuples() for feature in row.anomalous_features]
    counts = (pd.DataFrame(features, columns=['class', 'feature'])
              .groupby(['class', 'feature']).size().rename('occurrences').reset_index())
    counts.to_csv(args.input.parent / 'rejection_features.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
