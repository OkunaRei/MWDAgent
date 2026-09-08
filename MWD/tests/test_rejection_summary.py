import json
import sys

import pandas as pd

from summarize_rejections import main


def test_summary_counts_occurrences_separately_from_unique_rows(monkeypatch, tmp_path):
    record = dict(actual_class='A', row_index=7, extra_rejected=True,
                  confidence_accepted=True, dual_accepted=False, correct=True,
                  anomalous_features=['water'])
    source = tmp_path / 'predictions.jsonl'
    source.write_text('\n'.join(json.dumps(record) for _ in range(2)))
    monkeypatch.setattr(sys, 'argv', ['summarize_rejections.py', '--input', str(source)])
    main()
    row = pd.read_csv(tmp_path / 'rejection_classes.csv').iloc[0]
    assert row['occurrences'] == 2
    assert row['unique_rows'] == 1
    assert row['extra_reviewed_correct'] == 2
    assert pd.read_csv(tmp_path / 'rejection_features.csv').iloc[0]['occurrences'] == 2
