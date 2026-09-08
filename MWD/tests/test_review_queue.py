import copy
import json
import hashlib
import sys

import pandas as pd
import pytest

from src.review_queue import build_review_queue


def source():
    return pd.DataFrame({'Rock': ['A', 'B'], 'transition_zone': [False, True],
                         'Tunnel': ['T1', 'T1'], 'PegStart': [10., 20.], 'PegEnd': [11., 21.]})


def record(row=0, seed=11, correct=True, extra=True):
    return dict(row_index=row, seed=seed, actual_class='A' if row == 0 else 'B',
                predicted_class=('A' if row == 0 else 'B') if correct else 'C',
                correct=correct, confidence=.9, quality=.8 if extra else 1.,
                confidence_accepted=True, dual_accepted=not extra, extra_rejected=extra,
                transition_zone=row == 1, anomalous_features=['flow'] if extra else [])


def test_unique_queue_preserves_denominators_and_prioritizes_confident_errors():
    records = [record(), record(seed=42), record(row=1, correct=False, extra=False)]
    before = copy.deepcopy(records)
    queue = build_review_queue(records, source())
    assert [row['row_index'] for row in queue] == [1, 0]
    assert queue[0]['confident_error_occurrences'] == 1
    assert queue[1]['validation_occurrences'] == 2
    assert queue[1]['extra_correct_occurrences'] == 2
    assert queue[1]['unique_seed_count'] == 2
    assert queue[1]['anomalous_feature_counts'] == {'flow': 2}
    assert queue[0]['PegStart'] == 20.
    assert all(row['quality_label'] == '' and row['evidence_reference'] == '' for row in queue)
    assert records == before
    json.dumps(queue, allow_nan=False)


@pytest.mark.parametrize('change', [
    {'row_index': 10}, {'row_index': -1}, {'row_index': 0.5},
    {'actual_class': 'wrong'}, {'transition_zone': True}, {'correct': False},
    {'confidence': float('nan')}, {'confidence_accepted': 'False'},
    {'dual_accepted': True}, {'anomalous_features': ['flow', 'flow']},
])
def test_reject_inconsistent_evidence(change):
    with pytest.raises(ValueError):
        build_review_queue([{**record(), **change}], source())


def test_duplicate_seed_row_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        build_review_queue([record(), record()], source())


def test_clean_predictions_are_not_review_items():
    assert build_review_queue([record(extra=False)], source()) == []


def test_export_verifies_hash_and_preserves_existing_review(monkeypatch, tmp_path):
    from export_review_queue import main, verified_records

    csv = tmp_path / 'train.csv'
    source().to_csv(csv, index=False)
    splits = {name: {'indices': ids, 'rows': len(ids),
                     'indices_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
              for name, ids in [('train', [1]), ('calibration', []), ('validation', [0])]}
    report = {'metadata': {'source_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(), 'seeds': [11]},
              'runs': [{'seed': 11, 'splits': splits, 'classes': ['A', 'B'], 'features': ['flow'],
                        'quality_ablation': {'predictions': [record()],
                                             'confidence_threshold': .8, 'quality_threshold': .9}}]}
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(report))
    out = tmp_path / 'queue'
    monkeypatch.setattr(sys, 'argv', ['export_review_queue.py', '--report', str(path),
                                    '--source', str(csv), '--output-dir', str(out)])
    main()
    assert json.loads((out / 'queue.json').read_text())['metadata']['review_rows'] == 1
    (out / 'queue.csv').write_text('human annotations')
    with pytest.raises(FileExistsError):
        main()
    assert (out / 'queue.csv').read_text() == 'human annotations'
    changed = copy.deepcopy(report)
    changed['runs'][0]['quality_ablation']['predictions'][0]['dual_accepted'] = True
    with pytest.raises(ValueError, match='Gate flags'):
        verified_records(changed, csv)
    changed = copy.deepcopy(report)
    changed['runs'][0]['splits']['validation']['indices'] = [1]
    with pytest.raises(ValueError, match='provenance'):
        verified_records(changed, csv)
    csv.write_text('changed source')
    with pytest.raises(ValueError, match='SHA-256'):
        verified_records(report, csv)
