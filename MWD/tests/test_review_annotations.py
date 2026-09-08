import copy
import csv
import json
import sys

import pytest

from src.review_annotations import validate_annotations


def snapshot():
    return {'items': [dict(row_index=7, actual_class='A', transition_zone=False,
        PegStart=10.5, seeds=[11, 42], review_status='pending', quality_label='',
        evidence_reference='', reviewer='', reviewed_at='', notes='')]}


def csv_rows():
    return [{key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
             for key, value in row.items()} for row in snapshot()['items']]


def completed(label='measurement_issue'):
    return {**csv_rows()[0], 'review_status': 'completed', 'quality_label': label,
            'reviewer': 'engineer', 'reviewed_at': '2026-09-08',
            'evidence_reference': 'field-log:12', 'notes': 'Observed missing sensor values.'}


def test_pending_queue_produces_no_quality_labels():
    result = validate_annotations(snapshot(), csv_rows())
    assert result['valid']
    assert result['status_counts'] == {'pending': 1, 'in_review': 0, 'completed': 0}
    assert result['completed_reviews'] == []


def test_completed_and_insufficient_evidence_remain_distinct():
    row = completed()
    before = copy.deepcopy(row)
    result = validate_annotations(snapshot(), [row])
    assert result['valid']
    assert result['completed_reviews'][0]['quality_label'] == 'measurement_issue'
    assert result['evidence_backed_count'] == 1
    assert row == before
    row = {**completed('insufficient_evidence'), 'evidence_reference': ''}
    result = validate_annotations(snapshot(), [row])
    assert result['valid']
    assert result['evidence_backed_count'] == 0


@pytest.mark.parametrize('change', [
    {'actual_class': 'B'}, {'row_index': '8'}, {'seeds': '[11]'},
    {'transition_zone': 'True'}, {'PegStart': 'NaN'}, {'PegStart': '11'},
    {'review_status': 'done'}, {'quality_label': 'bad_sensor'},
    {'reviewer': ''}, {'reviewed_at': '2026-02-30'},
    {'evidence_reference': ''}, {'notes': ''},
])
def test_rejects_bad_completed_reviews_without_exporting_labels(change):
    result = validate_annotations(snapshot(), [{**completed(), **change}])
    assert not result['valid']
    assert result['errors']
    assert result['completed_reviews'] == []


def test_partial_and_duplicate_queues_are_rejected():
    assert not validate_annotations(snapshot(), [])['valid']
    assert not validate_annotations(snapshot(), csv_rows() * 2)['valid']


def test_unfinished_labels_cannot_enter_completed_results():
    row = {**completed(), 'review_status': 'in_review'}
    result = validate_annotations(snapshot(), [row])
    assert result['valid']
    assert result['completed_reviews'] == []


def test_insufficient_evidence_requires_reason():
    row = {**completed('insufficient_evidence'), 'notes': '', 'evidence_reference': ''}
    assert not validate_annotations(snapshot(), [row])['valid']


@pytest.mark.parametrize('invalid', [{}, {'items': []}, {'items': [{}]}, {'items': 'wrong'}])
def test_malformed_snapshot_fails_clearly(invalid):
    with pytest.raises(ValueError, match='Snapshot'):
        validate_annotations(invalid, csv_rows())


@pytest.mark.parametrize('change', [
    {'notes': '   '}, {'reviewed_at': '20260908'}, {'reviewer': None},
    {None: ['extra column']}, {'seeds': '[true, 42]'},
])
def test_csv_edge_cases_fail_without_export(change):
    result = validate_annotations(snapshot(), [{**completed(), **change}])
    assert not result['valid']
    assert result['completed_reviews'] == []


@pytest.mark.parametrize('valid', [True, False])
def test_cli_audits_input_hashes_returns_failure_and_never_overwrites(monkeypatch, tmp_path, valid):
    import hashlib
    from validate_review_queue import main

    original = tmp_path / 'queue.json'
    annotated = tmp_path / 'queue.csv'
    out = tmp_path / 'result'
    original.write_text(json.dumps({**snapshot(), 'metadata': {'source_sha256': 'source', 'experiment_sha256': 'experiment'}}))
    row = completed() if valid else {**completed(), 'actual_class': 'changed'}
    with annotated.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    before = annotated.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['validate_review_queue.py', '--snapshot', str(original),
                                    '--annotations', str(annotated), '--output-dir', str(out)])
    if valid:
        main()
    else:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    report = json.loads((out / 'validation.json').read_text())
    assert report['valid'] is valid
    assert report['provenance']['annotations_sha256'] == hashlib.sha256(before).hexdigest()
    assert len(report['completed_reviews']) == (1 if valid else 0)
    assert annotated.read_bytes() == before
    with pytest.raises(FileExistsError):
        main()
