from copy import deepcopy
import json
import sys
import pytest
from src.feedback_ablation import feedback_view


def observation():
    return {'revision': 1, 'observation_sha256': 'digest', 'policy': {'minimum_class_recall_delta': -.05},
        'evaluations': [{'candidate': 'candidate', 'accepted': False, 'status': 'completed',
            'reason': 'class_recall_floor', 'summary': {'selection_score_mean': .7},
            'class_violations': [{'class': 'SECRET_CLASS', 'minimum_recall_delta': -.3}]}]}


def test_aggregate_view_removes_specific_class_evidence_but_keeps_outcome_and_rules():
    original = observation()
    before = deepcopy(original)
    masked = feedback_view(original, 'aggregate')
    assert masked['evaluations'][0]['reason'] == 'not_promoted'
    assert 'class_violations' not in masked['evaluations'][0]
    assert 'SECRET_CLASS' not in str(masked)
    assert masked['evaluations'][0]['accepted'] is False
    assert masked['policy'] == original['policy']
    assert masked['observation_sha256'] == 'digest'
    assert original == before


def test_full_view_detached_and_unknown_condition_rejected():
    source = observation()
    full = feedback_view(source, 'class_feedback')
    assert full['evaluations'][0]['class_violations']
    full['evaluations'].clear()
    assert source['evaluations']
    with pytest.raises(ValueError): feedback_view(source, 'unknown')


def test_cli_freezes_condition_and_persists_masked_view(monkeypatch, tmp_path):
    import run_feedback_ablation as cli
    table_dir = tmp_path/'table'
    table_dir.mkdir()
    (table_dir/'manifest.json').write_text('{}')
    out = tmp_path/'session'
    monkeypatch.setattr(cli, 'load_table', lambda path: {})
    monkeypatch.setattr(cli, 'start', lambda table: {'revision': 0})
    monkeypatch.setattr(cli, 'compact_observation', lambda state: observation())
    monkeypatch.setattr(sys, 'argv', ['run_feedback_ablation.py','init','--condition','aggregate',
                                    '--table',str(table_dir),'--dir',str(out)])
    cli.main()
    saved = json.loads((out/'delivered-0000.json').read_text())
    assert 'SECRET_CLASS' not in str(saved)
    assert (out/'state-0000.json').exists()
    with pytest.raises(FileExistsError): cli.main()
    (table_dir/'manifest.json').write_text('{"changed":true}')
    monkeypatch.setattr(sys, 'argv', ['run_feedback_ablation.py','observe',
                                    '--table',str(table_dir),'--dir',str(out)])
    with pytest.raises(ValueError, match='changed'): cli.main()
