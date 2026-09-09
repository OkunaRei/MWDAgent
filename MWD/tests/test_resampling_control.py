import sys
import json
import pytest


@pytest.mark.parametrize('mismatch', [False, True])
def test_control_freezes_protocol_matches_counts_and_checks_guard(monkeypatch, tmp_path, mismatch):
    import run_resampling_control as cli
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['run_resampling_control.py', '--output-dir', str(out)])
    reads, calls = [], []
    monkeypatch.setattr(cli, 'load_dataset', lambda path: reads.append(path))
    def evaluate(frame, **kwargs):
        assert (out / 'protocol.json').exists()
        assert kwargs['include_diagnostics']
        candidate = kwargs['training_strategy'] == 'real_resample'
        calls.append((kwargs['seed'], kwargs['training_strategy']))
        score = .8 if candidate else .7
        return {'model': 'lightgbm', 'validation': {'accuracy': score, 'macro_f1': score,
            'balanced_accuracy': score}, 'transition_zone': {'macro_f1': score},
            'selection_score': score, 'fitted_train_class_counts': {'A': 3 if mismatch and candidate else 2}}
    monkeypatch.setattr(cli, 'run_validation_once', evaluate)
    monkeypatch.setattr(cli, 'compare_class_impact', lambda a,b: {'class_rows': [
        {'slice': 'overall', 'recall_delta': -.1}], 'confusion_rows': [], 'paired_rows': []})
    if mismatch:
        with pytest.raises(ValueError, match='counts'):
            cli.main()
    else:
        cli.main()
        result = json.loads((out/'result.json').read_text())
        assert result['aggregate_pass'] and not result['eligible']
    assert len(calls) == 10 and len(set(calls)) == 10
    assert len(reads) == 1 and reads[0].name.endswith('_train.csv')
