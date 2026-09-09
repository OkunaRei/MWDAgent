import sys

import pandas as pd


def test_impact_cli_runs_only_paired_strategies_on_training_source(monkeypatch, tmp_path):
    import analyze_strategy_impact as cli

    out = tmp_path / 'out'
    calls, reads = [], []
    monkeypatch.setattr(sys, 'argv', ['analyze_strategy_impact.py', '--output-dir', str(out)])
    monkeypatch.setattr(cli, 'load_dataset', lambda path: reads.append(path) or pd.DataFrame())

    def evaluate(frame, **kwargs):
        assert (out / 'protocol.json').exists()
        assert kwargs['include_diagnostics'] is True
        assert kwargs['model_name'] == 'lightgbm'
        calls.append((kwargs['training_strategy'], kwargs['seed']))
        return kwargs

    monkeypatch.setattr(cli, 'run_validation_once', evaluate)
    monkeypatch.setattr(cli, 'compare_class_impact', lambda a, b: {'class_rows': [], 'confusion_rows': [], 'paired_rows': []})
    cli.main()
    assert calls == [(s, seed) for s in ('paper_smote', 'class_weight') for seed in cli.SEED_LIST]
    assert len(reads) == 1 and reads[0].name.endswith('_train.csv')
    assert (out / 'impact.json').exists()
