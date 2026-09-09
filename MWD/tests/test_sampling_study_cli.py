import sys

import pandas as pd


def test_study_freezes_protocol_before_fit_and_only_loads_train(monkeypatch, tmp_path):
    import run_sampling_study as cli

    output = tmp_path / 'study'
    reads, fits = [], []
    monkeypatch.setattr(sys, 'argv', ['run_sampling_study.py', '--output-dir', str(output)])
    monkeypatch.setattr(cli, 'load_dataset', lambda path: reads.append(path) or pd.DataFrame())

    def evaluate(frame, *, model_name, training_strategy, validation_size, seed):
        assert (output / 'protocol.json').exists()
        assert validation_size == .2
        fits.append((model_name, training_strategy, seed))
        return {'model': model_name, 'seed': seed, 'validation': {
            'accuracy': .7, 'balanced_accuracy': .7, 'macro_f1': .7},
            'transition_zone': {'macro_f1': .6}, 'selection_score': .67}

    monkeypatch.setattr(cli, 'run_validation_once', evaluate)
    cli.main()
    assert len(reads) == 1 and reads[0].name.endswith('_train.csv')
    assert len(fits) == 40 and len(set(fits)) == 40
    assert len(pd.read_csv(output / 'summary.csv')) == 8
