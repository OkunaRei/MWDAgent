import json
import sys

import pytest


def setup_cli(monkeypatch, tmp_path):
    import run_mwd_agent as cli

    train = tmp_path / 'public_train.csv'
    train.write_text('feature,Rock\n1,A\n')
    monkeypatch.setattr(cli, 'TRAIN_SOURCE', train)
    monkeypatch.setattr(cli, 'code_hashes', lambda: {'executor': 'hash'})
    monkeypatch.setattr(cli, 'dependencies', lambda: {'test': '1'})
    monkeypatch.setattr(cli, 'observe', lambda state: {'revision': state['revision']})
    monkeypatch.setattr(cli, 'observation_digest', lambda state: str(state['revision']))
    return cli, train


def test_cli_pins_train_source_and_persists_immutable_revisions(monkeypatch, tmp_path):
    cli, train = setup_cli(monkeypatch, tmp_path)
    reads = []
    monkeypatch.setattr(cli, 'load_dataset', lambda path: reads.append(path))
    monkeypatch.setattr(cli, 'initialize', lambda frame, **options: {'revision': 0})
    monkeypatch.setattr(cli, 'apply_action', lambda state, frame, action: {**state, 'revision': 1})
    session = tmp_path / 'session'
    monkeypatch.setattr(sys, 'argv', ['run_mwd_agent.py', 'init', '--session', str(session)])
    cli.main()
    original = (session / 'state-0000.json').read_bytes()
    action = tmp_path / 'action.json'
    action.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['run_mwd_agent.py', 'act', '--session', str(session), '--action', str(action)])
    cli.main()
    assert reads == [train, train]
    assert (session / 'state-0000.json').read_bytes() == original
    assert (session / 'state-0001.json').exists()
    assert not (session / '.lock').exists()
    train.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        cli.main()
    assert reads == [train, train]


def test_observe_does_not_load_data_and_lock_blocks_actions(monkeypatch, tmp_path):
    cli, _ = setup_cli(monkeypatch, tmp_path)
    session = tmp_path / 'session'
    session.mkdir()
    (session / 'state-0000.json').write_text(json.dumps({'revision': 0}))
    monkeypatch.setattr(cli, 'load_dataset', lambda path: pytest.fail('observe must not read data'))
    monkeypatch.setattr(sys, 'argv', ['run_mwd_agent.py', 'observe', '--session', str(session)])
    cli.main()
    (session / '.lock').mkdir()
    with pytest.raises(ValueError, match='locked'):
        with cli.session_lock(session):
            pytest.fail('must not acquire an occupied lock')
