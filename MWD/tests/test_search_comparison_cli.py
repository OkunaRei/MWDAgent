from copy import deepcopy
import json
import sys

import pytest

from src.agent_loop import BASELINE, observation_digest


def session_states():
    first = {'revision': 0, 'status': 'active', 'budget': 4, 'incumbent': BASELINE,
             'source_sha256': 'a' * 64, 'policy': {}, 'actions': [],
             'evaluations': [{'candidate': BASELINE, 'summary': {}, 'accepted': True,
                              'reason': 'baseline', 'status': 'completed'}]}
    second = {**deepcopy(first), 'revision': 1,
              'actions': [{'action': 'stop', 'observation_sha256': observation_digest(first)}]}
    return first, second


def test_loader_checks_observation_chain_and_preserved_results(tmp_path):
    from compare_agent_search import load_session

    first, second = session_states()
    (tmp_path / 'state-0000.json').write_text(json.dumps(first))
    path = tmp_path / 'state-0001.json'
    path.write_text(json.dumps(second))
    state, hashes = load_session(tmp_path)
    assert state == second
    assert len(hashes) == 2
    second['evaluations'][0]['summary'] = {'changed': True}
    path.write_text(json.dumps(second))
    with pytest.raises(ValueError, match='rewritten'):
        load_session(tmp_path)
    second = session_states()[1]
    second['actions'][0]['observation_sha256'] = 'wrong'
    path.write_text(json.dumps(second))
    with pytest.raises(ValueError, match='observation'):
        load_session(tmp_path)


@pytest.mark.parametrize('invalid', ['initial_actions', 'stopped'])
def test_impossible_history_rejected(tmp_path, invalid):
    from compare_agent_search import load_session

    first, second = session_states()
    if invalid == 'initial_actions':
        first['actions'] = [{'action': 'stop'}]
    else:
        first['status'] = 'stopped'
    (tmp_path / 'state-0000.json').write_text(json.dumps(first))
    (tmp_path / 'state-0001.json').write_text(json.dumps(second))
    with pytest.raises(ValueError):
        load_session(tmp_path)


def test_cli_generates_replay_artifacts_and_does_not_overwrite(monkeypatch, tmp_path):
    import compare_agent_search as cli

    step = {'selection_score': .7, 'candidate_count': 1}
    result = {'oracle': {'selection_score': .7}, 'budget_summary': [step],
              'trajectories': {'fixed_order': {'steps': [step]}, 'host_agent': {'steps': [step]},
                               'random_search': [{'steps': [step]}]}}
    state = {'source_sha256': 'a' * 64, 'evaluations': [{'candidate': 'baseline',
              'attempted_seeds': [11], 'elapsed_seconds': 1., 'summary': {
                  'selection_score_mean': .7, 'macro_f1_mean': .8, 'transition_macro_f1_mean': .6}}]}
    monkeypatch.setattr(cli, 'load_session', lambda path: (state, {'state': 'hash'}))
    monkeypatch.setattr(cli, 'compare_search_orders', lambda state: result)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['compare_agent_search.py', '--output-dir', str(out)])
    cli.main()
    report = json.loads((out / 'comparison.json').read_text())
    assert report['provenance']['new_model_fits'] == 0
    assert report['provenance']['new_llm_trials'] == 0
    assert (out / 'trajectories.csv').exists()
    before = (out / 'comparison.json').read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert (out / 'comparison.json').read_bytes() == before
