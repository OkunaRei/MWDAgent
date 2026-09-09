from copy import deepcopy
import json
import pytest

from src.agent_loop import MIDPOINT_POLICY, observation_digest
from src.benchmark import SEED_LIST
from src.decision_benchmark import start, execute, compact_observation, compare


def table():
    candidates = {}
    for position, candidate in enumerate(MIDPOINT_POLICY['catalog']):
        feature, model, strategy = candidate.split('__')
        matrix = [[8, 2], [2, 8]] if position == 0 else [[9, 1], [1, 9]]
        if position == 2:
            matrix = [[5, 5], [0, 10]]
        candidates[candidate] = [dict(seed=seed, model=model, feature_group=feature,
            training_strategy=strategy, selection_score=.7 if position == 0 else .8,
            validation={'macro_f1': .7 if position == 0 else .8, 'balanced_accuracy': .8},
            transition_zone={'macro_f1': .7 if position == 0 else .8},
            diagnostics={'classes': ['A', 'B'], 'validation_indices': list(range(20)),
                'overall': {'confusion_matrix': matrix},
                'ordinary': {'confusion_matrix': matrix},
                'transition_zone': {'confusion_matrix': [[0, 0], [0, 0]]}}) for seed in SEED_LIST]
    return {'source_sha256': 'a'*64, 'policy': deepcopy(MIDPOINT_POLICY), 'candidates': candidates}


def action(state, index):
    return {'action': 'run_candidate', 'candidate': MIDPOINT_POLICY['catalog'][index],
            'reason': 'comparison', 'hypothesis': 'improve', 'physical_risk': 'offline only',
            'observation_sha256': observation_digest(state)}


def test_hidden_results_guard_and_budget():
    evidence = table()
    initial = start(evidence)
    observed = compact_observation(initial)
    assert len(observed['evaluations']) == 1
    assert '0.8' not in json.dumps(observed['evaluations'])
    assert observed['observation_sha256'] == observation_digest(initial)
    bad = execute(initial, evidence, action(initial, 2))
    assert bad['incumbent'] == initial['incumbent']
    assert 'class_recall_floor' in bad['evaluations'][-1]['reason']
    assert compact_observation(bad)['evaluations'][-1]['class_violations'][0]['class'] == 'A'
    good = execute(bad, evidence, action(bad, 1))
    assert good['incumbent'] == MIDPOINT_POLICY['catalog'][1]
    assert good['status'] == 'stopped'
    with pytest.raises(ValueError):
        execute(good, evidence, action(good, 3))
    with pytest.raises(ValueError, match='Stale'):
        execute(bad, evidence, action(initial, 1))


def test_comparison_requires_stop_and_exhaustive_pairs():
    evidence = table()
    state = start(evidence)
    with pytest.raises(ValueError, match='stopped'):
        compare(state, evidence)
    state = execute(state, evidence, {'action': 'stop', 'reason': 'stop',
                                     'observation_sha256': observation_digest(state)})
    result = compare(state, evidence)
    random = [row for row in result['rows'] if row['method'] == 'random']
    assert len(random) == 42
    assert all(row['unspent_budget'] == 0 for row in random)
    assert result['rows'][0]['unspent_budget'] == 2
    assert any(row['target_attained'] for row in random)


def test_invalid_evidence_rejected_before_execution():
    evidence = table()
    evidence['candidates'][MIDPOINT_POLICY['catalog'][2]][0]['seed'] = 999
    with pytest.raises(ValueError):
        start(evidence)


def test_preparation_freezes_before_training_and_detects_tampering(tmp_path, monkeypatch, capsys):
    import run_decision_benchmark as cli
    evidence = table()
    frozen = {'source_sha256': evidence['source_sha256'], 'code_sha256': {'stub': 'b'*64}, 'packages': {'stub': '1'}}
    monkeypatch.setattr(cli, 'fingerprints', lambda: frozen)
    monkeypatch.setattr(cli, 'load_dataset', lambda path: None)
    directory = tmp_path/'session'
    calls = []
    def train(frame, **options):
        assert (directory/'protocol.json').exists()
        assert not (directory/'manifest.json').exists()
        candidate = '__'.join((options['feature_group'], options['model_name'], options['training_strategy']))
        calls.append((candidate, options['seed']))
        return deepcopy(next(r for r in evidence['candidates'][candidate] if r['seed'] == options['seed']))
    monkeypatch.setattr(cli, 'run_validation_once', train)
    cli.prepare(directory)
    assert len(calls) == 40
    assert 'selection_score' not in capsys.readouterr().out
    assert cli.load_table(directory) == evidence
    with pytest.raises(FileExistsError):
        cli.prepare(directory)
    candidate = directory/'candidate-01.json'
    candidate.write_text(candidate.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        cli.load_table(directory)


def test_cli_persists_exact_action_and_full_observation(tmp_path, monkeypatch, capsys):
    import run_decision_benchmark as cli
    monkeypatch.setattr(cli, 'load_table', lambda directory: table())
    def invoke(command, proposal=None):
        args = ['benchmark', command, '--dir', str(tmp_path)]
        if proposal is not None:
            args += ['--action', json.dumps(proposal)]
        monkeypatch.setattr('sys.argv', args)
        cli.main()
        return json.loads(capsys.readouterr().out)
    baseline = invoke('observe')
    original = (tmp_path/'state-0000.json').read_bytes()
    state = json.loads(original)
    proposal = action(state, 2)
    result = invoke('act', proposal)
    saved = json.loads((tmp_path/'state-0001.json').read_text())
    assert all(saved['actions'][0][k] == v for k, v in proposal.items())
    assert (tmp_path/'state-0000.json').read_bytes() == original
    assert len(json.loads((tmp_path/'observation-0001.json').read_text())['evaluations']) == 2
    assert len(result['evaluations']) == 2
    assert baseline['budget_remaining'] == 2
