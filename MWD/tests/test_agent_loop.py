from copy import deepcopy

import pandas as pd
import pytest

from src.agent_loop import apply_action, initialize, observation_digest, observe
from src.benchmark import SEED_LIST


def evaluator(score=.7, macro=.7, transition=.6):
    def run(frame, **kwargs):
        assert kwargs['validation_size'] == .2
        assert kwargs['selection_weights'] == (.2, .5, .3)
        return {'model': kwargs['model_name'], 'feature_group': kwargs['feature_group'],
                'seed': kwargs['seed'], 'selection_score': score,
                'validation': {'accuracy': macro, 'balanced_accuracy': macro, 'macro_f1': macro},
                'transition_zone': {'macro_f1': transition}}
    return run


def start(budget=4):
    return initialize(pd.DataFrame(), source_sha256='a' * 64, budget=budget, evaluator=evaluator())


def action(state, candidate='all_48__extratrees'):
    return {'action': 'run_candidate', 'candidate': candidate, 'reason': 'Compare model family',
            'hypothesis': 'Ensembling may improve validation score',
            'physical_risk': 'No field inference is authorized',
            'observation_sha256': observation_digest(state)}


def test_baseline_and_improvement_are_immutable():
    state = start()
    previous = deepcopy(state)
    assert state['evaluations'][0]['attempted_seeds'] == list(SEED_LIST)
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator(.8, .8, .7))
    assert state == previous
    assert updated['revision'] == 1
    assert updated['incumbent'] == 'all_48__extratrees'
    assert updated['evaluations'][-1]['accepted'] is True
    assert observe(updated)['budget_remaining'] == 2


@pytest.mark.parametrize('score,macro,transition,reason', [
    (.8, .8, .59, 'transition_floor'), (.8, .69, .7, 'macro_f1_floor'),
    (.7, .8, .7, 'insufficient_improvement'),
])
def test_candidate_guards(score, macro, transition, reason):
    state = start()
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator(score, macro, transition))
    assert updated['incumbent'] == state['incumbent']
    assert reason in updated['evaluations'][-1]['reason']


@pytest.mark.parametrize('change', [
    {'candidate': 'all_48__lightgbm'}, {'candidate': 'unknown'},
    {'observation_sha256': 'stale'}, {'command': 'anything'}, {'path': 'test.csv'},
    {'reason': ''}, {'physical_risk': ''}, {'action': 'read_test'},
])
def test_invalid_actions_do_not_execute_or_mutate(change):
    state = start()
    previous = deepcopy(state)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid action executed')
    with pytest.raises(ValueError):
        apply_action(state, pd.DataFrame(), {**action(state), **change}, evaluator=forbidden)
    assert state == previous


def test_budget_exhaustion_and_explicit_stop():
    state = start(budget=2)
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator())
    assert updated['status'] == 'stopped'
    assert observe(updated)['budget_remaining'] == 0
    with pytest.raises(ValueError):
        apply_action(updated, pd.DataFrame(), action(updated))
    stopped = apply_action(state, pd.DataFrame(), {'action': 'stop', 'reason': 'Evidence sufficient',
                                                  'observation_sha256': observation_digest(state)})
    assert stopped['status'] == 'stopped'
    assert len(stopped['evaluations']) == 1


def test_failure_consumes_budget_and_records_attempts():
    state = start(budget=2)
    previous = deepcopy(state)
    def fail(frame, **kwargs):
        if kwargs['seed'] == SEED_LIST[1]:
            raise RuntimeError('model failed')
        return evaluator()(frame, **kwargs)
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=fail)
    assert state == previous
    failed = updated['evaluations'][-1]
    assert failed['status'] == 'failed'
    assert failed['attempted_seeds'] == list(SEED_LIST[:2])
    assert len(failed['runs']) == 1
    assert 'model failed' in failed['error']
    assert updated['actions'][-1]['status'] == 'failed'
    assert updated['status'] == 'stopped'


def test_observation_is_detached_and_has_no_full_runs():
    state = start()
    original = deepcopy(state)
    observed = observe(state)
    assert 'runs' not in observed['evaluations'][0]
    assert 'transition_macro_f1_mean' in observed['evaluations'][0]['summary']
    observed['allowed_candidates'].clear()
    assert state == original


@pytest.mark.parametrize('budget', [0, -1, True, 5, 2.5])
def test_budget_is_bounded(budget):
    with pytest.raises(ValueError):
        start(budget)


def test_failed_baseline_and_single_evaluation_budget_stop():
    def fail(*args, **kwargs):
        raise RuntimeError('baseline unavailable')
    state = initialize(pd.DataFrame(), source_sha256='a' * 64, evaluator=fail)
    assert state['status'] == 'stopped'
    assert state['incumbent'] is None
    assert state['evaluations'][0]['status'] == 'failed'
    assert start(1)['status'] == 'stopped'


def test_improvement_boundary_is_strict():
    state = start()
    score = state['evaluations'][0]['summary']['selection_score_mean'] + .001
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator(score))
    assert updated['evaluations'][-1]['accepted'] is False


def test_nonfinite_result_is_a_recorded_failure():
    state = start()
    updated = apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator(float('nan')))
    assert updated['evaluations'][-1]['status'] == 'failed'
    assert updated['evaluations'][-1]['attempted_seeds'] == [SEED_LIST[0]]


def test_mutating_evaluator_cannot_modify_input_frame():
    frame = pd.DataFrame({'value': [1]})
    def mutate(copy, **kwargs):
        copy.loc[0, 'value'] = 99
        return evaluator()(copy, **kwargs)
    initialize(frame, source_sha256='a' * 64, evaluator=mutate)
    assert frame.loc[0, 'value'] == 1


def test_source_digest_and_policy_validation():
    with pytest.raises(ValueError, match='source_sha256'):
        initialize(pd.DataFrame(), source_sha256='invalid', evaluator=evaluator())
    state = start()
    state['policy']['seeds'] = [42]
    with pytest.raises(ValueError, match='policy'):
        apply_action(state, pd.DataFrame(), action(state), evaluator=evaluator())
