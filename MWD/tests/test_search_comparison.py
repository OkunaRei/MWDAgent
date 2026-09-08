from copy import deepcopy

import pandas as pd
import pytest

from src.agent_loop import BASELINE, CATALOG, apply_action, initialize, observation_digest
from src.search_comparison import compare_search_orders


def completed_state(values=None, order=None):
    values = values or {key: (.7, .6) for key in CATALOG}

    def evaluator(frame, **kwargs):
        key = kwargs['feature_group'] + '__' + kwargs['model_name']
        macro, transition = values[key]
        return {'model': kwargs['model_name'], 'feature_group': kwargs['feature_group'],
                'seed': kwargs['seed'], 'selection_score': .7 * macro + .3 * transition,
                'validation': {'accuracy': macro, 'balanced_accuracy': macro, 'macro_f1': macro},
                'transition_zone': {'macro_f1': transition}}

    state = initialize(pd.DataFrame(), source_sha256='a' * 64, evaluator=evaluator)
    for candidate in order or CATALOG[1:]:
        action = {'action': 'run_candidate', 'candidate': candidate, 'reason': 'comparison',
                  'hypothesis': 'improvement', 'physical_risk': 'validation only',
                  'observation_sha256': observation_digest(state)}
        state = apply_action(state, pd.DataFrame(), action, evaluator=evaluator)
    return {**state, 'evaluations': [{**item, 'elapsed_seconds': 2.0}
                                    for item in state['evaluations']]}


def test_improving_order_prefix_and_transition_floor():
    values = dict(zip(CATALOG, [(.7, .6), (.8, .65), (.9, .7), (.99, .59)]))
    state = completed_state(values, [CATALOG[2], CATALOG[3], CATALOG[1]])
    original = deepcopy(state)
    report = compare_search_orders(state)
    fixed = report['trajectories']['fixed_order']['steps']
    host = report['trajectories']['host_agent']['steps']
    assert host[1]['selection_score'] > fixed[1]['selection_score']
    assert host[2]['accepted'] is False
    assert host[2]['floor_violations'] == ['transition_floor']
    assert fixed[-1]['incumbent'] == CATALOG[2]
    assert len(report['trajectories']['random_search']) == 6
    assert report['budget_summary'][1]['random_search']['target_achieved_fraction'] == pytest.approx(2 / 3)
    assert report['oracle']['improving_candidate_count'] == 2
    assert report['oracle']['candidate'] == CATALOG[2]
    assert host[-1]['training_attempts'] == 20
    assert host[-1]['replayed_evaluation_seconds'] == 8
    assert state == original


def test_baseline_best_ties_have_no_opportunities():
    report = compare_search_orders(completed_state())
    assert report['oracle']['no_improvement_opportunities'] is True
    for row in report['budget_summary']:
        assert row['random_search']['gain_vs_baseline']['max'] == 0
        assert row['random_search']['probability_beating_host'] == 0
        assert row['random_search']['target_achieved_fraction'] == 0
    assert report['trajectories']['fixed_order']['final']['incumbent'] == BASELINE


def test_future_evidence_cannot_change_prior_decisions():
    values = dict(zip(CATALOG, [(.7, .6), (.8, .65), (.9, .7), (.75, .6)]))
    before = compare_search_orders(completed_state(values))
    after = compare_search_orders(completed_state({**values, CATALOG[3]: (.99, .99)}))
    for name in ('host_agent', 'fixed_order'):
        assert before['trajectories'][name]['steps'][:3] == after['trajectories'][name]['steps'][:3]
    for left, right in zip(before['trajectories']['random_search'], after['trajectories']['random_search']):
        prefix = left['order'].index(CATALOG[3])
        assert left['steps'][:prefix] == right['steps'][:prefix]


def test_report_is_detached_from_source_evidence():
    state = completed_state()
    original = deepcopy(state)
    report = compare_search_orders(state)
    report['policy']['seeds'].clear()
    report['candidate_results'][0]['summary'].clear()
    report['trajectories']['host_agent']['final']['incumbent'] = 'changed'
    assert state == original
    assert report['trajectories']['host_agent']['steps'][-1]['incumbent'] == BASELINE


@pytest.mark.parametrize('corruption', [
    'policy', 'duplicate', 'missing', 'failure', 'seed', 'attempted_seeds',
    'summary', 'nan', 'elapsed', 'active', 'budget', 'identity', 'score', 'missing_metric',
])
def test_corrupt_completed_state_rejected(corruption):
    state = completed_state()
    item = state['evaluations'][1]
    if corruption == 'policy':
        state['policy']['minimum_improvement'] = .01
    elif corruption == 'duplicate':
        state['evaluations'][2] = deepcopy(item)
    elif corruption == 'missing':
        state['evaluations'].pop()
    elif corruption == 'failure':
        item['status'] = 'failed'
    elif corruption == 'seed':
        item['runs'][1]['seed'] = item['runs'][0]['seed']
    elif corruption == 'attempted_seeds':
        item['attempted_seeds'].pop()
    elif corruption == 'summary':
        item['summary']['macro_f1_mean'] += .01
    elif corruption == 'nan':
        item['runs'][0]['validation']['macro_f1'] = float('nan')
    elif corruption == 'elapsed':
        item['elapsed_seconds'] = -1
    elif corruption == 'active':
        state['status'] = 'active'
    elif corruption == 'budget':
        state['budget'] = 3
    elif corruption == 'identity':
        item['runs'][0]['model'] = 'other'
    elif corruption == 'score':
        item['runs'][0]['selection_score'] = .2
    else:
        del item['runs'][0]['transition_zone']['macro_f1']
    with pytest.raises(ValueError):
        compare_search_orders(state)
