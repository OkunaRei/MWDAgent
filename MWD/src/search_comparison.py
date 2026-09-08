"""Exact, validation-only counterfactual replay of bounded candidate orders."""
from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import math
from statistics import mean

from src.agent_loop import BASELINE, CATALOG, POLICY, _decision
from src.benchmark import summarize_candidate_runs


def _number(value, *, probability=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (probability and value > 1)):
        raise ValueError('Invalid numeric evidence')
    return value


def _validate_runs(item):
    candidate = item['candidate']
    feature_group, model = candidate.split('__')
    runs = item['runs']
    if (item['status'] != 'completed' or item.get('error') is not None
            or item['attempted_seeds'] != POLICY['seeds']
            or [run['seed'] for run in runs] != POLICY['seeds']):
        raise ValueError('Every candidate requires all five successful policy seeds')
    for run in runs:
        if run['model'] != model or run['feature_group'] != feature_group:
            raise ValueError('Candidate run identity mismatch')
        balanced = _number(run['validation']['balanced_accuracy'])
        macro = _number(run['validation']['macro_f1'])
        transition = _number(run['transition_zone']['macro_f1'])
        _number(run['validation']['accuracy'])
        score = _number(run['selection_score'])
        expected = sum(weight * value for weight, value in
                       zip(POLICY['selection_weights'], (balanced, macro, transition)))
        if not math.isclose(score, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Run selection score differs from the fixed policy')
    recomputed = summarize_candidate_runs(runs)[candidate]
    if set(item['summary']) != set(recomputed):
        raise ValueError('Summary fields differ from recomputed evidence')
    for key, value in recomputed.items():
        actual = _number(item['summary'][key], probability=key != 'runs')
        if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Summary differs from recomputed evidence')
    _number(item['elapsed_seconds'], probability=False)


def _validate(state):
    try:
        if (state['policy'] != POLICY or state['status'] != 'stopped'
                or type(state['budget']) is not int or state['budget'] != len(CATALOG)):
            raise ValueError('Replay requires a stopped full-budget fixed-policy state')
        items = state['evaluations']
        keys = [item['candidate'] for item in items]
        if len(keys) != len(CATALOG) or set(keys) != set(CATALOG) or keys[0] != BASELINE:
            raise ValueError('Replay requires each catalogue candidate exactly once, baseline first')
        for item in items:
            _validate_runs(item)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError('Malformed replay evidence') from error


def _trajectory(order, by_candidate):
    baseline = by_candidate[BASELINE]
    baseline_score = baseline['summary']['selection_score_mean']
    target = baseline_score + POLICY['minimum_improvement']
    replay = {'evaluations': [], 'incumbent': BASELINE}
    steps = []
    elapsed = 0.0
    for index, candidate in enumerate(order):
        item = by_candidate[candidate]
        decision = ({**item, 'accepted': True, 'reason': 'baseline'} if index == 0
                    else _decision(replay, item))
        incumbent = candidate if decision['accepted'] else replay['incumbent']
        replay = {'evaluations': [*replay['evaluations'], decision], 'incumbent': incumbent}
        summary = by_candidate[incumbent]['summary']
        elapsed += item['elapsed_seconds']
        steps.append({
            'candidate_count': index + 1, 'training_attempts': (index + 1) * len(POLICY['seeds']),
            'replayed_evaluation_seconds': elapsed, 'candidate': candidate, 'incumbent': incumbent,
            'selection_score': summary['selection_score_mean'], 'macro_f1': summary['macro_f1_mean'],
            'transition_macro_f1': summary['transition_macro_f1_mean'],
            'accepted': decision['accepted'], 'reason': decision['reason'],
            'floor_violations': [reason for reason in decision['reason'].split(',')
                                 if reason in ('macro_f1_floor', 'transition_floor')],
            'gain_vs_baseline': summary['selection_score_mean'] - baseline_score,
            'target_achieved': summary['selection_score_mean'] > target,
        })
    return {'order': list(order), 'steps': steps, 'final': deepcopy(steps[-1])}


def _stats(values):
    return {'mean': mean(values), 'min': min(values), 'max': max(values)}


def _budget_summary(fixed, host, random):
    rows = []
    for index, host_step in enumerate(host['steps']):
        samples = [trajectory['steps'][index] for trajectory in random]
        random_summary = {key: _stats([sample[key] for sample in samples]) for key in
                          ('selection_score', 'gain_vs_baseline', 'replayed_evaluation_seconds')}
        random_summary = {**random_summary, 'permutation_count': len(samples),
                          'probability_beating_host': mean(
                              sample['selection_score'] > host_step['selection_score'] for sample in samples),
                          'target_achieved_fraction': mean(sample['target_achieved'] for sample in samples)}
        rows.append({'candidate_count': index + 1, 'training_attempts': host_step['training_attempts'],
                     'fixed_order': deepcopy(fixed['steps'][index]), 'host_agent': deepcopy(host_step),
                     'random_search': random_summary})
    return rows


def compare_search_orders(state) -> dict:
    """Replay all six nonbaseline permutations without fitting or loading data.

    Random-order probabilities are exact over this finite catalogue. They are
    not uncertainty estimates over independent agent runs or data splits.
    """
    _validate(state)
    by_candidate = {item['candidate']: item for item in state['evaluations']}
    fixed = _trajectory(CATALOG, by_candidate)
    host = _trajectory(tuple(by_candidate), by_candidate)
    random = [_trajectory((BASELINE, *order), by_candidate) for order in permutations(CATALOG[1:])]
    baseline = by_candidate[BASELINE]['summary']
    eligible = [key for key in CATALOG if all(
        by_candidate[key]['summary'][metric] >= baseline[metric] for metric in POLICY['floor_metrics'])]
    best = max(eligible, key=lambda key: by_candidate[key]['summary']['selection_score_mean'])
    improving = sum(by_candidate[key]['summary']['selection_score_mean'] >
                    baseline['selection_score_mean'] + POLICY['minimum_improvement'] for key in eligible)
    return {
        'schema_version': 1, 'mode': 'counterfactual_replay',
        'source_sha256': state['source_sha256'], 'policy': deepcopy(POLICY),
        'timing_note': 'REPLAY: cumulative recorded candidate evaluation seconds; no new training or agent overhead.',
        'statistical_note': 'Six exact catalogue permutations, not independent LLM runs; no population significance claim.',
        'candidate_results': [{'candidate': key, 'summary': deepcopy(by_candidate[key]['summary']),
                               'elapsed_seconds': by_candidate[key]['elapsed_seconds']} for key in CATALOG],
        'trajectories': {'fixed_order': fixed, 'host_agent': host, 'random_search': random},
        'budget_summary': _budget_summary(fixed, host, random),
        'oracle': {'candidate': best, 'selection_score': by_candidate[best]['summary']['selection_score_mean'],
                   'improving_candidate_count': improving, 'no_improvement_opportunities': improving == 0,
                   'note': 'Future-aware static best satisfying baseline floors; analysis only, never a routing policy.'},
    }
