"""Bounded hidden-result decisions and exhaustive cached-compute comparisons."""
from copy import deepcopy
from itertools import permutations
import math

import pandas as pd

from src.agent_loop import MIDPOINT_POLICY, initialize, apply_action, observe, observation_digest
from src.benchmark import SEED_LIST
from src.class_impact import compare_class_impact


def validate_table(table):
    if table['policy'] != MIDPOINT_POLICY or set(table['candidates']) != set(MIDPOINT_POLICY['catalog']):
        raise ValueError('Table must cover exactly the frozen v3 catalogue')
    baseline = table['candidates'][MIDPOINT_POLICY['catalog'][0]]
    for candidate, runs in table['candidates'].items():
        feature, model, strategy = candidate.split('__')
        if len(runs) != len(SEED_LIST) or {r['seed'] for r in runs} != set(SEED_LIST):
            raise ValueError('Every candidate requires the same five seeds')
        for run in runs:
            if (run['model'], run['feature_group'], run['training_strategy']) != (model, feature, strategy):
                raise ValueError('Candidate identity mismatch')
            values = [run['selection_score'], run['validation']['macro_f1'], run['transition_zone']['macro_f1']]
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError('Invalid selection metrics')
        compare_class_impact(baseline, runs)


def table_evaluator(table):
    def evaluate(frame, **options):
        candidate = '__'.join((options['feature_group'], options['model_name'], options['training_strategy']))
        if (options['validation_size'] != .2 or tuple(options['selection_weights']) != (.2, .5, .3)
                or options['include_diagnostics'] is not True):
            raise ValueError('Cached evaluation policy mismatch')
        return deepcopy(next(run for run in table['candidates'][candidate] if run['seed'] == options['seed']))
    return evaluate


def start(table):
    validate_table(table)
    return initialize(pd.DataFrame(), source_sha256=table['source_sha256'], budget=3,
                      policy=MIDPOINT_POLICY, evaluator=table_evaluator(table))


def execute(state, table, action):
    if state['source_sha256'] != table['source_sha256'] or state['policy'] != table['policy']:
        raise ValueError('Session and table provenance mismatch')
    return apply_action(state, pd.DataFrame(), action, evaluator=table_evaluator(table))


def compact_observation(state):
    full = observe(state)
    evaluations = []
    for item in full['evaluations']:
        grouped = {}
        for row in item.get('class_feedback', {}).get('violations', []):
            name = row['class']
            previous = grouped.get(name, {'class': name, 'minimum_recall_delta': 0, 'seed_supports': []})
            grouped[name] = {'class': name,
                'minimum_recall_delta': min(previous['minimum_recall_delta'], row['recall_delta']),
                'seed_supports': [*previous['seed_supports'], {'seed': row['seed'], 'support': row['support']}]}
        evaluations.append({key: item[key] for key in ('candidate', 'accepted', 'reason', 'status')})
        evaluations[-1] = {**evaluations[-1], 'summary': {key: item['summary'][key] for key in
            ('selection_score_mean', 'macro_f1_mean', 'transition_macro_f1_mean') if key in item['summary']},
            'class_violations': list(grouped.values())}
    return {**full, 'evaluations': evaluations, 'observation_sha256': observation_digest(state)}


def _replay(table, order):
    state = start(table)
    for candidate in order:
        state = execute(state, table, {'action': 'run_candidate', 'candidate': candidate,
            'reason': 'Frozen nonadaptive order', 'hypothesis': 'Offline catalogue comparison',
            'physical_risk': 'No deployment', 'observation_sha256': observation_digest(state)})
    return state


def _row(state, method, index):
    incumbent = next(x for x in state['evaluations'] if x['candidate'] == state['incumbent'])
    score = incumbent['summary']['selection_score_mean']
    baseline = state['evaluations'][0]['summary']['selection_score_mean']
    return {'method': method, 'order_id': index, 'candidate_order': [x['candidate'] for x in state['evaluations']],
            'incumbent': state['incumbent'], 'final_score': score, 'score_gain': score - baseline,
            'evaluations_used': len(state['evaluations']), 'unspent_budget': state['budget'] - len(state['evaluations']),
            'target_attained': score > baseline + MIDPOINT_POLICY['minimum_improvement']}


def compare(state, table):
    if state['status'] != 'stopped':
        raise ValueError('Host session must be stopped before comparison')
    if state['budget'] != 3 or state['source_sha256'] != table['source_sha256']:
        raise ValueError('Host session does not match comparison protocol')
    rows = [_row(state, 'host', 0), _row(_replay(table, MIDPOINT_POLICY['catalog'][1:3]), 'fixed', 0)]
    rows.extend(_row(_replay(table, order), 'random', i)
                for i, order in enumerate(permutations(MIDPOINT_POLICY['catalog'][1:], 2)))
    random = rows[2:]
    return {'scope': 'cached_compute_replay_on_known_dataset; no independent research replication',
            'budget_including_baseline': 3, 'random_order_count': len(random),
            'random_mean_final_score': sum(r['final_score'] for r in random) / len(random),
            'random_target_attainment_rate': sum(r['target_attained'] for r in random) / len(random),
            'rows': rows}
