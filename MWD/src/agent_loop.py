"""Bounded experiment executor for JSON actions proposed by an external agent."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from time import monotonic

from src.benchmark import SEED_LIST, run_validation_once, summarize_validation_runs


CATALOG = (
    'all_48__lightgbm', 'all_48__extratrees',
    'central_statistics__lightgbm', 'variability_statistics__lightgbm',
)
BASELINE = CATALOG[0]
POLICY = {'seeds': list(SEED_LIST), 'validation_size': .2,
          'selection_weights': [.2, .5, .3], 'catalog': list(CATALOG),
          'minimum_improvement': .001, 'floor_reference': BASELINE,
          'floor_metrics': ['macro_f1_mean', 'transition_macro_f1_mean'],
          'data_scope': 'public_training_validation_only'}

STRATEGY_CATALOG = tuple(f'all_48__{model}__{strategy}'
                         for model in ('lightgbm', 'extratrees')
                         for strategy in ('paper_smote', 'original', 'class_weight'))
STRATEGY_POLICY = {**deepcopy(POLICY), 'catalog': list(STRATEGY_CATALOG),
                   'floor_reference': STRATEGY_CATALOG[0], 'version': 'training-strategies-v1'}
CLASS_GUARD_POLICY = {**deepcopy(STRATEGY_POLICY), 'version': 'training-strategies-v2',
                      'minimum_class_recall_delta': -.05}
MIDPOINT_POLICY = {**deepcopy(CLASS_GUARD_POLICY), 'version': 'training-strategies-v3',
                   'catalog': [*STRATEGY_CATALOG, 'all_48__lightgbm__midpoint_smote',
                               'all_48__extratrees__midpoint_smote']}


def _evaluate(frame, candidate, evaluator):
    started = monotonic()
    parts = candidate.split('__')
    feature_group, model_name = parts[:2]
    options = {'training_strategy': parts[2]} if len(parts) == 3 else {}
    runs, attempted = [], []
    result = {'candidate': candidate, 'accepted': False, 'summary': {}, 'reason': '',
              'status': 'completed', 'error': None}
    try:
        for seed in SEED_LIST:
            attempted.append(seed)
            eval_kwargs = dict(model_name=model_name, feature_group=feature_group, seed=seed,
                               validation_size=.2, selection_weights=(.2, .5, .3), **options)
            if len(parts) == 3:
                eval_kwargs['include_diagnostics'] = True
            run = evaluator(frame.copy(deep=True), **eval_kwargs)
            if options and run.get('training_strategy') != options['training_strategy']:
                raise ValueError('Evaluator returned mismatched training strategy')
            if (run['seed'] != seed or run['model'] != model_name
                    or run['feature_group'] != feature_group):
                raise ValueError('Evaluator returned mismatched candidate or seed')
            metrics = [run['selection_score'], run['validation']['macro_f1'],
                       run['transition_zone']['macro_f1']]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics):
                raise ValueError('Evaluator returned invalid selection metrics')
            json.dumps(run, allow_nan=False)
            runs.append(deepcopy(run))
        result['summary'] = summarize_validation_runs(runs)[model_name]
    except Exception as error:
        result = {**result, 'status': 'failed', 'reason': 'evaluation_failed',
                  'error': f'{type(error).__name__}: {error}'}
    return {**result, 'runs': runs, 'attempted_seeds': attempted,
            'elapsed_seconds': monotonic() - started}


def initialize(frame, *, source_sha256: str, budget: int = 4,
               evaluator=run_validation_once, policy=None) -> dict:
    """Evaluate the mandatory baseline under the fixed validation policy."""
    chosen_policy = POLICY if policy is None else policy
    if chosen_policy not in (POLICY, STRATEGY_POLICY, CLASS_GUARD_POLICY, MIDPOINT_POLICY):
        raise ValueError('Unsupported experiment policy')
    if type(budget) is not int or not 1 <= budget <= len(chosen_policy['catalog']):
        raise ValueError('budget must be an integer within the policy catalogue size')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('source_sha256 must be a lowercase SHA-256 digest')
    baseline_key = chosen_policy['catalog'][0]
    baseline = _evaluate(frame, baseline_key, evaluator)
    if chosen_policy in (CLASS_GUARD_POLICY, MIDPOINT_POLICY) and baseline['status'] == 'completed':
        from src.class_impact import compare_class_impact
        try:
            rows = compare_class_impact(baseline['runs'], baseline['runs'])['class_rows']
            if not any(row['slice'] == 'overall' and row['support'] > 0 for row in rows):
                raise ValueError('No supported baseline classes')
            baseline = {**baseline, 'class_feedback': {'scope': 'overall_each_seed',
                'floor': chosen_policy['minimum_class_recall_delta'], 'violations': [], 'per_class_seed': rows}}
        except ValueError as error:
            baseline = {**baseline, 'status': 'failed', 'reason': 'invalid_class_evidence', 'error': str(error)}
    successful = baseline['status'] == 'completed'
    baseline = {**baseline, 'accepted': successful,
                'reason': 'baseline' if successful else baseline['reason']}
    return {'revision': 0, 'status': 'active' if successful and budget > 1 else 'stopped',
            'budget': budget, 'source_sha256': source_sha256,
            'policy': deepcopy(chosen_policy), 'evaluations': [baseline],
            'incumbent': baseline_key if successful else None, 'actions': []}


def observe(state) -> dict:
    """Expose validation summaries, without detailed sample or model outputs."""
    evaluated = {item['candidate'] for item in state['evaluations']}
    return deepcopy({'revision': state['revision'], 'status': state['status'],
                     'incumbent': state['incumbent'],
                     'source_sha256': state['source_sha256'], 'policy': state['policy'],
                     'budget_remaining': state['budget'] - len(state['evaluations']),
                     'evaluations': [{key: item[key] for key in
                                      ('candidate', 'summary', 'accepted', 'reason', 'status', 'deltas', 'class_feedback')
                                      if key in item}
                                     for item in state['evaluations']],
                     'allowed_candidates': [key for key in state['policy']['catalog'] if key not in evaluated]
                     if state['status'] == 'active' else []})


def observation_digest(state) -> str:
    serialized = json.dumps(observe(state), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _validate_action(state, action):
    if state['policy'] not in (POLICY, STRATEGY_POLICY, CLASS_GUARD_POLICY, MIDPOINT_POLICY):
        raise ValueError('State policy differs from the fixed executor policy')
    if state['status'] != 'active':
        raise ValueError('Session is stopped')
    if not isinstance(action, dict) or action.get('action') not in ('run_candidate', 'stop'):
        raise ValueError('Unsupported action')
    fields = {'action', 'reason', 'observation_sha256'}
    if action['action'] == 'run_candidate':
        fields |= {'candidate', 'hypothesis', 'physical_risk'}
    if set(action) != fields:
        raise ValueError('Action fields do not match the allowed schema')
    if any(not isinstance(action[key], str) or not action[key].strip() for key in fields):
        raise ValueError('Every action field must be a nonempty string')
    if action['observation_sha256'] != observation_digest(state):
        raise ValueError('Stale observation digest')
    if action['action'] == 'run_candidate':
        if len(state['evaluations']) >= state['budget']:
            raise ValueError('Experiment budget exhausted')
        if action['candidate'] not in observe(state)['allowed_candidates']:
            raise ValueError('Candidate is unsupported or already evaluated')


def _decision(state, result):
    if result['status'] == 'failed':
        return result
    baseline = state['evaluations'][0]['summary']
    incumbent = next(item['summary'] for item in state['evaluations']
                     if item['candidate'] == state['incumbent'])
    summary = result['summary']
    deltas = {'selection_score_vs_incumbent': summary['selection_score_mean'] - incumbent['selection_score_mean'],
              'macro_f1_vs_baseline': summary['macro_f1_mean'] - baseline['macro_f1_mean'],
              'transition_macro_f1_vs_baseline': summary['transition_macro_f1_mean'] - baseline['transition_macro_f1_mean']}
    reasons = []
    if summary['selection_score_mean'] <= incumbent['selection_score_mean'] + .001:
        reasons.append('insufficient_improvement')
    if deltas['macro_f1_vs_baseline'] < 0:
        reasons.append('macro_f1_floor')
    if deltas['transition_macro_f1_vs_baseline'] < 0:
        reasons.append('transition_floor')
    if state.get('policy', {}).get('version') in ('training-strategies-v2', 'training-strategies-v3'):
        from src.class_impact import compare_class_impact

        floor = state['policy']['minimum_class_recall_delta']
        try:
            rows = compare_class_impact(state['evaluations'][0]['runs'], result['runs'])['class_rows']
        except ValueError as error:
            return {**result, 'accepted': False, 'status': 'failed',
                    'reason': 'invalid_class_evidence', 'error': str(error)}
        overall = [row for row in rows if row['slice'] == 'overall' and row['recall_delta'] is not None]
        if not overall:
            return {**result, 'accepted': False, 'status': 'failed', 'reason': 'invalid_class_evidence',
                    'error': 'No supported overall classes'}
        violations = [row for row in overall if row['recall_delta'] < floor - 1e-12]
        result = {**result, 'class_feedback': {'scope': 'overall_each_seed', 'floor': floor,
            'violations': violations, 'per_class_seed': rows}}
        if violations:
            reasons.append('class_recall_floor')
        deltas['minimum_class_recall_delta'] = min(row['recall_delta'] for row in overall)
    return {**result, 'accepted': not reasons, 'reason': ','.join(reasons) or 'accepted',
            'deltas': deltas}


def apply_action(state, frame, action, *, evaluator=run_validation_once) -> dict:
    """Validate a proposal and return a new state; invalid proposals do no work."""
    _validate_action(state, action)
    updated = deepcopy(state)
    updated['revision'] += 1
    record = {**deepcopy(action), 'revision': updated['revision'], 'status': 'completed'}
    if action['action'] == 'stop':
        updated['status'] = 'stopped'
    else:
        result = _decision(state, _evaluate(frame, action['candidate'], evaluator))
        updated['evaluations'].append(result)
        record = {**record, 'status': result['status'], 'accepted': result['accepted'],
                  'decision_reason': result['reason']}
        if result['accepted']:
            updated['incumbent'] = result['candidate']
        if len(updated['evaluations']) >= updated['budget']:
            updated['status'] = 'stopped'
    updated['actions'].append(record)
    return updated
