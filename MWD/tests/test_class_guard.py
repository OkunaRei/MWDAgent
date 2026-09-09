from copy import deepcopy
import pandas as pd
import pytest
from src.agent_loop import MIDPOINT_POLICY, CLASS_GUARD_POLICY, STRATEGY_POLICY, initialize, apply_action, observe, observation_digest


def evaluator(drop=0, corrupt=False):
    def run(frame, **kw):
        candidate = kw['training_strategy'] == 'class_weight'
        n = drop if candidate else 0
        matrix = [[20, 0], [n, 20-n]]
        result = dict(seed=kw['seed'], model=kw['model_name'], feature_group=kw['feature_group'],
            training_strategy=kw['training_strategy'], selection_score=.8 if candidate else .7,
            validation=dict(accuracy=.8, balanced_accuracy=.8, macro_f1=.8 if candidate else .7),
            transition_zone=dict(macro_f1=.8 if candidate else .7),
            diagnostics=dict(classes=['A','B'], validation_indices=list(range(40)),
                overall=dict(confusion_matrix=matrix), ordinary=dict(confusion_matrix=matrix),
                transition_zone=dict(confusion_matrix=[[0,0],[0,0]])))
        if corrupt:
            del result['diagnostics']
        return result
    return run


def advance(state, eval_fn):
    return apply_action(state, pd.DataFrame(), dict(action='run_candidate',
        candidate='all_48__lightgbm__class_weight', reason='test', hypothesis='test',
        physical_risk='test', observation_sha256=observation_digest(state)), evaluator=eval_fn)


@pytest.mark.parametrize('drop,accepted', [(0, True),(1, True),(2, False)])
def test_guard_boundary_and_observation(drop, accepted):
    state = initialize(pd.DataFrame(), source_sha256='a'*64, budget=2,
                       policy=CLASS_GUARD_POLICY, evaluator=evaluator())
    before = deepcopy(state)
    updated = advance(state, evaluator(drop))
    assert updated['evaluations'][-1]['accepted'] is accepted
    feedback = observe(updated)['evaluations'][-1]['class_feedback']
    assert len(feedback['violations']) == (0 if accepted else 5)
    assert feedback['per_class_seed'][1]['support'] == 20
    feedback['per_class_seed'].clear()
    assert updated['evaluations'][-1]['class_feedback']['per_class_seed']
    assert state == before


def test_invalid_evidence_fails_closed_and_consumes_budget():
    state = initialize(pd.DataFrame(), source_sha256='a'*64, budget=2,
                       policy=CLASS_GUARD_POLICY, evaluator=evaluator())
    updated = advance(state, evaluator(corrupt=True))
    assert updated['status'] == 'stopped'
    assert updated['evaluations'][-1]['reason'] == 'invalid_class_evidence'
    assert updated['incumbent'] == state['incumbent']
    failed = initialize(pd.DataFrame(), source_sha256='a'*64,
                        policy=CLASS_GUARD_POLICY, evaluator=evaluator(corrupt=True))
    assert failed['incumbent'] is None and failed['status'] == 'stopped'


def test_original_v1_semantics_preserved():
    assert 'minimum_class_recall_delta' not in STRATEGY_POLICY
    state = initialize(pd.DataFrame(), source_sha256='a'*64,
                       policy=STRATEGY_POLICY, evaluator=evaluator())
    assert advance(state, evaluator(10))['evaluations'][-1]['accepted']


def test_v3_midpoint_action_preserves_guard_and_old_catalogues():
    calls = []
    def run(frame, **kw):
        calls.append(kw['training_strategy'])
        translated = {**kw, 'training_strategy': 'class_weight' if kw['training_strategy'] == 'midpoint_smote' else kw['training_strategy']}
        result = evaluator(3)(frame, **translated)
        return {**result, 'training_strategy': kw['training_strategy']}
    state = initialize(pd.DataFrame(), source_sha256='a'*64, budget=2,
                       policy=MIDPOINT_POLICY, evaluator=run)
    updated = apply_action(state, pd.DataFrame(), dict(action='run_candidate',
        candidate='all_48__lightgbm__midpoint_smote', reason='test', hypothesis='test',
        physical_risk='test', observation_sha256=observation_digest(state)), evaluator=run)
    assert updated['evaluations'][-1]['reason'] == 'class_recall_floor'
    assert updated['status'] == 'stopped'
    assert calls == ['paper_smote']*5 + ['midpoint_smote']*5
    assert len(CLASS_GUARD_POLICY['catalog']) == 6
    assert len(MIDPOINT_POLICY['catalog']) == 8
