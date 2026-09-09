from copy import deepcopy
import pandas as pd
import pytest
from src.agent_loop import SEED_LIST, apply_action, observation_digest, observe
from replay_agent_evidence import build_replay


def runs(strategy):
    return [dict(seed=seed, training_strategy=strategy, model='lightgbm', feature_group='all_48',
        validation_size=.2, selection_score=.7, validation=dict(accuracy=.7, macro_f1=.7, balanced_accuracy=.7),
        transition_zone=dict(macro_f1=.7), diagnostics=dict(classes=['A','B'], validation_indices=[0,1],
        overall=dict(confusion_matrix=[[1,0],[0,1]]), ordinary=dict(confusion_matrix=[[1,0],[0,0]]),
        transition_zone=dict(confusion_matrix=[[0,0],[0,1]]))) for seed in SEED_LIST]


def test_replay_stop_preserves_unspent_budget_and_inputs():
    inputs = [runs(s) for s in ('paper_smote','class_weight','midpoint_smote')]
    before = deepcopy(inputs)
    states = build_replay(*inputs, 'a'*64)
    assert len(states) == 3 and observe(states[-1])['budget_remaining'] == 1
    action = dict(action='stop', reason='Evidence supports stopping', observation_sha256=observation_digest(states[-1]))
    def forbidden(*args, **kwargs):
        pytest.fail('Stop must not evaluate')
    result = apply_action(states[-1], pd.DataFrame(), action, evaluator=forbidden)
    assert result['status'] == 'stopped'
    assert observe(result)['budget_remaining'] == 1
    assert len(result['evaluations']) == 3
    assert inputs == before and states[-1]['status'] == 'active'
    with pytest.raises(ValueError, match='stopped'):
        apply_action(result, pd.DataFrame(), action, evaluator=forbidden)


@pytest.mark.parametrize('change', ['seed','strategy','indices'])
def test_replay_rejects_mixed_evidence(change):
    inputs = [runs(s) for s in ('paper_smote','class_weight','midpoint_smote')]
    if change == 'seed': inputs[1][0]['seed'] = 999
    if change == 'strategy': inputs[1][0]['training_strategy'] = 'original'
    if change == 'indices': inputs[1][0]['diagnostics']['validation_indices'] = [1,0]
    with pytest.raises(ValueError): build_replay(*inputs, 'a'*64)
