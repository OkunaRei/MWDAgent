import pandas as pd

from src.agent_loop import STRATEGY_POLICY, initialize, apply_action, observation_digest, observe


def test_versioned_strategy_actions_execute_and_promote_under_partial_budget():
    calls = []
    def evaluator(frame, **kwargs):
        strategy = kwargs['training_strategy']
        calls.append(strategy)
        metric = .8 if strategy == 'class_weight' else .7
        return {'model': kwargs['model_name'], 'feature_group': kwargs['feature_group'],
                'training_strategy': strategy, 'seed': kwargs['seed'],
                'selection_score': metric, 'validation': {'accuracy': metric,
                    'balanced_accuracy': metric, 'macro_f1': metric},
                'transition_zone': {'macro_f1': metric}, 'diagnostics': {
                    'classes': ['A', 'B'], 'validation_indices': [0, 1],
                    'overall': {'classification_report': {'A': {'recall': 1.}, 'B': {'recall': 1.}}, 'confusion_matrix': [[1,0],[0,1]]},
                    'ordinary': {'classification_report': {'A': {'recall': 1.}, 'B': {'recall': 1.}}, 'confusion_matrix': [[1,0],[0,0]]},
                    'transition_zone': {'classification_report': {'A': {'recall': 1.}, 'B': {'recall': 1.}}, 'confusion_matrix': [[0,0],[0,1]]}}}
    state = initialize(pd.DataFrame(), source_sha256='a'*64, budget=3,
                       policy=STRATEGY_POLICY, evaluator=evaluator)
    assert len(observe(state)['allowed_candidates']) == 5
    updated = apply_action(state, pd.DataFrame(), {'action': 'run_candidate',
        'candidate': 'all_48__lightgbm__class_weight', 'reason': 'retain real rows',
        'hypothesis': 'weighted loss', 'physical_risk': 'probability change',
        'observation_sha256': observation_digest(state)}, evaluator=evaluator)
    assert updated['incumbent'] == 'all_48__lightgbm__class_weight'
    assert updated['evaluations'][-1]['accepted']
    assert calls == ['paper_smote']*5 + ['class_weight']*5
    assert observe(updated)['budget_remaining'] == 1
