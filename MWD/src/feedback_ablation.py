"""Mask diagnostic detail while preserving metrics, gate outcome and policy."""
from copy import deepcopy


def feedback_view(observation, condition):
    if condition not in ('aggregate', 'class_feedback'):
        raise ValueError('Unknown feedback condition')
    result = deepcopy(observation)
    if condition == 'aggregate':
        result['evaluations'] = [{
            **{key: value for key, value in item.items() if key not in ('class_violations', 'reason')},
            'reason': ('evaluation_failed' if item['status'] != 'completed' else
                       'promoted' if item['accepted'] else 'not_promoted')}
            for item in result['evaluations']]
    return {**result, 'feedback_condition': condition}
