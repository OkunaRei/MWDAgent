"""Assemble retrospective review evidence without inventing quality labels."""
from collections import Counter, defaultdict

import numpy as np


def _validate(record, source):
    index, seed = record['row_index'], record['seed']
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (index, seed)):
        raise ValueError('Row index and seed must be integers')
    if index not in source.index:
        raise ValueError('Row index does not exist in source')
    row = source.loc[index]
    if record['actual_class'] != row['Rock'] or record['transition_zone'] != row['transition_zone']:
        raise ValueError('Source labels differ from prediction evidence')
    for name in ('correct', 'confidence_accepted', 'dual_accepted', 'extra_rejected', 'transition_zone'):
        if not isinstance(record[name], bool):
            raise ValueError('Decision flags must be booleans')
    if record['correct'] != (record['actual_class'] == record['predicted_class']):
        raise ValueError('Correctness flag disagrees with classes')
    if record['extra_rejected'] != (record['confidence_accepted'] and not record['dual_accepted']):
        raise ValueError('Extra rejection disagrees with gate flags')
    if record['dual_accepted'] and not record['confidence_accepted']:
        raise ValueError('Dual acceptance must imply confidence acceptance')
    for name in ('quality', 'confidence'):
        if not np.isfinite(record[name]) or not 0 <= record[name] <= 1:
            raise ValueError('Scores must be finite in [0, 1]')
    features = record['anomalous_features']
    if (not isinstance(features, list) or not all(isinstance(name, str) for name in features)
            or len(set(features)) != len(features)):
        raise ValueError('Anomalous features must be a unique list of names')


def _item(index, records, source):
    first = records[0]
    return {
        'row_index': index, 'actual_class': first['actual_class'],
        'transition_zone': first['transition_zone'],
        **{name: source.loc[index, name] for name in ('Tunnel', 'PegStart', 'PegEnd')},
        'validation_occurrences': len(records),
        'unique_seed_count': len({record['seed'] for record in records}),
        'seeds': sorted(record['seed'] for record in records),
        'error_occurrences': sum(not record['correct'] for record in records),
        'confident_error_occurrences': sum(record['confidence_accepted'] and not record['correct'] for record in records),
        'extra_reviewed_occurrences': sum(record['extra_rejected'] for record in records),
        'extra_correct_occurrences': sum(record['extra_rejected'] and record['correct'] for record in records),
        'confidence_review_occurrences': sum(not record['confidence_accepted'] for record in records),
        'predicted_class_counts': dict(Counter(record['predicted_class'] for record in records)),
        'anomalous_feature_counts': dict(Counter(name for record in records for name in record['anomalous_features'])),
        'review_status': 'pending', 'quality_label': '', 'evidence_reference': '',
        'reviewer': '', 'reviewed_at': '', 'notes': '',
    }


def build_review_queue(records, source):
    if not source.index.is_unique:
        raise ValueError('Source indices must be unique')
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        _validate(record, source)
        key = record['seed'], record['row_index']
        if key in seen:
            raise ValueError('Duplicate seed/row prediction')
        seen.add(key)
        grouped[record['row_index']].append(record)
    items = [_item(index, group, source) for index, group in grouped.items()
             if any(not record['correct'] or not record['dual_accepted'] for record in group)]
    return sorted(items, key=lambda item: (-item['confident_error_occurrences'],
                  -item['extra_correct_occurrences'], -item['error_occurrences'], item['row_index']))
