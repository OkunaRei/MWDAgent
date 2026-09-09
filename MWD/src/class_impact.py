"""Paired class-level diagnostics on identical validation samples."""
from __future__ import annotations

import math

SLICES = ('overall', 'ordinary', 'transition_zone')


def _index_runs(runs):
    indexed = {}
    for run in runs:
        seed = run['seed']
        if type(seed) is not int or seed in indexed:
            raise ValueError('Seeds must be unique integers')
        indexed = {**indexed, seed: run}
    if not indexed:
        raise ValueError('At least one paired seed is required')
    return indexed


def _matrix(evidence, size):
    matrix = evidence['confusion_matrix']
    if (not isinstance(matrix, list) or len(matrix) != size
            or any(not isinstance(row, list) or len(row) != size for row in matrix)
            or any(type(value) is not int or value < 0 for row in matrix for value in row)):
        raise ValueError('Confusion matrices must be nonnegative integer NxN arrays')
    if 'support' in evidence:
        support = evidence['support']
        if type(support) is not int or support != sum(map(sum, matrix)):
            raise ValueError('Slice support must equal confusion matrix total')
    return matrix


def _diagnostics(run):
    diagnostics = run['diagnostics']
    classes = diagnostics['classes']
    indices = diagnostics['validation_indices']
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError('Classes must be nonempty unique names in a fixed order')
    if (not isinstance(indices, list) or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Validation indices must be unique integers')
    matrices = {name: _matrix(diagnostics[name], len(classes)) for name in SLICES}
    if sum(map(sum, matrices['overall'])) != len(indices):
        raise ValueError('Overall support differs from validation indices')
    if any(matrices['overall'][i][j] != matrices['ordinary'][i][j]
           + matrices['transition_zone'][i][j]
           for i in range(len(classes)) for j in range(len(classes))):
        raise ValueError('Ordinary and transition matrices must partition overall counts')
    return classes, indices, matrices


def _metrics(matrix, index):
    support = sum(matrix[index])
    predicted = sum(row[index] for row in matrix)
    correct = matrix[index][index]
    return {
        'recall': correct / support if support else None,
        'precision': correct / predicted if predicted else None,
        'f1': 2 * correct / (support + predicted) if support + predicted else None,
    }


def _class_rows(seed, name, classes, baseline, candidate):
    rows = []
    for index, label in enumerate(classes):
        before, after = _metrics(baseline, index), _metrics(candidate, index)
        rows.append({
            'seed': seed, 'slice': name, 'class': label, 'support': sum(baseline[index]),
            **{metric + '_baseline': value for metric, value in before.items()},
            **{metric + '_candidate': value for metric, value in after.items()},
            'recall_delta': after['recall'] - before['recall']
            if before['recall'] is not None else None,
        })
    return rows


def _confusion_rows(seed, name, classes, baseline, candidate):
    return [
        {'seed': seed, 'slice': name, 'actual': actual, 'predicted': predicted,
         'baseline_count': baseline[i][j], 'candidate_count': candidate[i][j],
         'delta': candidate[i][j] - baseline[i][j], 'support': sum(baseline[i])}
        for i, actual in enumerate(classes) for j, predicted in enumerate(classes)
        if i != j and (baseline[i][j] or candidate[i][j])
    ]


def _paired_row(seed, baseline, candidate):
    paths = {'selection_score': ('selection_score',),
             'macro_f1': ('validation', 'macro_f1'),
             'balanced_accuracy': ('validation', 'balanced_accuracy'),
             'transition_macro_f1': ('transition_zone', 'macro_f1')}
    row = {'seed': seed}
    for metric, path in paths.items():
        values = []
        for run in (baseline, candidate):
            value = run
            for key in path:
                value = value[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise ValueError('Paired run metrics must be finite values in [0, 1]')
            values.append(value)
        row = {**row, metric + '_baseline': values[0],
               metric + '_candidate': values[1], metric + '_delta': values[1] - values[0]}
    return row


def _compare(baseline_runs, candidate_runs):
    baseline, candidate = _index_runs(baseline_runs), _index_runs(candidate_runs)
    if set(baseline) != set(candidate):
        raise ValueError('Baseline and candidate seed sets must match')
    classes_expected = None
    class_rows, confusion_rows, paired_rows = [], [], []
    for seed in sorted(baseline):
        classes, indices, before = _diagnostics(baseline[seed])
        other_classes, other_indices, after = _diagnostics(candidate[seed])
        if classes != other_classes or (classes_expected is not None and classes != classes_expected):
            raise ValueError('All runs must use the same ordered class universe')
        if indices != other_indices:
            raise ValueError('Paired validation indices must match in order')
        classes_expected = classes
        for name in SLICES:
            if list(map(sum, before[name])) != list(map(sum, after[name])):
                raise ValueError('Paired true class supports must match per slice')
            class_rows.extend(_class_rows(seed, name, classes, before[name], after[name]))
            confusion_rows.extend(_confusion_rows(seed, name, classes, before[name], after[name]))
        paired_rows.append(_paired_row(seed, baseline[seed], candidate[seed]))
    return {'class_rows': class_rows, 'confusion_rows': confusion_rows, 'paired_rows': paired_rows}


def compare_class_impact(baseline_runs, candidate_runs):
    """Compare paired runs without trusting classification-report rounding.

    Recall is undefined for zero true support; precision for no predictions;
    F1 for no true or predicted support. Those cases are represented as None.
    Input evidence is never changed. Scalar deltas use recorded run metrics.
    """
    try:
        return _compare(baseline_runs, candidate_runs)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Incomplete or malformed class-impact evidence') from error
