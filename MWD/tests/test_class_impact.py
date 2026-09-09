from copy import deepcopy

import pytest

from src.class_impact import compare_class_impact


def run(seed=42, matrix=None):
    matrix = matrix or [[4, 1, 0], [2, 3, 0], [0, 0, 0]]
    return {'seed': seed, 'selection_score': .6,
            'validation': {'macro_f1': .6, 'balanced_accuracy': .7},
            'transition_zone': {'macro_f1': .5},
            'diagnostics': {'classes': ['A', 'B', 'absent'],
                            'validation_indices': list(range(10)),
                            'overall': {'confusion_matrix': matrix},
                            'ordinary': {'support': 10, 'confusion_matrix': matrix},
                            'transition_zone': {'support': 0, 'confusion_matrix': [[0]*3 for _ in range(3)]}}}


def test_pairing_metrics_and_confusions_from_counts_without_mutation():
    baseline = run()
    candidate = run(matrix=[[5, 0, 0], [5, 0, 0], [0, 0, 0]])
    candidate['validation'] = {'macro_f1': .65, 'balanced_accuracy': .5}
    before = deepcopy([baseline, candidate])
    result = compare_class_impact([baseline], [candidate])
    rows = result['class_rows']
    a, b, absent = rows[:3]
    assert a['recall_baseline'] == .8
    assert a['recall_candidate'] == 1
    assert a['recall_delta'] == pytest.approx(.2)
    assert a['precision_baseline'] == pytest.approx(4/6)
    assert a['f1_baseline'] == pytest.approx(8/11)
    assert b['recall_candidate'] == 0
    assert b['precision_candidate'] is None
    assert b['f1_candidate'] == 0
    assert absent['support'] == 0
    assert absent['recall_candidate'] is None
    assert absent['f1_candidate'] is None
    assert all(row['recall_baseline'] is None for row in rows[-3:])
    assert result['paired_rows'][0]['balanced_accuracy_delta'] == pytest.approx(-.2)
    assert result['paired_rows'][0]['macro_f1_delta'] == pytest.approx(.05)
    assert result['confusion_rows'][0] == {
        'seed': 42, 'slice': 'overall', 'actual': 'A', 'predicted': 'B',
        'baseline_count': 1, 'candidate_count': 0, 'delta': -1, 'support': 5}
    assert [baseline, candidate] == before


def test_pairs_by_seed_and_keeps_fixed_class_universe():
    result = compare_class_impact([run(2), run(1)], [run(1), run(2)])
    assert [row['seed'] for row in result['paired_rows']] == [1, 2]
    assert len(result['class_rows']) == 18


@pytest.mark.parametrize('change', [
    lambda r: r.update(seed=99),
    lambda r: r['diagnostics'].update(classes=['B', 'A', 'absent']),
    lambda r: r['diagnostics'].update(classes=['A', 'A', 'absent']),
    lambda r: r['diagnostics'].update(validation_indices=list(reversed(range(10)))),
    lambda r: r['diagnostics']['ordinary'].update(support=9),
    lambda r: r['diagnostics']['overall'].update(confusion_matrix=[[1]]),
    lambda r: r['diagnostics']['overall'].update(confusion_matrix=[[4, -1, 0], [2, 3, 0], [0, 0, 0]]),
    lambda r: r['diagnostics']['overall'].update(confusion_matrix=[[4, 1.0, 0], [2, 3, 0], [0, 0, 0]]),
    lambda r: r['diagnostics']['overall'].update(confusion_matrix=[[True, 1, 0], [2, 3, 0], [0, 0, 0]]),
    lambda r: r['diagnostics']['overall'].update(confusion_matrix=[[3, 1, 0], [3, 3, 0], [0, 0, 0]]),
    lambda r: r['validation'].update(macro_f1=float('nan')),
])
def test_rejects_misaligned_or_invalid_evidence(change):
    candidate = run()
    change(candidate)
    with pytest.raises(ValueError):
        compare_class_impact([run()], [candidate])


def test_rejects_duplicate_seeds_empty_inputs_and_bad_shape():
    for baseline, candidate in [([run(), run()], [run()]), ([], []), ([{}], [run()])]:
        with pytest.raises(ValueError):
            compare_class_impact(baseline, candidate)


def test_class_tradeoff_can_raise_macro_f1_while_lowering_balanced_accuracy():
    baseline = run(matrix=[[3, 7, 0], [19, 1, 0], [0, 0, 0]])
    candidate = run(matrix=[[1, 9, 0], [16, 4, 0], [0, 0, 0]])
    for item in (baseline, candidate):
        item['diagnostics']['validation_indices'] = list(range(30))
        item['diagnostics']['ordinary']['support'] = 30
    result = compare_class_impact([baseline], [candidate])
    a, b = result['class_rows'][:2]
    assert a['recall_delta'] == pytest.approx(-.2)
    assert b['recall_delta'] == pytest.approx(.15)
    assert (a['recall_delta'] + b['recall_delta']) / 2 < 0
    assert (a['f1_candidate'] + b['f1_candidate']) > (a['f1_baseline'] + b['f1_baseline'])


@pytest.mark.parametrize('change', [
    lambda r: r['diagnostics'].update(validation_indices=[0]*10),
    lambda r: r['diagnostics'].update(validation_indices=list(range(9))),
    lambda r: r['diagnostics'].update(classes=[]),
    lambda r: r['diagnostics']['ordinary'].update(confusion_matrix=[[5, 0, 0], [2, 3, 0], [0, 0, 0]]),
    lambda r: r.update(seed=True),
    lambda r: r.update(selection_score=1.1),
])
def test_rejects_inconsistent_partition_indices_and_numeric_bounds(change):
    candidate = run()
    change(candidate)
    with pytest.raises(ValueError):
        compare_class_impact([run()], [candidate])


def test_rejects_class_order_drift_across_seeds():
    other = run(99)
    other['diagnostics']['classes'] = ['B', 'A', 'absent']
    with pytest.raises(ValueError, match='ordered class universe'):
        compare_class_impact([run(), other], [run(), deepcopy(other)])
