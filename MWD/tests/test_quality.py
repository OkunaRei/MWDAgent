import numpy as np
import pytest

from src.quality import fit_quality_bounds, quality_scores, gating_ablation


def test_train_bounds_remain_fixed_with_outliers_and_constant_columns():
    train = np.array([[0., 5.], [1., 5.], [2., 5.], [3., 5.], [4., 5.]])
    original = train.copy()
    lower, upper = fit_quality_bounds(train)
    np.testing.assert_array_equal(lower, [-2., 5.])
    np.testing.assert_array_equal(upper, [6., 5.])
    validation = np.array([[-2., 5.], [6., 5.], [100., 5.], [2., 6.], [np.nan, np.inf]])
    before = validation.copy()
    np.testing.assert_array_equal(quality_scores(validation, lower, upper), [1, 1, .5, .5, 0])
    np.testing.assert_array_equal(train, original)
    np.testing.assert_array_equal(validation, before)
    np.testing.assert_array_equal(lower, [-2., 5.])
    np.testing.assert_array_equal(upper, [6., 5.])
    assert quality_scores(np.empty((0, 2)), lower, upper).shape == (0,)


@pytest.mark.parametrize('features', [[], [[np.nan]], [[np.inf]], np.empty((0, 2)), np.empty((2, 0)), [1, 2]])
def test_invalid_training_features(features):
    with pytest.raises(ValueError):
        fit_quality_bounds(features)


@pytest.mark.parametrize('features,lower,upper', [
    ([[1, 2]], [0], [2]), ([[1]], [2], [0]), ([[1]], [np.nan], [2]),
    ([[1]], [0], [np.inf]), ([[1]], [[0]], [[2]]), ([1], [0], [2]),
    (np.empty((2, 0)), [], []),
])
def test_invalid_score_bounds(features, lower, upper):
    with pytest.raises(ValueError):
        quality_scores(features, lower, upper)


def test_ablation_intersection_threshold_equality_and_fixed_class_universe():
    labels = np.array([0, 1, 2, 1])
    probabilities = np.array([[.8, .1, .1], [.85, .1, .05], [.2, .3, .5], [.2, .5, .3]])
    quality = np.array([.9, .5, 1, .2])
    result = gating_ablation(labels, probabilities, quality)
    assert result['no_gate']['accepted'] == 4
    assert result['quality_only']['accepted'] == 2
    assert result['confidence_only']['accepted'] == 2
    assert result['dual']['accepted'] == 1
    assert result['dual']['reviewed'] == 3
    assert result['dual']['coverage'] == .25
    assert result['dual']['risk'] == 0
    assert result['dual']['macro_f1'] == pytest.approx(1 / 3)
    assert result['dual']['class_support'] == [1, 2, 1]
    assert result['dual']['accepted_class_support'] == [1, 0, 0]
    assert result['no_gate']['risk'] == .25
    np.testing.assert_array_equal(quality, [.9, .5, 1, .2])


def test_no_accepted_samples_have_null_metrics():
    result = gating_ablation(np.array([0]), np.array([[.6, .4]]), np.array([.1]))
    assert result['dual']['accepted'] == 0
    assert result['dual']['coverage'] == 0
    assert result['dual']['risk'] is None
    assert result['dual']['accuracy'] is None
    assert result['dual']['macro_f1'] is None
    assert result['dual']['accepted_class_support'] == [0, 0]


def test_all_modes_accept_without_mutating_input_arrays():
    labels = np.array([0, 1])
    probabilities = np.array([[.8, .2], [.1, .9]])
    quality = np.array([.9, 1])
    originals = [value.copy() for value in (labels, probabilities, quality)]
    result = gating_ablation(labels, probabilities, quality)
    for metrics in result.values():
        assert metrics['accepted'] == 2
        assert metrics['reviewed'] == 0
        assert metrics['coverage'] == 1
        assert metrics['macro_f1'] == 1
        assert metrics['risk'] == 0
    for value, original in zip((labels, probabilities, quality), originals):
        np.testing.assert_array_equal(value, original)


def test_extreme_training_range_fails_before_returning_infinite_bounds():
    with pytest.raises(ValueError, match='overflows'):
        fit_quality_bounds(np.array([[-1e308], [-1e308], [1e308], [1e308]]))


def test_empty_slice_has_undefined_coverage_and_risk():
    result = gating_ablation(np.array([], dtype=int), np.empty((0, 3)), np.array([]))
    assert result['no_gate']['coverage'] is None
    assert result['no_gate']['risk'] is None
    assert result['no_gate']['support'] == 0


@pytest.mark.parametrize('labels,probabilities,quality', [
    ([0], [[.2, .2]], [1]), ([0], [[1.1, -.1]], [1]),
    ([0], [[np.nan, .5]], [1]), ([2], [[.5, .5]], [1]),
    ([.5], [[.5, .5]], [1]), ([0], [[.5, .5]], [np.nan]),
    ([0], [[.5, .5]], [1.1]), ([0], [[.5, .5]], []),
    ([[0]], [[.5, .5]], [1]), ([0], [[1]], [1]),
])
def test_invalid_ablation_inputs(labels, probabilities, quality):
    with pytest.raises(ValueError):
        gating_ablation(labels, probabilities, quality)


@pytest.mark.parametrize('threshold', [-.1, 1.1, np.nan, np.inf])
def test_invalid_ablation_thresholds(threshold):
    with pytest.raises(ValueError):
        gating_ablation([0], [[.5, .5]], [1], quality_threshold=threshold)
    with pytest.raises(ValueError):
        gating_ablation([0], [[.5, .5]], [1], confidence_threshold=threshold)
