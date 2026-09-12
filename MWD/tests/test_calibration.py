import numpy as np
import pandas as pd
import pytest

from src.calibration import fit_temperature, temperature_scale, split_calibration, probability_metrics


def test_temperature_improves_overconfident_probabilities_without_changing_labels():
    probabilities = np.array([[.99, .01]] * 8 + [[.01, .99]] * 2)
    labels = np.zeros(10, dtype=int)
    temperature = fit_temperature(labels, probabilities)
    calibrated = temperature_scale(probabilities, temperature)
    assert temperature > 1
    assert probability_metrics(labels, calibrated)["nll"] < probability_metrics(labels, probabilities)["nll"]
    np.testing.assert_array_equal(calibrated.argmax(1), probabilities.argmax(1))
    np.testing.assert_allclose(temperature_scale(probabilities, 1), probabilities)


@pytest.mark.parametrize("probabilities", [[[1.1, -.1]], [[.2, .2]], [[np.nan, .1]], [], [[1]]])
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        temperature_scale(probabilities, 1)


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        temperature_scale([[.4, .6]], temperature)


def test_split_is_disjoint_complete_reproducible_and_preserves_source():
    frame = pd.DataFrame({"Rock": ["a", "b"] * 100})
    original = frame.copy(deep=True)
    splits = split_calibration(frame, seed=42)
    assert [len(part) for part in splits] == [120, 40, 40]
    indices = [set(part.index) for part in splits]
    assert not any(indices[i] & indices[j] for i in range(3) for j in range(i))
    assert set.union(*indices) == set(frame.index)
    for first, second in zip(splits, split_calibration(frame, seed=42)):
        pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, original)


def test_empty_selection_has_undefined_risk():
    metrics = probability_metrics(np.array([0, 1]), np.array([[.6, .4], [.4, .6]]))
    assert metrics["confidence_sweep"][-1]["risk"] is None
    assert metrics["confidence_sweep"][0]["coverage"] == 1


def test_invalid_labels():
    with pytest.raises(ValueError):
        fit_temperature(np.array([2]), np.array([[.5, .5]]))


def test_summary_missing_slices_and_single_run_are_json_safe():
    import json
    from run_calibration import summarize_rows

    summary = summarize_rows([{"slice": "overall", "stage": "before",
                               "accuracy": .8, "ece": .1, "nll": .5}])
    assert summary["overall"]["before"]["ece"] == {"mean": .1, "std": None, "runs": 1}
    assert summary["ordinary"]["before"]["ece"] == {"mean": None, "std": None, "runs": 0}
    json.dumps(summary, allow_nan=False)


def test_cli_reads_only_public_training_file(monkeypatch, tmp_path):
    import sys
    import run_calibration as cli

    source = tmp_path / "mwd_rocktype_blastholes_model_ready_train.csv"
    source.write_text("Rock\na\n")
    reads = []
    monkeypatch.setattr(sys, "argv", ["run_calibration.py", "--data-dir", str(tmp_path),
                                      "--output-dir", str(tmp_path / "out")])
    monkeypatch.setattr(cli, "load_dataset", lambda path: reads.append(path))
    monkeypatch.setattr(cli, "run_calibration_once", lambda frame, seed: {
        "seed": seed, "temperature": 1., "validation": {}})
    cli.main()
    assert reads == [source]
    assert (tmp_path / "out" / "calibration.json").exists()


def test_quality_cli_uses_separate_report_directory(monkeypatch, tmp_path):
    import sys
    import run_calibration as cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'mwd_rocktype_blastholes_model_ready_train.csv').write_text('Rock\na\n')
    monkeypatch.setattr(sys, 'argv', ['run_calibration.py', '--data-dir', str(tmp_path), '--quality-ablation'])

    def run(frame, *, seed, include_quality):
        assert include_quality is True
        return {'seed': seed, 'temperature': 1., 'validation': {}, 'quality_ablation': {
            'slices': {'overall': {'no_gate': {'accepted': 1, 'coverage': 1.}}}}}

    monkeypatch.setattr(cli, 'run_calibration_once', run)
    cli.main()
    assert (tmp_path / 'reports/quality_ablation/gating_summary.csv').exists()
    assert not (tmp_path / 'reports/calibration').exists()


@pytest.mark.parametrize('include_quality', [False, True])
def test_run_keeps_sampling_model_fit_and_temperature_labels_isolated(monkeypatch, include_quality):
    import src.calibration as module

    frame = pd.DataFrame({"Rock": ["a", "b"] * 100,
                          "transition_zone": [False, True] * 100,
                          "feature": np.arange(200)})
    train, calibration, validation = split_calibration(frame, seed=42)
    calls = []

    def sample(features, labels, *, random_state):
        assert set(features.index) == set(train.index)
        assert labels.index.equals(features.index)
        calls.append("sample")
        return features, labels

    class Model:
        def fit(self, features, labels):
            assert set(features.index) == set(train.index)
            calls.append("fit")

        def predict_proba(self, features):
            calls.append(set(features.index))
            return np.tile([.6, .4], (len(features), 1))

        def get_params(self):
            return {}

    original_fit = module.fit_temperature

    def fit(labels, probabilities):
        np.testing.assert_array_equal(labels, (calibration["Rock"] == "b").astype(int))
        assert len(probabilities) == len(calibration)
        calls.append("temperature")
        return original_fit(labels, probabilities)

    monkeypatch.setattr(module, "feature_columns", lambda frame: ["feature"])
    monkeypatch.setattr(module, "rebalance_training_set", sample)
    monkeypatch.setattr(module, "_make_model", lambda name, seed: Model())
    monkeypatch.setattr(module, "fit_temperature", fit)
    result = module.run_calibration_once(frame, seed=42, include_quality=include_quality)
    assert calls == ["sample", "fit", set(calibration.index), "temperature", set(validation.index)]
    assert result["splits"]["train"]["indices"] == train.index.tolist()
    assert result["validation"]["overall"]["before"]["support"] == len(validation)
    if include_quality:
        records = result['quality_ablation']['predictions']
        assert len(records) == len(validation)
        assert [record['row_index'] for record in records] == validation.index.tolist()
        q1, q3 = np.quantile(train[['feature']].to_numpy(), [.25, .75], axis=0)
        np.testing.assert_allclose(result['quality_ablation']['bounds']['lower'], q1 - 1.5 * (q3 - q1))
        assert result['quality_ablation']['slices']['overall']['no_gate']['accepted'] == len(validation)
