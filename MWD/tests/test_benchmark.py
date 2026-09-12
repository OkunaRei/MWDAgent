from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.benchmark import (
    compute_selection_score,
    candidate_key,
    choose_test_seed,
    feature_group_columns,
    select_candidate,
    select_model,
    split_train_validation,
    summarize_validation_runs,
    write_jsonl,
)


def test_split_train_validation_is_deterministic_and_stratified() -> None:
    frame = pd.DataFrame(
        {
            "feature": range(20),
            "Rock": ["A"] * 10 + ["B"] * 10,
            "transition_zone": [False, True] * 10,
        }
    )

    first_train, first_validation = split_train_validation(frame, validation_size=0.25, seed=11)
    second_train, second_validation = split_train_validation(frame, validation_size=0.25, seed=11)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_validation, second_validation)
    assert len(first_train) == 15
    assert len(first_validation) == 5
    assert set(first_train.index).isdisjoint(first_validation.index)
    assert first_train["Rock"].value_counts().to_dict() == {"A": 8, "B": 7}
    assert first_validation["Rock"].value_counts().to_dict() == {"A": 2, "B": 3}


def test_selection_score_prioritizes_transition_performance() -> None:
    score = compute_selection_score(
        balanced_accuracy=0.80,
        macro_f1=0.80,
        transition_macro_f1=0.60,
    )

    assert score == 0.74


def test_selection_score_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        compute_selection_score(
            balanced_accuracy=0.80,
            macro_f1=0.80,
            transition_macro_f1=0.60,
            balanced_accuracy_weight=0.2,
            macro_f1_weight=0.5,
            transition_macro_f1_weight=0.4,
        )


def test_validation_summary_reports_mean_and_std_without_test_data() -> None:
    runs = [
        {"model": "lightgbm", "seed": 1, "validation": {"macro_f1": 0.80}},
        {"model": "lightgbm", "seed": 2, "validation": {"macro_f1": 0.60}},
        {"model": "extratrees", "seed": 1, "validation": {"macro_f1": 0.70}},
    ]

    summary = summarize_validation_runs(runs)

    assert summary["lightgbm"]["macro_f1_mean"] == 0.70
    assert np.isclose(summary["lightgbm"]["macro_f1_std"], 0.1)
    assert "test" not in json.dumps(summary)


def test_write_jsonl_writes_one_record_per_line(tmp_path) -> None:
    path = tmp_path / "optimization_log.jsonl"
    records = [{"model": "lightgbm", "seed": 1}, {"model": "extratrees", "seed": 1}]

    write_jsonl(path, records)

    assert path.read_text(encoding="utf-8").splitlines() == [json.dumps(record) for record in records]


def test_model_selection_uses_validation_mean_and_test_seed_is_fixed() -> None:
    summary = {
        "lightgbm": {"selection_score_mean": 0.75},
        "extratrees": {"selection_score_mean": 0.74},
    }

    assert select_model(summary) == "lightgbm"
    assert choose_test_seed([11, 19, 42, 73, 101], reference_seed=42) == 42


def test_choose_test_seed_rejects_unlisted_reference_seed() -> None:
    with pytest.raises(ValueError, match="reference_seed"):
        choose_test_seed([11, 19], reference_seed=42)


def test_feature_groups_are_subsets_of_the_public_mwd_features() -> None:
    feature_names = [f"feature_{i}" for i in range(48)]
    feature_names[0] = "PenetrNormMean"
    feature_names[1] = "RotaPressNormMean"
    feature_names[2] = "FeedPressNormMean"
    feature_names[3] = "HammerPressNormMean"
    feature_names[4] = "WaterFlowNormMean"
    frame = pd.DataFrame({name: [1.0] for name in feature_names})
    frame["Rock"] = "A"
    frame["transition_zone"] = False
    frame["Tunnel"] = "T1"
    frame["PegStart"] = 0.0
    frame["PegEnd"] = 1.0
    frame["round_length"] = 1.0

    assert feature_group_columns(frame, "all_48") == feature_names
    assert feature_group_columns(frame, "water_flow") == ["WaterFlowNormMean"]


def test_select_candidate_uses_validation_average_score() -> None:
    summary = {
        "all_48__lightgbm": {"selection_score_mean": 0.70},
        "water_flow__extratrees": {"selection_score_mean": 0.72},
    }

    assert select_candidate(summary) == "water_flow__extratrees"


def test_candidate_key_is_stable_and_rejects_unknown_options() -> None:
    assert candidate_key("water_flow", "extratrees") == "water_flow__extratrees"
    with pytest.raises(ValueError, match="Unsupported feature group"):
        candidate_key("unknown", "extratrees")
