from pathlib import Path

import numpy as np

from src.contract import MWD_FEATURE_COUNT, feature_columns, load_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mwd_rocktype_10358374"


def test_model_ready_train_matches_public_data_contract() -> None:
    dataset = load_dataset(DATA_DIR / "mwd_rocktype_blastholes_model_ready_train.csv")

    assert len(dataset) == 3671
    assert dataset["Rock"].nunique() == 10
    assert len(feature_columns(dataset)) == MWD_FEATURE_COUNT
    assert dataset[feature_columns(dataset)].select_dtypes(include=[np.number]).shape[1] == MWD_FEATURE_COUNT


def test_feature_columns_exclude_targets_and_location_metadata() -> None:
    dataset = load_dataset(DATA_DIR / "mwd_rocktype_blastholes_model_ready_train.csv")
    features = feature_columns(dataset)

    assert "Rock" not in features
    assert "transition_zone" not in features
    assert "Tunnel" not in features
    assert "PegStart" not in features
    assert "PegEnd" not in features
    assert "round_length" not in features
