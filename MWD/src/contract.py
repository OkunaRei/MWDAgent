from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMNS = frozenset({"Rock", "transition_zone"})
LOCATION_COLUMNS = frozenset({"Tunnel", "PegStart", "PegEnd", "round_length"})
MWD_FEATURE_COUNT = 48


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load one public model-ready CSV without changing its values."""
    frame = pd.read_csv(path)
    if frame.columns[0] == "" or str(frame.columns[0]).startswith("Unnamed"):
        frame = frame.drop(columns=[frame.columns[0]])
    return frame.copy(deep=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = TARGET_COLUMNS | LOCATION_COLUMNS
    columns = [column for column in frame.columns if column not in excluded]
    if len(columns) != MWD_FEATURE_COUNT:
        raise ValueError(f"Expected {MWD_FEATURE_COUNT} MWD features, found {len(columns)}")
    if not frame[columns].apply(pd.api.types.is_numeric_dtype).all():
        raise TypeError("All MWD feature columns must be numeric")
    return columns
