from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

from src.config import BOOKING_TIME_FEATURES
from src.data.load import load_raw_data
from src.pipelines import run_training_pipeline
from src.utils.validate_data import clean_raw


def _sanitize_record(record: Mapping[object, Any]) -> dict[str, Any]:
    return {str(key): (None if pd.isna(value) else value) for key, value in record.items()}


@pytest.fixture(scope="session")
def sample_record() -> dict[str, Any]:
    df = load_raw_data()
    df, _ = clean_raw(df)
    record = df.iloc[0][BOOKING_TIME_FEATURES].to_dict()
    return _sanitize_record(record)


@pytest.fixture(scope="session")
def trained_artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("trained-artifacts")
    outputs = run_training_pipeline(
        artifacts_dir=root / "artifacts",
        reports_dir=root / "reports",
        max_rows=5000,
    )
    return outputs.artifacts_dir
