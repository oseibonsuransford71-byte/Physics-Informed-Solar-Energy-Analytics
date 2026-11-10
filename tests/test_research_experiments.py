from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_research_experiments import (  # noqa: E402
    calculate_shortfall_percentage,
    chronological_three_way_split,
    predict_by_threshold,
    select_threshold,
)


def test_three_way_split_is_ordered_non_overlapping_and_complete():
    frame = pd.DataFrame({"value": range(10)})
    train, validation, test = chronological_three_way_split(
        frame, train_fraction=0.6, validation_fraction=0.2
    )
    assert train["value"].tolist() == list(range(6))
    assert validation["value"].tolist() == [6, 7]
    assert test["value"].tolist() == [8, 9]
    assert len(train) + len(validation) + len(test) == len(frame)


def test_shortfall_is_signed_and_zero_expected_power_is_missing():
    frame = pd.DataFrame({
        "expected_power_kw": [10.0, 10.0, 0.0],
        "actual_power_kw": [8.0, 12.0, 1.0],
    })
    result = calculate_shortfall_percentage(frame)
    assert result.iloc[0] == 20.0
    assert result.iloc[1] == -20.0
    assert pd.isna(result.iloc[2])


def test_threshold_prediction_uses_greater_than_or_equal():
    prediction = predict_by_threshold(pd.Series([10.0, 20.0, 21.0]), 20.0)
    assert prediction.tolist() == [0, 1, 1]


def test_threshold_search_uses_only_the_validation_frame():
    validation = pd.DataFrame({
        "shortfall_percentage": [2.0, 12.0, 22.0, 32.0],
        "fault_event": [0, 1, 1, 1],
    })
    selected, curve = select_threshold(validation, candidates=[5.0, 15.0, 25.0])
    assert selected in {5.0, 15.0, 25.0}
    assert set(curve["threshold_percent"]) == {5.0, 15.0, 25.0}
