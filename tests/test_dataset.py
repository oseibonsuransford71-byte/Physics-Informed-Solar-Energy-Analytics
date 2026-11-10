from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_schema_and_size():
    df = pd.read_csv(ROOT / "data" / "synthetic_solar_pv_hourly.csv")
    required = {"timestamp","irradiance_w_m2","actual_power_kw","fault_event","fault_type"}
    assert required.issubset(df.columns)
    assert len(df) == 17520

def test_values_and_labels():
    df = pd.read_csv(ROOT / "data" / "synthetic_solar_pv_hourly.csv")
    assert (df["actual_power_kw"] >= 0).all()
    assert set(df["fault_event"].unique()).issubset({0,1})
    assert df["fault_event"].sum() > 0
