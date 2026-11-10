"""Reproducible research experiments for the synthetic PV portfolio dataset.

This script preserves the original modelling pipeline and reports. It adds:
1) a daylight Random Forest feature-ablation experiment; and
2) a physics-based shortfall detector with validation-only threshold selection.

All results use the committed synthetic dataset. They are not evidence of
performance on a real photovoltaic plant.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_solar_pv_hourly.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DAYLIGHT_IRRADIANCE_MIN = 80.0
THRESHOLD_CANDIDATES = np.arange(5.0, 40.0001, 0.25)


def chronological_three_way_split(frame, train_fraction=0.64, validation_fraction=0.16):
    """Split an already time-sorted frame into non-overlapping chronological blocks."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    n = len(frame)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def calculate_shortfall_percentage(frame):
    """Signed percentage shortfall; negative values are deliberately retained."""
    expected = pd.to_numeric(frame["expected_power_kw"], errors="coerce")
    actual = pd.to_numeric(frame["actual_power_kw"], errors="coerce")
    valid = expected.gt(0)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid] = (
        (expected.loc[valid] - actual.loc[valid]) / expected.loc[valid] * 100.0
    )
    return result


def predict_by_threshold(shortfall_percentage, threshold):
    return (pd.Series(shortfall_percentage) >= float(threshold)).astype(int)


def metric_row(split_name, y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    auc = (
        roc_auc_score(y_true, y_score)
        if y_score is not None and np.unique(y_true).size == 2
        else None
    )
    return {
        "split": split_name,
        "records": int(len(y_true)),
        "actual_faults": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def select_threshold(validation_frame, candidates=THRESHOLD_CANDIDATES):
    """Select threshold by validation F1 only. Test rows are never accepted here."""
    y = validation_frame["fault_event"].astype(int)
    rows = []
    for threshold in candidates:
        pred = predict_by_threshold(
            validation_frame["shortfall_percentage"], threshold
        )
        rows.append({
            "threshold_percent": float(threshold),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
    curve = pd.DataFrame(rows)
    # Deterministic tie-break: best F1, then precision, then lower threshold.
    best = curve.sort_values(
        ["f1", "precision", "threshold_percent"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return float(best["threshold_percent"]), curve


def run_ablation(data):
    """Compare baseline and RF on daylight data without actual output or PR."""
    daylight = data.loc[
        data["irradiance_w_m2"] >= DAYLIGHT_IRRADIANCE_MIN
    ].sort_values("timestamp").reset_index(drop=True)
    cut = int(len(daylight) * 0.80)
    train, test = daylight.iloc[:cut].copy(), daylight.iloc[cut:].copy()

    features = [
        "irradiance_w_m2", "ambient_temperature_c",
        "module_temperature_c", "wind_speed_m_s",
        "cloud_index", "expected_power_kw",
    ]
    X_train, X_test = train[features], test[features]
    y_train, y_test = train["fault_event"].astype(int), test["fault_event"].astype(int)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=180, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)

    base_pred = baseline.predict(X_test)
    pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    rows = []
    for name, prediction, score in [
        ("Majority baseline", base_pred, None),
        ("Random Forest (environmental ablation)", pred, probability),
    ]:
        row = metric_row("test", y_test, prediction, score)
        row["model"] = name
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        REPORTS / "fault_detection_metrics_ablation.csv", index=False
    )

    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_no_fault", "actual_fault"],
        columns=["predicted_no_fault", "predicted_fault"],
    ).to_csv(
        REPORTS / "fault_confusion_matrix_ablation.csv",
        index_label="actual_condition",
    )
    (REPORTS / "fault_classification_report_ablation.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8"
    )

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=42,
        scoring="f1", n_jobs=-1,
    )
    pd.DataFrame({
        "feature": features,
        "mean_f1_decrease": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("mean_f1_decrease", ascending=False).to_csv(
        REPORTS / "fault_permutation_importance_ablation.csv", index=False
    )
    return pd.DataFrame(rows), matrix


def run_physics_threshold(data):
    """Select shortfall threshold on validation, then evaluate frozen threshold on test."""
    daylight = data.loc[
        (data["irradiance_w_m2"] >= DAYLIGHT_IRRADIANCE_MIN)
        & (data["expected_power_kw"] > 0)
    ].copy()
    daylight = daylight.sort_values("timestamp").reset_index(drop=True)
    daylight["shortfall_percentage"] = calculate_shortfall_percentage(daylight)
    daylight = daylight.dropna(subset=["shortfall_percentage", "fault_event"])
    daylight["fault_event"] = daylight["fault_event"].astype(int)

    train, validation, test = chronological_three_way_split(daylight)
    threshold, curve = select_threshold(validation)
    val_pred = predict_by_threshold(validation["shortfall_percentage"], threshold)
    test_pred = predict_by_threshold(test["shortfall_percentage"], threshold)

    val_metrics = metric_row("validation", validation["fault_event"], val_pred)
    test_metrics = metric_row("test", test["fault_event"], test_pred)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        REPORTS / "physics_threshold_metrics.csv", index=False
    )
    curve.to_csv(REPORTS / "physics_threshold_validation_curve.csv", index=False)

    matrix = confusion_matrix(test["fault_event"], test_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_no_fault", "actual_fault"],
        columns=["predicted_no_fault", "predicted_fault"],
    ).to_csv(
        REPORTS / "physics_threshold_confusion_matrix.csv",
        index_label="actual_condition",
    )

    split_rows = []
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        fault_rows = part.loc[part["fault_event"] == 1]
        split_rows.append({
            "split": name,
            "records": int(len(part)),
            "normal_records": int((part["fault_event"] == 0).sum()),
            "fault_records": int(len(fault_rows)),
            "fault_prevalence_percent": float(part["fault_event"].mean() * 100),
            "mean_fault_shortfall_percent": (
                float(fault_rows["shortfall_percentage"].mean())
                if len(fault_rows) else np.nan
            ),
        })
    pd.DataFrame(split_rows).to_csv(
        REPORTS / "fault_distribution_by_split.csv", index=False
    )

    test_faults = test.loc[test["fault_event"] == 1].copy()
    test_faults["detected"] = test_pred.loc[test_faults.index].to_numpy()
    type_rows = []
    for fault_type, group in test_faults.groupby("fault_type", dropna=False):
        type_rows.append({
            "fault_type": str(fault_type),
            "test_fault_records": int(len(group)),
            "detected": int(group["detected"].sum()),
            "missed": int((1 - group["detected"]).sum()),
            "recall": float(group["detected"].mean()),
            "mean_shortfall_percent": float(group["shortfall_percentage"].mean()),
        })
    pd.DataFrame(type_rows).to_csv(
        REPORTS / "fault_type_test_performance.csv", index=False
    )

    all_splits = pd.concat([
        part.assign(split=name)
        for name, part in [
            ("train", train), ("validation", validation), ("test", test)
        ]
    ])
    type_distribution = (
        all_splits.loc[all_splits["fault_event"] == 1]
        .groupby(["split", "fault_type"]).size().unstack(fill_value=0)
    )
    type_distribution.to_csv(REPORTS / "fault_type_distribution_by_split.csv")

    summary = {
        "method": "physics-based percentage shortfall threshold",
        "daylight_irradiance_min_w_m2": DAYLIGHT_IRRADIANCE_MIN,
        "threshold_selection_metric": "validation F1",
        "threshold_grid_percent": {
            "minimum": float(THRESHOLD_CANDIDATES.min()),
            "maximum": float(THRESHOLD_CANDIDATES.max()),
            "step": 0.25,
        },
        "selected_threshold_percent": threshold,
        "train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "test_records": int(len(test)),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix_order": ["actual_no_fault", "actual_fault"],
        "test_confusion_matrix": matrix.tolist(),
        "synthetic_data": True,
        "interpretation": (
            "Retrospective detection using measured actual power; not early-warning "
            "prediction before a power shortfall is observed."
        ),
    }
    (REPORTS / "physics_threshold_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest")
    ax.set_title(f"Physics threshold test confusion matrix ({threshold:.2f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["No fault", "Fault"])
    ax.set_yticks([0, 1], ["No fault", "Fault"])
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "physics_threshold_confusion_matrix.png", dpi=160)
    plt.close(fig)

    if not type_distribution.empty:
        ax = type_distribution.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Injected fault-type counts by chronological split")
        ax.set_xlabel("Chronological split")
        ax.set_ylabel("Fault records")
        ax.legend(title="Fault type")
        plt.tight_layout()
        plt.savefig(FIGURES / "fault_type_distribution_by_split.png", dpi=160)
        plt.close()

    return threshold, val_metrics, test_metrics, matrix


def run():
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    required = {
        "timestamp", "irradiance_w_m2", "expected_power_kw",
        "actual_power_kw", "fault_event", "fault_type",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data["fault_event"].isna().any():
        raise ValueError("fault_event contains missing labels")

    ablation_metrics, ablation_matrix = run_ablation(data)
    threshold, validation_metrics, test_metrics, threshold_matrix = (
        run_physics_threshold(data)
    )
    print("\\nABLATION METRICS\\n", ablation_metrics.to_string(index=False))
    print(f"\\nValidation-selected physics threshold: {threshold:.2f}%")
    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)
    print("Ablation confusion matrix:\\n", ablation_matrix)
    print("Physics threshold test confusion matrix:\\n", threshold_matrix)
    print("\\nResearch experiment outputs saved under reports/.")


if __name__ == "__main__":
    run()
