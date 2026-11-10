# Physics-Informed Solar Energy Forecasting and Fault Detection

**Portfolio project | Applied Physics • Python • Machine Learning • Data Analytics**

## Overview
This project investigates two problems in photovoltaic (PV) energy systems:
1. Forecasting PV power one hour ahead from weather, operating conditions and recent power output.
2. Detecting simulated performance fault events from environmental and electrical-performance measurements.

The project combines a simplified physics-inspired PV reference equation with machine-learning models. It demonstrates data preparation, exploratory analysis, time-aware validation, regression, classification and model interpretation.

> **Data integrity statement:** Every row in the included dataset is synthetic. The data generator simulates weather, PV output and injected fault intervals. No measurements were obtained from a real solar farm, employer, laboratory or research institution. Results are educational demonstrations only.

## Why it is relevant to a Physics graduate
The project connects irradiance and temperature effects on photovoltaic output with computational modelling and data-driven monitoring. It offers a concrete example of applying physical-science reasoning, Python and machine learning to an energy problem.

## Research questions
- How do irradiance, module temperature, ambient temperature, wind and cloud conditions relate to simulated PV output?
- How well can a model forecast next-hour power compared with a simple mean baseline?
- Can a classifier identify simulated fault intervals during daylight?
- Which variables influence held-out classification F1, and what are the limitations of feature-importance analysis?

## Simplified physical reference
The simulation uses:

`P_expected = P_rated × (G / 1000) × [1 + γ(T_module − 25)]`

where rated capacity is 100 kW, irradiance `G` is measured in W/m² and the illustrative temperature coefficient `γ` is −0.004 per °C. This is deliberately simplified; it is not a bankable PV engineering model.

## Machine-learning tasks
### A. Next-hour power forecasting
- Target: actual power in the following hour.
- Predictors: current weather and operating variables, expected power, lagged output, rolling historical output and time features.
- Validation: chronological 80/20 split.
- Models: mean baseline and Random Forest regressor.
- Metrics: MAE, RMSE and R².

### B. Daylight fault classification
- Scope: records with irradiance ≥ 80 W/m².
- Target: simulated fault-event label.
- Models: majority-class baseline and class-weighted Random Forest.
- Metrics: accuracy, precision, recall, F1 and ROC-AUC; confusion matrix and permutation importance.
- Caution: the `performance_ratio` and actual output are derived from the same simulated mechanism used to inject faults. This can make classification easier than a real predictive-maintenance task. Run an ablation without `performance_ratio` and `actual_power_kw` to test robustness.

## Dataset
`data/synthetic_solar_pv_hourly.csv` contains 17,520 hourly observations from 2024–2025. Fields include timestamp, irradiance, ambient and module temperatures, wind speed, cloud index, expected and actual power, performance ratio, fault-event flag and illustrative fault type.

## Repository layout
```text
.
├── data/synthetic_solar_pv_hourly.csv
├── notebooks/project_walkthrough.ipynb
├── reports/
│   ├── forecast_metrics.csv
│   ├── fault_detection_metrics.csv
│   ├── fault_classification_report.txt
│   ├── fault_confusion_matrix.csv
│   ├── fault_permutation_importance.csv
│   ├── run_summary.json
│   ├── REPORT_GUIDE.md
│   └── figures/
├── src/run_analysis.py
├── tests/test_dataset.py
├── requirements.txt
└── README.md
```

## Run on Windows 11 (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/run_analysis.py
```
If activation is blocked, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/run_analysis.py
```
Open the notebook with:
```powershell
jupyter notebook
```
Then open `notebooks/project_walkthrough.ipynb`.

## Understanding the metrics
- **MAE:** average absolute forecast error in kW; lower is better.
- **RMSE:** penalises large forecast errors more heavily; lower is better.
- **R²:** compares the model with a mean-prediction reference; it can be negative.
- **Precision:** among predicted faults, the fraction that are faults.
- **Recall:** among actual faults, the fraction detected.
- **F1:** harmonic balance of precision and recall.
- **ROC-AUC:** ranking ability across thresholds; it does not select a safe alarm threshold.

## Limitations and research integrity
1. Synthetic data and labels are not field evidence.
2. The PV equation omits inverter efficiency, orientation, shading geometry, soiling dynamics, electrical losses and other engineering factors.
3. Fault labels are known because intervals were injected by the generator; real labels require verified maintenance records or expert review.
4. Derived features can leak information about injected faults. Define exactly when a prediction is intended to be made and exclude variables unavailable at that time.
5. Permutation importance is model- and dataset-dependent, not causal evidence.
6. No model here is approved for operational control or maintenance decisions.

## Recommended extensions
- Compare persistence, gradient boosting and physics-informed residual approaches.
- Run feature-ablation experiments, particularly excluding actual power and performance ratio from fault classification.
- Tune alarm thresholds based on the relative costs of missed faults and false alarms.
- Add prediction intervals and uncertainty calibration.
- Validate on real, permissioned PV plant data with a site/time-based external test set.

## Suggested CV wording
Only after personally running and understanding the work:

> Built a physics-informed Python workflow on simulated photovoltaic operating data to forecast next-hour power and classify injected fault events; implemented chronological validation, regression and classification metrics, confusion-matrix analysis and permutation-based feature interpretation.

Do not claim real solar-farm data, deployment, energy savings or validated real-world fault detection.
