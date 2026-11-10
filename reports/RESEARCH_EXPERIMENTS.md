# Solar fault-detection research experiments

## Purpose and provenance

This document describes two additional experiments implemented in
`src/run_research_experiments.py`:

1. a daylight Random Forest feature ablation; and
2. a physics-based percentage-shortfall threshold detector.

The committed dataset contains 17,520 hourly synthetic observations from
2024–2025. It is not real PV plant data. Fault intervals and fault types were
injected in the simulation. A data-generation script is not currently present
in the public repository, so the exact generation process cannot yet be
regenerated from source. The committed CSV is the input for the experiments.

The physical reference equation is simplified and illustrative. This is not
an engineering-grade PV model or a Physics-Informed Neural Network (PINN).

## Experiment A — feature ablation

**Question:** How much does the classifier rely on variables that may directly
encode the synthetic fault mechanism?

The script filters daylight records (irradiance >= 80 W/m²), sorts by time,
and uses a chronological 80/20 split. It fits a most-frequent baseline and a
class-weighted Random Forest (180 trees, minimum leaf size 2, random seed 42).
The ablation excludes `actual_power_kw` and `performance_ratio`, retaining
irradiance, ambient/module temperature, wind, cloud index and expected power.
It writes metrics, a confusion matrix, a classification report and
permutation importance to separate `_ablation` files.

### Earlier author-run result (requires reproduction)

A local run previously supplied by the author reported environmental-only
Random Forest accuracy 0.9680, precision 0.1250, recall 0.0222, F1 0.0377
and ROC-AUC 0.5010. The reported confusion matrix was TN=1543, FP=7,
FN=44, TP=1 (45 positive test records).

These are historical console results supplied by the project author, not
independently executed during preparation of this update. Run the script on
the committed dataset and compare the newly generated files before quoting
these numbers in an application or research statement.

The large decline from the original classifier is evidence that the original
score depended strongly on output-derived variables. It does not prove that
environmental predictors can never support fault detection; it shows that
this particular restricted model/dataset combination contains little
discriminative signal.

## Experiment B — physics-based threshold

For daylight records with positive expected power:

`shortfall_percent = 100 * (expected_power_kw - actual_power_kw) / expected_power_kw`

Negative values are retained. An alarm is raised when the signed shortfall is
greater than or equal to the selected threshold.

The data are sorted chronologically and split 64%/16%/20% into development,
validation and final test blocks. The threshold is selected by validation F1
over 5%–40% in 0.25 percentage-point steps. The selected threshold is frozen
before evaluation on the later test period. The test period must not be used
for threshold selection.

### Earlier author-run result (requires reproduction)

The author's earlier local analysis reported 5,104 development records,
1,276 validation records and 1,595 test records. It selected a 21% threshold
using validation data. Reported validation F1 was 0.70. On the untouched test
period, reported TN=1544, FP=6, FN=36, TP=9, precision=0.60, recall=0.20
and F1=0.30.

These figures are historical author-supplied console results. The script
regenerates the threshold, metrics and confusion matrix from the public CSV.
Compare its output with the historical figures before treating them as final.

**Important:** because the detector uses actual measured power, it is a
retrospective diagnostic detector. It is not early-warning prediction before
the output shortfall is observed.

## Fault-type distribution shift

The author's earlier split-level analysis reported:

| Fault type | Development | Validation | Test |
|---|---:|---:|---:|
| Inverter derating | 41 | 1 | 1 |
| Soiling | 27 | 5 | 37 |
| Partial shading | 15 | 15 | 7 |

The test period is dominated by soiling events, unlike the development
period. This is a substantial temporal change in fault-type composition.
The earlier 21% threshold test results by type were:

| Fault type | Test cases | Detected | Missed | Recall |
|---|---:|---:|---:|---:|
| Inverter derating | 1 | 0 | 1 | 0% |
| Soiling | 37 | 5 | 32 | 13.51% |
| Partial shading | 7 | 4 | 3 | 57.14% |

The inverter test support is only one record; its recall is not a stable
estimate. The script regenerates split prevalence, type counts and
type-specific test results.

## Generated outputs

- `fault_detection_metrics_ablation.csv`
- `fault_confusion_matrix_ablation.csv`
- `fault_classification_report_ablation.txt`
- `fault_permutation_importance_ablation.csv`
- `physics_threshold_metrics.csv`
- `physics_threshold_validation_curve.csv`
- `physics_threshold_confusion_matrix.csv`
- `physics_threshold_summary.json`
- `fault_distribution_by_split.csv`
- `fault_type_test_performance.csv`
- `fault_type_distribution_by_split.csv`
- `figures/physics_threshold_confusion_matrix.png`
- `figures/fault_type_distribution_by_split.png`

## Reproduction on Windows PowerShell

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python src/run_research_experiments.py
python -m pytest -q
```

The script writes separate outputs and does not overwrite the original
forecasting or classification reports.

## Interpretation and next research steps

1. Treat the physics threshold as a transparent retrospective diagnostic
   baseline, not an early-warning model.
2. Treat the ablation as a shortcut/leakage diagnostic, not proof of
   generalisation to real PV plants.
3. Investigate event-level and site-level evaluation when multiple independent
   plants or fault episodes are available.
4. Study fault-type robustness under controlled temporal distribution shifts.
5. Obtain permissioned real PV data and independently verified maintenance
   labels before making field-performance claims.
6. Publish the dataset generator, random seeds and fault-injection assumptions
   before claiming full end-to-end data regeneration.

## Safe academic summary

The project builds a reproducible workflow on synthetic photovoltaic data,
combining a simplified physics-based expected-power reference, next-hour
forecasting, a Random Forest fault-classification baseline, feature ablation
and a validation-selected shortfall-threshold detector. The experiments
examine information leakage and temporal fault-type distribution shift.
No real-plant performance or deployment claim is made.
