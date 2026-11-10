# How to read the generated reports

- `forecast_metrics.csv`: mean baseline versus Random Forest for one-hour-ahead power forecasting. Compare MAE, RMSE and R².
- `fault_detection_metrics.csv`: majority baseline versus Random Forest on daylight-only fault classification.
- `fault_classification_report.txt`: precision, recall, F1 and support for each class.
- `fault_confusion_matrix.csv`: true/false predictions by actual and predicted class.
- `fault_permutation_importance.csv`: change in held-out F1 after shuffling each feature; this is not causal evidence.
- `run_summary.json`: row counts, date range, fault prevalence and chronological split sizes.
- `figures/forecast_sample.png`: observed versus predicted power in a held-out sample.
- `figures/fault_confusion_matrix.png`: visual confusion matrix.
- `figures/fault_permutation_importance.png`: permutation-importance chart.

All outputs are based on simulated data. The injected fault labels and derived performance ratio make the classification task a controlled teaching example, not a validated real-world fault detector.
