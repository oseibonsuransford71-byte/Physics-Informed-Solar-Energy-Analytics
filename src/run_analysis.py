from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA = ROOT / "data" / "synthetic_solar_pv_hourly.csv"

def chronological_split(frame, fraction=0.8):
    cut = int(len(frame) * fraction)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()

def run():
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")

    # Next-hour forecasting. Features contain only current/past information.
    f = data.copy()
    f["target_next_hour_kw"] = f["actual_power_kw"].shift(-1)
    f["lag_1h_kw"] = f["actual_power_kw"]
    f["lag_24h_kw"] = f["actual_power_kw"].shift(24)
    f["rolling_24h_mean_kw"] = f["actual_power_kw"].rolling(24).mean()
    f["hour"] = f["timestamp"].dt.hour
    f["day_of_year"] = f["timestamp"].dt.dayofyear
    ff = ["irradiance_w_m2","ambient_temperature_c","module_temperature_c",
          "wind_speed_m_s","cloud_index","expected_power_kw","lag_1h_kw",
          "lag_24h_kw","rolling_24h_mean_kw","hour","day_of_year"]
    f = f.dropna(subset=ff+["target_next_hour_kw"])
    tr, te = chronological_split(f)
    Xtr, Xte = tr[ff], te[ff]
    ytr, yte = tr["target_next_hour_kw"], te["target_next_hour_kw"]
    base = DummyRegressor(strategy="mean").fit(Xtr,ytr)
    rf = RandomForestRegressor(n_estimators=180,min_samples_leaf=2,
                               random_state=42,n_jobs=-1).fit(Xtr,ytr)
    pb, p = base.predict(Xte), rf.predict(Xte)
    fm = pd.DataFrame([
      {"model":"Mean baseline","MAE_kw":mean_absolute_error(yte,pb),
       "RMSE_kw":mean_squared_error(yte,pb)**0.5,"R2":r2_score(yte,pb)},
      {"model":"Random Forest","MAE_kw":mean_absolute_error(yte,p),
       "RMSE_kw":mean_squared_error(yte,p)**0.5,"R2":r2_score(yte,p)}
    ])
    fm.to_csv(REPORTS/"forecast_metrics.csv",index=False)

    # Daylight fault classification. Features include derived PR; README flags
    # this as a potential shortcut and asks users to run an ablation without it.
    cf = ["irradiance_w_m2","ambient_temperature_c","module_temperature_c",
          "wind_speed_m_s","cloud_index","expected_power_kw","actual_power_kw",
          "performance_ratio"]
    daylight = data[data.irradiance_w_m2>=80].copy()
    ctr, cte = chronological_split(daylight)
    Xct, Xcv = ctr[cf], cte[cf]
    yct, ycv = ctr.fault_event, cte.fault_event
    dummy = DummyClassifier(strategy="most_frequent").fit(Xct,yct)
    clf = RandomForestClassifier(n_estimators=180,min_samples_leaf=2,
          class_weight="balanced",random_state=42,n_jobs=-1).fit(Xct,yct)
    pred = clf.predict(Xcv)
    prob = clf.predict_proba(Xcv)[:,1]
    cm = confusion_matrix(ycv,pred,labels=[0,1])
    auc = roc_auc_score(ycv,prob) if ycv.nunique()==2 else np.nan
    cm0 = dummy.predict(Xcv)
    cm_auc = np.nan
    cmetrics = pd.DataFrame([
      {"model":"Majority baseline","accuracy":accuracy_score(ycv,cm0),
       "precision":precision_score(ycv,cm0,zero_division=0),
       "recall":recall_score(ycv,cm0,zero_division=0),
       "f1":f1_score(ycv,cm0,zero_division=0),"roc_auc":cm_auc},
      {"model":"Random Forest","accuracy":accuracy_score(ycv,pred),
       "precision":precision_score(ycv,pred,zero_division=0),
       "recall":recall_score(ycv,pred,zero_division=0),
       "f1":f1_score(ycv,pred,zero_division=0),"roc_auc":auc}
    ])
    cmetrics.to_csv(REPORTS/"fault_detection_metrics.csv",index=False)
    (REPORTS/"fault_classification_report.txt").write_text(
       classification_report(ycv,pred,zero_division=0),encoding="utf-8")
    pd.DataFrame(cm,index=["actual_no_fault","actual_fault"],
       columns=["predicted_no_fault","predicted_fault"]).to_csv(
       REPORTS/"fault_confusion_matrix.csv")
    pi = permutation_importance(clf,Xcv,ycv,n_repeats=8,random_state=42,
                                scoring="f1",n_jobs=-1)
    imp = pd.DataFrame({"feature":cf,"mean_f1_decrease":pi.importances_mean,
                        "std":pi.importances_std}).sort_values(
                        "mean_f1_decrease",ascending=False)
    imp.to_csv(REPORTS/"fault_permutation_importance.csv",index=False)

    # Visual reports
    plt.figure(figsize=(10,4))
    plt.plot(te.timestamp.iloc[:240],yte.iloc[:240],label="Observed",lw=1)
    plt.plot(te.timestamp.iloc[:240],p[:240],label="Predicted",lw=1)
    plt.ylabel("Power (kW)"); plt.title("Next-hour PV forecast: held-out sample")
    plt.xticks(rotation=30); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES/"forecast_sample.png",dpi=150); plt.close()

    plt.figure(figsize=(6,5)); plt.imshow(cm,interpolation="nearest")
    plt.title("Fault detection confusion matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1],["No fault","Fault"]); plt.yticks([0,1],["No fault","Fault"])
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.tight_layout(); plt.savefig(FIGURES/"fault_confusion_matrix.png",dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    si=imp.sort_values("mean_f1_decrease")
    plt.barh(si.feature,si.mean_f1_decrease)
    plt.xlabel("Mean decrease in F1 after shuffling")
    plt.title("Permutation importance (fault classification)")
    plt.tight_layout(); plt.savefig(FIGURES/"fault_permutation_importance.png",dpi=150); plt.close()

    summary={"rows":int(len(data)),"start":str(data.timestamp.min()),
      "end":str(data.timestamp.max()),"fault_rate_all_hours":float(data.fault_event.mean()),
      "fault_rate_daylight":float(daylight.fault_event.mean()),
      "forecast_train_rows":int(len(tr)),"forecast_test_rows":int(len(te)),
      "fault_train_rows":int(len(ctr)),"fault_test_rows":int(len(cte)),
      "synthetic_data":True}
    (REPORTS/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Analysis complete. Results saved in reports/.")
    print("\\nFORECAST METRICS\\n",fm.to_string(index=False))
    print("\\nFAULT METRICS\\n",cmetrics.to_string(index=False))

if __name__ == "__main__":
    run()
