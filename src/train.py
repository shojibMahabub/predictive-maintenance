"""End-to-end pipeline: preprocessing -> analysis -> models -> anomaly detection -> evaluation.

Run:  python src/train.py
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import (
    FIG_DIR, HORIZON_HOURS, MODEL_DIR, OUTPUT_DIR, SENSORS,
    add_labels, clean_telemetry, load_raw, sensor_bounds,
)
from features import AGE_FEATURES, FEATURES, SENSOR_FEATURES, build_features

TRAIN_END = pd.Timestamp("2015-09-01")  # train: Jan-Aug
VAL_END = pd.Timestamp("2015-10-01")    # validation: Sep (threshold tuning / model selection)
                                        # test: Oct-Dec (never seen until final evaluation)
RAW_FEATURES = [f"{s}_raw" for s in SENSORS]
SEED = 42


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=110)
    plt.close()
    print(f"  saved figure: outputs/figures/{name}")


def is_normal(df):
    """Hours clearly far from any failure (> 3 days away, or no upcoming failure)."""
    return df["hours_to_failure"].isna() | (df["hours_to_failure"] > 72)


# --------------------------------------------------------------------------- 1. preprocessing
def preprocess():
    section("1. DATA PREPROCESSING")
    telemetry, failures, maint = load_raw()
    print(f"Telemetry: {telemetry.shape}, machines: {telemetry.machineID.nunique()}, "
          f"{telemetry.datetime.min()} -> {telemetry.datetime.max()}")
    print(f"Failures: {len(failures)} component failures\n{failures.failure.value_counts().to_string()}")
    print(f"Maintenance log: {len(maint)} component replacements (planned + after failure)")

    report = {"sensor_bounds": sensor_bounds(telemetry)}
    df, cleaning = clean_telemetry(telemetry)
    report.update(cleaning)
    print("\nCleaning report:")
    for k, v in cleaning.items():
        print(f"  {k:<24} {v:,}")

    df = add_labels(df, failures)
    df = build_features(df, maint).dropna(subset=FEATURES).reset_index(drop=True)
    print(f"\nLabel: failure within next {HORIZON_HOURS}h -> positive rate "
          f"{df.failure_soon.mean():.2%} (highly imbalanced)")
    return df, report


# --------------------------------------------------------------------------- 2. analysis
def analyse(df):
    section("2. DATA ANALYSIS & FEATURE ENGINEERING")
    print(f"{len(FEATURES)} features per reading:\n"
          f"  sensors   : raw value, 3h mean, 24h mean, 24h std, trend (3h mean - 24h mean) for {SENSORS}\n"
          f"  wear      : hours since each component was last replaced {AGE_FEATURES}")

    # Sensor timeline for the machine with the most failures.
    mid = df.loc[df.hours_to_failure == 0, "machineID"].value_counts().idxmax()
    m = df[(df.machineID == mid) & (df.datetime < "2015-04-01")]
    fails = m.loc[m.hours_to_failure == 0, ["datetime", "next_failure_comp"]]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ax, s in zip(axes, SENSORS):
        ax.plot(m.datetime, m[s], color="lightgray", lw=0.6, label="hourly")
        ax.plot(m.datetime, m[f"{s}_mean_24h"], color="tab:blue", lw=1.3, label="24h mean")
        for _, f in fails.iterrows():
            ax.axvspan(f.datetime - pd.Timedelta(hours=HORIZON_HOURS), f.datetime, color="red", alpha=0.15)
            ax.axvline(f.datetime, color="red", lw=1)
        ax.set_ylabel(s)
    for _, f in fails.iterrows():
        axes[0].annotate(f.next_failure_comp, (f.datetime, axes[0].get_ylim()[1]), color="red", fontsize=8)
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Machine {mid}: sensor trends (red = failure, shaded = 24h before failure)")
    savefig("01_sensor_trends_machine.png")

    # Which sensor drifts before which component fails?
    normal = df[is_normal(df)]
    pre = df[(df.failure_soon == 1) & ~df.next_failure_comp.str.contains(r"\+", na=False)]
    cols = [f"{s}_mean_24h" for s in SENSORS]
    z = (pre.groupby("next_failure_comp")[cols].mean() - normal[cols].mean()) / normal[cols].std()
    z.columns = SENSORS
    print("\nAverage 24h-mean deviation (in std units) in the 24h before each failure type:")
    print(z.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(SENSORS)), SENSORS)
    ax.set_yticks(range(len(z)), z.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center")
    plt.colorbar(im, label="z-score vs normal")
    ax.set_title("Failure signatures: sensor deviation 24h before failure")
    savefig("02_failure_signatures.png")

    # Normal vs abnormal distributions.
    sample = pd.concat([
        normal.sample(20000, random_state=SEED).assign(state="normal"),
        df[df.failure_soon == 1].assign(state="<24h to failure"),
    ])
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, s in zip(axes, SENSORS):
        data = [sample.loc[sample.state == st, f"{s}_mean_24h"] for st in ("normal", "<24h to failure")]
        ax.boxplot(data, tick_labels=["normal", "<24h to\nfailure"], showfliers=False)
        ax.set_title(f"{s} (24h mean)")
    fig.suptitle("Normal vs abnormal machine behaviour")
    savefig("03_normal_vs_abnormal.png")
    return z


# --------------------------------------------------------------------------- 3. models
def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "threshold": thr,
    }


def train_models(train, val, test):
    section("3. MODEL DEVELOPMENT (time-based split, no future leakage)")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:<5} {part.datetime.min().date()} -> {part.datetime.max().date()} "
              f"rows={len(part):,} positives={part.failure_soon.sum():,}")

    y_tr = train.failure_soon
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    # Trees don't need scaling; it is kept so every model shares one interface.
    def xgb():
        return make_pipeline(StandardScaler(), XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=SEED))

    candidates = {
        "Baseline: always normal": (DummyClassifier(strategy="prior"), RAW_FEATURES),
        "Baseline: LogReg on raw sensors": (make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)), RAW_FEATURES),
        "XGBoost (sensor features only)": (xgb(), SENSOR_FEATURES),
        "Random Forest": (make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)), FEATURES),
        "XGBoost": (xgb(), FEATURES),
    }

    results, fitted = {}, {}
    for name, (model, cols) in candidates.items():
        t0 = time.time()
        model.fit(train[cols], y_tr)
        p_val = model.predict_proba(val[cols])[:, 1]
        thr = 0.5 if name.startswith("Baseline: always") else best_threshold(val.failure_soon, p_val)
        p_test = model.predict_proba(test[cols])[:, 1]
        results[name] = {"val_f1": scores(val.failure_soon, p_val, thr)["f1"],
                         **scores(test.failure_soon, p_test, thr)}
        fitted[name] = (model, cols, p_test)
        print(f"  trained {name:<34} in {time.time() - t0:5.1f}s  val F1={results[name]['val_f1']:.3f}")

    table = pd.DataFrame(results).T[["precision", "recall", "f1", "pr_auc", "roc_auc", "threshold", "val_f1"]]
    print("\nTest-set results (threshold tuned on validation month):")
    print(table.round(3).to_string())

    best = max((n for n, (_, cols, _) in fitted.items() if cols == FEATURES),
               key=lambda n: results[n]["val_f1"])
    print(f"\nSelected model (best validation F1): {best}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    table[["precision", "recall", "f1"]].rename(index=lambda n: n.replace(": ", ":\n").replace(" (", "\n(")).plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison on test set (Oct-Dec) - failure within {HORIZON_HOURS}h")
    savefig("04_model_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (_, _, p) in fitted.items():
        if name.startswith("Baseline: always"):
            continue
        prec, rec, _ = precision_recall_curve(test.failure_soon, p)
        ax.plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.2f})")
    ax.axhline(test.failure_soon.mean(), ls="--", color="gray", label="random")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test)")
    ax.legend()
    savefig("05_precision_recall_curves.png")
    return results, fitted, best


# --------------------------------------------------------------------------- 4. anomaly detection
def anomaly_detection(train, val, test):
    section("4. ANOMALY DETECTION (Isolation Forest, unsupervised)")
    normal_train = train[is_normal(train)].sample(100000, random_state=SEED)
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1))
    iso.fit(normal_train[SENSOR_FEATURES])  # trained on normal sensor behaviour only, labels not used

    def score(df):
        return -iso.score_samples(df[SENSOR_FEATURES])  # higher = more unusual

    # Alarm threshold = 99th percentile of normal validation data -> ~1% false-alarm budget.
    thr = float(np.percentile(score(val[is_normal(val)]), 99))
    s = score(test)
    flag = s >= thr

    buckets = pd.cut(test.hours_to_failure.fillna(1e9), [-1, 23.99, 72, 1e10],
                     labels=["< 24h to failure", "24-72h to failure", "normal (> 72h)"])
    rates = pd.Series(flag).groupby(buckets.values, observed=False).mean()
    print("Share of test hours flagged as anomalous:")
    for k, v in rates.items():
        print(f"  {k:<20} {v:6.1%}")
    events = test[test.failure_soon == 1].assign(flag=flag[test.failure_soon.values == 1])
    ev_rate = events.groupby(["machineID", "next_failure"])["flag"].max().mean()
    print(f"Failures with at least one anomaly in the preceding 24h: {ev_rate:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    is_pre = (test.failure_soon == 1).values
    axes[0].hist(s[~is_pre], bins=80, alpha=0.6, density=True, label="normal operation")
    axes[0].hist(s[is_pre], bins=80, alpha=0.6, density=True, label="< 24h before failure")
    axes[0].axvline(thr, color="red", ls="--", label="anomaly threshold")
    axes[0].set_xlabel("anomaly score")
    axes[0].set_title("Isolation Forest separates normal vs pre-failure")
    axes[0].legend()

    mid = test.loc[test.hours_to_failure == 0, "machineID"].value_counts().idxmax()
    m = test.machineID.values == mid
    axes[1].plot(test.datetime[m], s[m], lw=0.7, label="anomaly score")
    axes[1].axhline(thr, color="red", ls="--", label="threshold")
    for t in test.loc[m & (test.hours_to_failure == 0).values, "datetime"]:
        axes[1].axvline(t, color="black", lw=1.2)
    axes[1].set_title(f"Machine {mid} (test period) - black lines = actual failures")
    axes[1].legend()
    savefig("06_anomaly_detection.png")
    return iso, thr, {"flag_rate": rates.to_dict(), "failures_preceded_by_anomaly": ev_rate}


# --------------------------------------------------------------------------- 5. evaluation
def evaluate(test, results, fitted, best):
    section(f"5. MODEL EVALUATION - {best}")
    model, cols, p = fitted[best]
    thr = results[best]["threshold"]
    pred = (p >= thr).astype(int)
    y = test.failure_soon.values

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"Precision {results[best]['precision']:.3f} | Recall {results[best]['recall']:.3f} "
          f"| F1 {results[best]['f1']:.3f}")
    print(f"Confusion matrix (hourly): TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    base_model, base_cols, base_p = fitted["Baseline: LogReg on raw sensors"]
    base_pred = (base_p >= results["Baseline: LogReg on raw sensors"]["threshold"]).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, pr, title in ((axes[0], base_pred, "Baseline: LogReg on raw sensors"), (axes[1], pred, best)):
        ConfusionMatrixDisplay(confusion_matrix(y, pr), display_labels=["normal", "fail<24h"]).plot(
            ax=ax, colorbar=False, values_format=",")
        ax.set_title(title)
    savefig("07_confusion_matrix.png")

    # False alarms: how far were they from a real failure?
    fp_h = test.hours_to_failure.values[(pred == 1) & (y == 0)]
    fp_buckets = {
        "early warning (24-72h before a failure)": int(np.sum((fp_h >= 24) & (fp_h <= 72))),
        "3-7 days before a failure": int(np.sum((fp_h > 72) & (fp_h <= 168))),
        "true false alarm (> 7 days / no failure)": int(np.sum(np.isnan(fp_h) | (fp_h > 168))),
    }
    print("\nFalse alarms analysis (hourly false positives):")
    for k, v in fp_buckets.items():
        print(f"  {k:<42} {v:,} ({v / max(fp, 1):.1%})")

    # Missed failures, measured per failure event (what maintenance actually cares about).
    win = test[test.failure_soon == 1].assign(alarm=pred[y == 1])
    lead = win[win.alarm == 1].groupby(["machineID", "next_failure"])["hours_to_failure"].max()
    events = win.groupby(["machineID", "next_failure"])["next_failure_comp"].first().to_frame()
    events["lead_time_h"] = lead
    events["detected"] = events.lead_time_h.notna()
    by_comp = events.groupby("next_failure_comp")["detected"].agg(["sum", "count", "mean"])
    print(f"\nFailure events in test period: {len(events)} | detected in advance: "
          f"{events.detected.sum()} ({events.detected.mean():.1%}) | missed: {(~events.detected).sum()}")
    print(f"Median warning lead time: {events.lead_time_h.median():.0f}h before failure")
    print("Detection rate by failed component:")
    print(by_comp.rename(columns={"sum": "detected", "count": "events", "mean": "rate"}).round(2).to_string())

    clf = model[-1]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=cols).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot.barh(ax=ax)
        ax.set_title(f"{best}: top 15 features")
        savefig("08_feature_importance.png")
        print("\nTop features:", ", ".join(imp.index[::-1][:6]))

    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_alarms": fp_buckets,
        "events_total": int(len(events)),
        "events_detected": int(events.detected.sum()),
        "event_recall": float(events.detected.mean()),
        "median_lead_time_h": float(events.lead_time_h.median()),
        "event_recall_by_component": by_comp["mean"].round(3).to_dict(),
    }


def status_check(test, p, thr, anomaly_flag):
    """How the Normal / Warning / Critical rule used in real time behaves on test data."""
    status = np.where(p >= thr, "Critical", np.where((p >= thr / 2) | anomaly_flag, "Warning", "Normal"))
    truth = np.where(test.failure_soon == 1, "fails < 24h", np.where(is_normal(test), "normal", "24-72h"))
    table = pd.crosstab(pd.Series(truth, name="actual"), pd.Series(status, name="status"), normalize="index")
    print("\nStatus distribution by actual condition (rows sum to 100%):")
    print((table * 100).round(1).to_string())
    return (table * 100).round(1).to_dict()


def main():
    for d in (MODEL_DIR, OUTPUT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, report = preprocess()
    signatures = analyse(df)

    train = df[df.datetime < TRAIN_END]
    val = df[(df.datetime >= TRAIN_END) & (df.datetime < VAL_END)]
    test = df[df.datetime >= VAL_END].reset_index(drop=True)

    results, fitted, best = train_models(train, val, test)
    iso, iso_thr, anomaly = anomaly_detection(train, val, test)
    evaluation = evaluate(test, results, fitted, best)

    model, cols, p_test = fitted[best]
    thr = results[best]["threshold"]
    anomaly_flag = -iso.score_samples(test[SENSOR_FEATURES]) >= iso_thr
    status = status_check(test, p_test, thr, anomaly_flag)

    joblib.dump({
        "model_name": best, "classifier": model, "features": cols,
        "critical_threshold": thr, "warning_threshold": thr / 2,
        "anomaly_model": iso, "anomaly_features": SENSOR_FEATURES, "anomaly_threshold": iso_thr,
        "sensor_bounds": report["sensor_bounds"],
        "horizon_hours": HORIZON_HOURS, "test_start": str(VAL_END),
    }, MODEL_DIR / "pdm_model.joblib")

    metrics = {
        "data_cleaning": report, "selected_model": best,
        "models_test": results, "evaluation": evaluation,
        "anomaly_detection": anomaly, "status_by_condition_pct": status,
        "failure_signatures_z": signatures.round(2).to_dict(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    section("DONE")
    print(f"Model saved   : models/pdm_model.joblib ({best})")
    print("Metrics saved : outputs/metrics.json")
    print("Figures saved : outputs/figures/")
    print(f"Total time    : {time.time() - t0:.0f}s")
    print("Next: python src/simulate.py   (real-time prediction demo)")


if __name__ == "__main__":
    main()
