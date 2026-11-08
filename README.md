# ML-Based Predictive Maintenance

Predicts whether an industrial machine will **fail within the next 24 hours** from hourly sensor data,
detects abnormal operating conditions, and runs a **real-time monitoring simulation** that labels every
machine as **Normal / Warning / Critical**.

| | Result on unseen test data (Oct–Dec 2015) |
|---|---|
| Model | XGBoost on engineered sensor + component-wear features |
| Precision / Recall / F1 | **0.81 / 0.95 / 0.88** (baseline logistic regression: 0.10 / 0.15 / 0.12) |
| Failures caught in advance | **169 of 171 (98.8%)** Critical, median warning **23 h** before failure. The other 2 reached Warning |
| Normal hours wrongly marked Critical | **0.3%** |

---

## Quick start

Requirements: Python 3.10+ and internet on the first run (the dataset downloads automatically, ~80 MB).

**Windows** – double-click `run.bat`. It sets everything up, trains the model on the first run, and opens the
dashboard at http://localhost:8501.
```bat
run.bat            :: web dashboard (default)
run.bat train      :: train only
run.bat simulate   :: real-time demo in the terminal
```

**macOS / Linux**
```bash
chmod +x run.sh && ./run.sh       # same options: ./run.sh train | simulate
```

**Manual**
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py               # ~2-3 min: preprocessing, plots, models, evaluation
streamlit run src/dashboard.py    # web dashboard
python src/simulate.py            # real-time demo in the terminal (replays unseen real data)
python src/simulate.py --synthetic           # same, with brand-new generated machines
python src/simulate.py --machines 3 17 42 --hours 72 --delay 0.2 --dropout 0.05
```

## Dashboard

A simple web UI (`src/dashboard.py`, Streamlit) with three tabs:

| Tab | What you can do |
|---|---|
| **Overview** | headline metrics, model comparison table, browse all analysis charts |
| **Live monitor** | stream replayed real data or **generated synthetic machines**, and watch the status cards, failure-probability chart and event log update every simulated hour. When a machine is Warning or Critical, click **Swap machine** to replace it with a new unit and restart its sensor; its upcoming failure is then logged as *avoided*. Download the log at the end |
| **Machine explorer** | pick any machine and period, then see its sensors, predicted risk, anomaly score and real failures, plus how early each failure was predicted |

The sidebar has a **Retrain model** button that reruns the full pipeline.

![dashboard overview](docs/dashboard_overview.png)
![dashboard live monitor](docs/dashboard_live.png)

### Optional: What-if predictor (hidden by default)
An extra tab, beyond the assessment brief, where sliders for sensor levels and component age give a live prediction.
Enable it with the `SHOW_WHAT_IF` environment variable:
```powershell
$env:SHOW_WHAT_IF = "true"; .\run.bat      # PowerShell
set SHOW_WHAT_IF=true && run.bat           # Command Prompt
SHOW_WHAT_IF=true ./run.sh                 # macOS / Linux
```

In the What-if tab, try raising vibration to 50. The machine goes **Critical** (98.8%) while comp4 is 60 days old, but stays
**Normal** if comp4 was replaced 10 days ago. The model has learned to combine sensor drift with component wear.

![dashboard what-if](docs/dashboard_whatif.png)

To move the project, copy the folder **without `.venv`** – the run script recreates it.

## Project structure

```
src/data.py        download, cleaning (duplicates, missing hours, glitches), 24h failure labels
src/features.py    feature engineering - shared by training AND real-time (no train/serve skew)
src/train.py       full pipeline: preprocessing -> analysis -> models -> anomaly detection -> evaluation
src/simulate.py    real-time streaming simulation with Normal / Warning / Critical status
src/generator.py   synthetic sensor data generator (healthy + degrading machines)
src/dashboard.py   web dashboard (Streamlit)
models/            trained model (pdm_model.joblib: classifier, thresholds, anomaly detector)
outputs/figures/   all visualisations
outputs/metrics.json              all metrics
outputs/realtime_predictions.csv  log of every live prediction
outputs/realtime_anomalies.csv    log of detected anomalies
```

## Dataset

[Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
(public, downloaded from Microsoft's GitHub): 100 machines, hourly readings for all of 2015.

* **Telemetry** (876,100 rows): voltage, rotation (RPM), pressure, vibration
* **Failures** (761): component failures (comp1–comp4)
* **Maintenance** (3,286): every component replacement, planned or after a failure

---

## How each task is solved

### 1. Data preprocessing
* Duplicate records removed (exact and same machine + timestamp).
* Every machine is re-indexed onto a complete hourly grid, so missing hours become explicit gaps, then **interpolated per machine**.
* **Noisy values / sensor glitches**: readings outside median ± 6·IQR are treated as missing. The bound is wide on purpose so that real degradation is kept.
* Rolling means also smooth the hourly noise.
* **Scaling**: `StandardScaler` inside each sklearn pipeline, fitted on training data only.
* **Train / validation / test split by time**: Jan–Aug train, Sep validation (threshold tuning and model selection), Oct–Dec test. A random split would leak the future into training.
* The dataset is clean (0 duplicates, 0 gaps, 1 glitch). The cleaning code is still needed for real plants, and the real-time simulator injects missing values to exercise it.

### 2. Analysis & feature engineering
24 features per machine-hour:
* per sensor: raw value, **3h moving average**, **24h moving average**, **24h standard deviation**, **trend** (3h mean − 24h mean)
* component wear: **hours since each component was last replaced**, using only maintenance logged *before* the current hour (prevents leakage)

The key finding is that each failure type has a clear sensor signature in the 24 h before it fails:

![failure signatures](outputs/figures/02_failure_signatures.png)

| failed component | sensor that drifts |
|---|---|
| comp1 | voltage ↑ (+4.6σ) |
| comp2 | rotation ↓ (−4.8σ) |
| comp3 | pressure ↑ (+5.6σ) |
| comp4 | vibration ↑ (+5.4σ) |

![sensor trends](outputs/figures/01_sensor_trends_machine.png)
![normal vs abnormal](outputs/figures/03_normal_vs_abnormal.png)

### 3. Model development
Target: `failure within next 24h` (1.9% positive, so heavily imbalanced). Imbalance is handled with class weights and
`scale_pos_weight`, and the decision threshold is tuned on the validation month, not left at the default 0.5.

| Model (test set) | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline: always "normal" | 0.00 | 0.00 | 0.00 | 0.02 |
| Baseline: logistic regression, raw sensors | 0.10 | 0.15 | 0.12 | 0.07 |
| XGBoost, sensor features only | 0.18 | 0.93 | 0.30 | 0.22 |
| Random Forest, all features | 0.70 | 0.98 | 0.81 | 0.86 |
| **XGBoost, all features (selected)** | **0.81** | **0.95** | **0.88** | **0.88** |

![model comparison](outputs/figures/04_model_comparison.png)

What the comparison shows:
* Rolling-window features take sensors from almost useless (F1 0.12) to catching 93% of failure hours.
* With sensors alone, precision stays at about 0.2. The sensors drift sometimes without a failure following
  (see March in the trend plot). **Component wear** tells the model whether a drift happens on a part that is due to fail,
  and adding it raises precision from 0.18 to 0.81.

### 4. Anomaly detection
An **Isolation Forest** is trained only on normal behaviour (> 72 h from any failure), using the sensor features and no labels.
The alarm threshold is the 99th percentile of normal validation scores, which gives about a 1% false-alarm budget.

| Test hours | Flagged as anomalous |
|---|---|
| normal operation | 1.1% |
| 24–72 h before failure | 8.1% |
| < 24 h before failure | 15.2% |

55% of failures have an anomaly in the 24 h before them. It is weaker than the supervised model but needs **no failure history**,
so it works on new machines or failure types never seen before. In the live demo it often fires *earlier* than the classifier.

![anomaly detection](outputs/figures/06_anomaly_detection.png)

### 5. Evaluation
![confusion matrix](outputs/figures/07_confusion_matrix.png)

* **Hourly**: TP 3,864 · FP 889 · FN 189 · TN 216,558
* **Per failure event** (what maintenance teams care about): 169/171 detected in advance, median lead time 23 h.

**False alarms (889 hours)**: 22% happen 24–72 h before a real failure, so they are early warnings rather than true mistakes.
The remaining 78% are more than 7 days from any failure, so they are real false alarms. Overall, only 0.3% of normal hours are marked Critical.

**Missed failures**: 2 of 171 events (one comp3, one comp4). Their probability peaked at 0.75 and 0.80, just under the
Critical threshold (0.86). Both still reached **Warning** and were also flagged by the anomaly detector, so with the
3-level status **no failure went completely unflagged**.

**How to improve**
1. Add the error-log and machine metadata (model, age) as features. Non-fatal errors often come before failures.
2. Predict *which* component will fail (multi-class), so the right spare part is ready before the technician arrives.
3. Require an alarm to persist for N hours, and set the threshold from the cost of a missed failure vs a false alarm, not from F1.
4. Train an LSTM or 1D-CNN directly on raw sequences to learn longer degradation patterns. Check gains against XGBoost, since an
   LSTM costs more to train and explain.
5. Retrain regularly and monitor data drift. Calibrate probabilities so "p = 0.9" really means 90%.
6. Use walk-forward cross-validation over several months instead of a single validation month.

### 6. Real-time prediction simulation
`src/simulate.py` streams sensor data hour by hour as if it came live from the machines. It has two data sources:

* **Replay (default)**: real test-period data the model never saw. Because the actual failure times are known,
  every prediction can be checked against what really happened.
* **Synthetic (`--synthetic`)**: `src/generator.py` creates brand-new machines in Python. Healthy machines produce
  normal noisy readings. Degrading machines (one per failure type) shift the matching sensor about 48 h before
  they fail and then break. All parameters are measured from the real data: shift sizes (voltage +20, RPM −80,
  pressure +25, vibration +10), noise levels, component ages, and the 15-day maintenance cycle.

Both sources go through the same pipeline:

* the feed randomly drops 2% of sensor values to imitate faulty sensors;
* one `MachineMonitor` per machine: it validates each reading (missing or impossible value → last good value),
  keeps a rolling 24 h buffer, computes features with the **same code as training**, and runs the classifier and anomaly detector;
* status rule: **Critical** if P(failure in 24h) ≥ tuned threshold; **Warning** if P ≥ half that threshold *or* an anomaly is detected; otherwise **Normal**;
* actual failures are printed when they happen so predictions can be checked live; every prediction and anomaly is logged to CSV.

Example output (auto-selected window: 3 machines that fail, plus 1 healthy control machine):
```
[2015-11-26 03:00]  M1   Normal   p=0.00   | M16  Normal   p=0.00   | M17  Normal   p=0.00   | M24  Normal   p=0.00
[2015-11-26 04:00]  M1   Normal   p=0.00   | M16  Normal   p=0.00   | M17  Warning  p=0.65   | M24  Normal   p=0.00
[2015-11-26 09:00]  M1   Normal   p=0.00   | M16  Normal   p=0.00   | M17  Critical p=0.96   | M24  Normal   p=0.00
[2015-11-26 21:00]  M1   Normal   p=0.00   | M16  Warning  p=0.00 A | M17  Critical p=0.99   | M24  Normal   p=0.00
[2015-11-27 06:00]  M1   Normal   p=0.00   | M16  Warning  p=0.01 A | M17  Critical p=0.98   | M24  Normal   p=0.00
    >>> ACTUAL FAILURE: machine 17, comp2 replaced
[2015-11-27 07:00]  M1   Normal   p=0.00   | M16  Critical p=1.00   | M17  Normal   p=0.00   | M24  Warning  p=0.79
...
Machine 16 failed at 2015-11-28 06:00:00 (comp2): first Critical alert 23h before
Machine 17 failed at 2015-11-27 06:00:00 (comp2): first Critical alert 21h before
Machine 24 failed at 2015-11-28 06:00:00 (comp3): first Critical alert 22h before
```
(`A` = anomaly detected. Machine 16 was flagged by the anomaly detector 33 h before failure.)

**Synthetic mode results**: across 7 test runs with different random seeds, all 23 generated failures got a Critical alert 19–28 h in advance,
and the healthy machines were never marked Critical.

![synthetic machines in the dashboard](docs/dashboard_synthetic.png)

Building the generator also exposed something the model relies on. In this dataset a failure almost never happens
within 15 days of a maintenance visit (0.5% of real failures), so the model learned that "recently serviced"
means "safe". When the generator first produced recently serviced machines, alerts were suppressed. That is
a reasonable rule for this fleet, but it should be checked with maintenance engineers before deploying on a real plant.

---

## Design decisions
* **No data leakage**: time-based split; the maintenance record created *because of* a failure is excluded from that hour's features; the threshold is tuned on validation, not on test.
* **Event-level metrics** as well as hourly ones: "did we warn before the failure, and how early?" is the question the business asks.
* **Supervised + unsupervised**: the classifier is precise for known failure modes; the anomaly detector covers unknown ones.
* **Simple and portable**: plain Python scripts, 7 dependencies, one command to run, and no GPU or deep-learning framework needed.
  TensorFlow/PyTorch were left out on purpose: gradient boosting already reaches F1 0.88 here, and an LSTM is listed as a next step.
