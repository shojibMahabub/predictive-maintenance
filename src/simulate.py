"""Real-time prediction simulation.

Streams sensor data hour by hour as if it were arriving live from the machines - either replayed
unseen test-period data, or brand-new data from the synthetic generator (--synthetic).
Each machine keeps a rolling 24h buffer; features are computed with the same code used in training,
and the machine status is reported as Normal / Warning / Critical. Everything is logged to CSV.

Run:  python src/simulate.py                         (auto-picks a window with real failures)
      python src/simulate.py --machines 3 17 42 --hours 96 --delay 0
      python src/simulate.py --synthetic             (generated machines that degrade and fail)
"""
import argparse
import os
import sys
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd

from data import COMPONENTS, MODEL_DIR, OUTPUT_DIR, SENSORS, load_raw
from features import LONG, latest_features
from generator import make_scenario

COLORS = {"Normal": "\033[92m", "Warning": "\033[93m", "Critical": "\033[91m", "event": "\033[96m"}
RESET = "\033[0m"


def classify(artifact, p, anomaly):
    if p >= artifact["critical_threshold"]:
        return "Critical"
    if p >= artifact["warning_threshold"] or anomaly:
        return "Warning"
    return "Normal"


class MachineMonitor:
    """Online processor for one machine: validate -> buffer -> features -> predict -> status."""

    def __init__(self, machine_id, artifact, maint, history):
        self.id = machine_id
        self.art = artifact
        self.maint = maint[maint.machineID == machine_id]
        self.buffer = deque(history.to_dict("records"), maxlen=LONG)
        self.last_good = {s: history[s].iloc[-1] for s in SENSORS}

    def process(self, reading):
        imputed = []
        for s in SENSORS:
            lo, hi = self.art["sensor_bounds"][s]
            v = reading[s]
            if pd.isna(v) or not lo <= v <= hi:  # dropout or glitch -> carry last good value
                reading[s] = self.last_good[s]
                imputed.append(s)
            self.last_good[s] = reading[s]
        self.buffer.append(reading)

        X = latest_features(pd.DataFrame(self.buffer), self.maint)
        p = float(self.art["classifier"].predict_proba(X[self.art["features"]])[0, 1])
        a_score = float(-self.art["anomaly_model"].score_samples(X[self.art["anomaly_features"]])[0])
        anomaly = a_score >= self.art["anomaly_threshold"]
        return {**reading, "failure_prob_24h": round(p, 4), "anomaly_score": round(a_score, 4),
                "anomaly": anomaly, "status": classify(self.art, p, anomaly), "imputed": ",".join(imputed)}

    def swap(self, ts, history):
        """Faulty unit replaced by a new one: all components new, sensor buffer restarted."""
        new_parts = pd.DataFrame({"machineID": self.id, "datetime": ts, "comp": COMPONENTS})
        self.maint = pd.concat([self.maint, new_parts], ignore_index=True)
        self.buffer = deque(history.to_dict("records"), maxlen=LONG)
        self.last_good = {s: history[s].iloc[-1] for s in SENSORS}


def sensor_stream(telemetry, machines, start, hours, dropout, rng):
    """Simulated IoT feed: one reading per machine per hour, with random sensor dropouts."""
    end = start + pd.Timedelta(hours=hours)
    feed = telemetry[telemetry.machineID.isin(machines) & (telemetry.datetime >= start)
                     & (telemetry.datetime < end)].sort_values(["datetime", "machineID"])
    for ts, group in feed.groupby("datetime"):
        readings = []
        for row in group.itertuples(index=False):
            r = {"datetime": ts, "machineID": row.machineID, **{s: getattr(row, s) for s in SENSORS}}
            for s in SENSORS:
                if rng.random() < dropout:
                    r[s] = np.nan
            readings.append(r)
        yield ts, readings


def pick_demo(failures, test_start, hours, n_failing):
    """Choose a window in the unseen test period where several machines really fail."""
    f = failures[failures.datetime >= test_start + pd.Timedelta(days=3)].drop_duplicates(["machineID", "datetime"])
    best = (None, [])
    for t in sorted(f.datetime.unique()):
        start = t - pd.Timedelta(hours=60)
        # Failures at least 48h in, so each machine is seen going Normal -> Warning -> Critical.
        inside = f[(f.datetime >= start + pd.Timedelta(hours=48)) & (f.datetime <= start + pd.Timedelta(hours=hours - 12))]
        if inside.machineID.nunique() > len(best[1]):
            best = (start, sorted(int(m) for m in inside.machineID.unique()))
    start, failing = best
    busy = failures[(failures.datetime >= start) & (failures.datetime <= start + pd.Timedelta(hours=hours + 48))]
    healthy = [m for m in range(1, 101) if m not in set(busy.machineID)][:1]
    return start, failing[:n_failing] + healthy


def choose_window(failures, test_start, hours, machines=None, start=None):
    """Resolve which machines to stream and from when, filling in sensible defaults."""
    if not machines:
        auto_start, machines = pick_demo(failures, test_start, hours, n_failing=3)
        return (pd.Timestamp(start) if start else auto_start), machines
    if start:
        return pd.Timestamp(start), machines
    mf = failures[(failures.machineID == machines[0]) & (failures.datetime >= test_start + pd.Timedelta(days=3))]
    auto_start = mf.datetime.min() - pd.Timedelta(hours=60) if len(mf) else test_start + pd.Timedelta(days=1)
    return auto_start, machines


def build_monitors(machines, start, telemetry, maint, artifact):
    """One monitor per machine, its buffer primed with the previous 23 hours (the 'historian')."""
    monitors = {}
    for m in machines:
        hist = telemetry[(telemetry.machineID == m) & (telemetry.datetime < start)].tail(LONG - 1)
        monitors[m] = MachineMonitor(m, artifact, maint, hist[["datetime", "machineID", *SENSORS]])
    return monitors


def main():
    ap = argparse.ArgumentParser(description="Real-time predictive maintenance simulation")
    ap.add_argument("--machines", type=int, nargs="*", help="machine IDs (default: auto-pick)")
    ap.add_argument("--start", help="start timestamp, e.g. 2015-11-01 (default: auto)")
    ap.add_argument("--hours", type=int, default=96, help="hours to simulate (default 96)")
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between hourly ticks")
    ap.add_argument("--dropout", type=float, default=0.02, help="probability a sensor value is missing")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--synthetic", action="store_true",
                    help="generate brand-new machine data instead of replaying the real test data")
    args = ap.parse_args()

    model_path = MODEL_DIR / "pdm_model.joblib"
    if not model_path.exists():
        sys.exit("No trained model found. Run:  python src/train.py")
    art = joblib.load(model_path)
    telemetry, failures, maint = load_raw()
    test_start = pd.Timestamp(art["test_start"])

    if args.synthetic:
        sc = make_scenario(telemetry, seed=args.seed)
        machines, start, telemetry, maint, failures = sc.machines, sc.start, sc.history, sc.maint, sc.failures
        stream = sc.stream(args.hours, args.dropout)
        source = "synthetic generated"
    else:
        start, machines = choose_window(failures, test_start, args.hours, args.machines, args.start)
        if start < test_start:
            print("Note: start is before the test period - the model saw this data during training.")
        stream = sensor_stream(telemetry, machines, start, args.hours, args.dropout, np.random.default_rng(args.seed))
        source = "unseen real"

    use_color = sys.stdout.isatty()
    if use_color and os.name == "nt":
        os.system("")  # enables ANSI colours in the Windows console

    def paint(text, key):
        return f"{COLORS[key]}{text}{RESET}" if use_color else text

    monitors = build_monitors(machines, start, telemetry, maint, art)

    fail_events = failures[failures.machineID.isin(machines)].groupby(["machineID", "datetime"])["failure"] \
        .agg("+".join).reset_index()
    print(f"\nModel: {art['model_name']} | horizon: failure within {art['horizon_hours']}h | "
          f"Critical >= {art['critical_threshold']:.2f}, Warning >= {art['warning_threshold']:.2f} or anomaly")
    print(f"Streaming {args.hours}h of {source} data from {start} for machines {machines} "
          f"(sensor dropout rate {args.dropout:.0%})\n")

    log = []
    try:
        for ts, readings in stream:
            cells = []
            for r in readings:
                res = monitors[r["machineID"]].process(r)
                log.append(res)
                flag = " A" if res["anomaly"] else "  "
                cells.append(paint(f"M{r['machineID']:<3} {res['status']:<8} p={res['failure_prob_24h']:.2f}{flag}",
                                   res["status"]))
            print(f"[{ts:%Y-%m-%d %H:%M}]  " + " | ".join(cells))
            for e in fail_events[fail_events.datetime == ts].itertuples():
                print(paint(f"    >>> ACTUAL FAILURE: machine {e.machineID}, {e.failure} replaced", "event"))
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    if not log:
        sys.exit("No data in the selected window.")
    df = pd.DataFrame(log)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "realtime_predictions.csv", index=False)
    df[df.anomaly].to_csv(OUTPUT_DIR / "realtime_anomalies.csv", index=False)

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print("Hours per status:")
    print(df.pivot_table(index="machineID", columns="status", values="datetime", aggfunc="count", fill_value=0)
          .to_string())
    window = fail_events[(fail_events.datetime >= start) & (fail_events.datetime <= df.datetime.max())]
    for e in window.itertuples():
        before = df[(df.machineID == e.machineID) & (df.datetime <= e.datetime)
                    & (df.datetime > e.datetime - pd.Timedelta(hours=48))]
        crit = before[before.status == "Critical"]
        lead = f"first Critical alert {(e.datetime - crit.datetime.min()).total_seconds() / 3600:.0f}h before" \
            if len(crit) else "NOT predicted"
        print(f"Machine {e.machineID} failed at {e.datetime} ({e.failure}): {lead}")
    print(f"Sensor dropouts handled: {(df.imputed != '').sum()} readings had imputed values")
    print(f"Anomalies detected: {int(df.anomaly.sum())}")
    print("Logs: outputs/realtime_predictions.csv, outputs/realtime_anomalies.csv")


if __name__ == "__main__":
    main()
