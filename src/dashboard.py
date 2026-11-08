"""Simple web dashboard for the predictive maintenance project.

Run:  streamlit run src/dashboard.py      (or: run.bat dashboard)
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import dotenv_values

from data import COMPONENTS, FIG_DIR, MODEL_DIR, OUTPUT_DIR, ROOT, SENSORS, load_raw
from features import LONG, build_features
from generator import healthy_readings, make_scenario
from simulate import build_monitors, choose_window, classify, sensor_stream

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
MODEL_PATH = MODEL_DIR / "pdm_model.joblib"
# Extra tab beyond the assessment brief. Set SHOW_WHAT_IF in the project .env file.
_env = dotenv_values(ROOT / ".env")
SHOW_WHAT_IF = str(_env.get("SHOW_WHAT_IF", "false")).strip().lower() in ("1", "true", "yes")
COLORS = {"Normal": "#2e7d32", "Warning": "#ef8f00", "Critical": "#c62828"}


@st.cache_data(show_spinner="Loading sensor data ...")
def get_data():
    return load_raw()


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Scoring machine history ...")
def score_machine(mid):
    telemetry, failures, maint = get_data()
    art = get_model()
    df = build_features(telemetry[telemetry.machineID == mid].sort_values("datetime"), maint)
    df = df.dropna(subset=art["features"])
    df["prob"] = art["classifier"].predict_proba(df[art["features"]])[:, 1]
    df["anomaly_score"] = -art["anomaly_model"].score_samples(df[art["anomaly_features"]])
    return df


@st.cache_data
def typical_noise():
    """Median 24h standard deviation of each sensor, used to build realistic what-if inputs."""
    telemetry, _, maint = get_data()
    df = build_features(telemetry[telemetry.machineID <= 5].sort_values(["machineID", "datetime"]), maint)
    return {s: float(df[f"{s}_std_24h"].median()) for s in SENSORS}


ONE_LINE = "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"


def card(title, status, lines, height=None):
    """Status card. With `height`, every line stays on one row so the card never changes size."""
    c = COLORS[status]
    size = f"height:{height}px;box-sizing:border-box;overflow:hidden;" if height else ""
    body = "".join(f"<div style='{ONE_LINE}'>{line}</div>" for line in lines)
    # Font sizes shrink with the card width (cqw) so narrow screens don't cut text off.
    return (f"<div style='border-left:8px solid {c};background:#f4f4f4;color:#222;padding:10px 12px;"
            f"border-radius:6px;margin-bottom:8px;container-type:inline-size;{size}'>"
            f"<div style='{ONE_LINE}font-weight:700;font-size:clamp(12px, 13cqw, 1em)'>{title}</div>"
            f"<div style='{ONE_LINE}color:{c};font-size:min(1.4em, 18cqw);font-weight:700'>{status.upper()}</div>"
            f"<div style='font-size:clamp(10px, 9.7cqw, 0.85em);line-height:1.5'>{body}</div></div>")


@st.cache_data
def healthy_stats():
    telemetry = get_data()[0]
    return telemetry[SENSORS].median(), telemetry[SENSORS].std()


def advance(live):
    """Process the next simulated hour of the live stream."""
    try:
        ts, readings = next(live["stream"])
    except StopIteration:
        finish(live)
        return
    base, noise = healthy_stats()
    for rd in readings:
        m = rd["machineID"]
        if m in live["swapped"]:  # the new unit sends healthy readings
            rd.update(healthy_readings(base, noise, m, ts, 1, live["rng"]).iloc[0][SENSORS].to_dict())
        res = live["monitors"][m].process(rd)
        res["swapped"] = m in live["swapped"]
        previous = live["latest"].get(m, {}).get("status", "Normal")
        if res["status"] != previous:
            live["feed"].insert(0, f"{ts:%m-%d %H:%M} - machine {m} is now **{res['status']}**")
        live["latest"][m] = res
        live["log"].append(res)
    for e in live["fail_events"][live["fail_events"].datetime == ts].itertuples():
        swapped_at = live["swapped"].get(e.machineID)
        if swapped_at is not None and swapped_at < ts:
            hours = (ts - swapped_at).total_seconds() / 3600
            live["feed"].insert(0, f"{ts:%m-%d %H:%M} - :green[**FAILURE AVOIDED** machine {e.machineID} "
                                   f"({e.failure}) - swapped {hours:.0f}h earlier]")
        else:
            live["feed"].insert(0, f"{ts:%m-%d %H:%M} - :red[**ACTUAL FAILURE** machine {e.machineID} ({e.failure})]")
    live["ts"] = ts


def finish(live):
    live["running"] = False
    df = pd.DataFrame(live["log"])
    if len(df):
        df.to_csv(OUTPUT_DIR / "realtime_predictions.csv", index=False)
        df[df.anomaly].to_csv(OUTPUT_DIR / "realtime_anomalies.csv", index=False)
    st.rerun()  # full rerun so the panel stops auto-refreshing


def swap_machine(m):
    live = st.session_state["live"]
    ts = live["ts"]
    base, noise = healthy_stats()
    live["monitors"][m].swap(ts, healthy_readings(base, noise, m, ts, LONG - 1, live["rng"]))
    live["swapped"][m] = ts
    live["latest"][m] = {**live["latest"][m], "status": "Normal", "failure_prob_24h": 0.0, "anomaly": False}
    live["feed"].insert(0, f"{ts:%m-%d %H:%M} - :blue[**machine {m} swapped** for a new unit, sensor restarted]")


def stop_live():
    st.session_state["live"]["stream"] = iter(())


def live_panel():
    live = st.session_state["live"]
    if live["running"]:
        advance(live)
    top = st.columns([4, 1])
    if live["ts"] is not None:
        top[0].subheader(f"{live['ts']:%Y-%m-%d %H:%M}")
    if live["running"]:
        top[1].button("Stop stream", on_click=stop_live, width="stretch")

    for col, m in zip(st.columns(len(live["machines"])), live["machines"]):
        res = live["latest"].get(m)
        if res is None:
            continue
        lines = [f"Risk 24h: <b>{res['failure_prob_24h']:.0%}</b>",
                 f"Anomaly: {'<b>yes</b>' if res['anomaly'] else 'no'}",
                 f"Volt {res['volt']:.0f} · RPM {res['rotate']:.0f}",
                 f"Press {res['pressure']:.0f} · Vib {res['vibration']:.0f}",
                 "<span style='color:#1565c0;font-weight:600'>New unit</span>" if m in live["swapped"] else "&nbsp;"]
        col.markdown(card(f"Machine {m}", res["status"], lines, height=190), unsafe_allow_html=True)
        faulty = live["running"] and res["status"] in ("Warning", "Critical")
        # Always drawn (disabled when healthy) so the layout doesn't jump between refreshes.
        col.button("Swap machine", key=f"swap_{m}", on_click=swap_machine, args=(m,), disabled=not faulty,
                   type="primary" if faulty else "secondary", width="stretch")

    if live["log"]:
        df = pd.DataFrame(live["log"])
        probs = df.pivot(index="datetime", columns="machineID", values="failure_prob_24h")
        st.line_chart(probs.rename(columns=lambda x: f"Machine {x}"), y_label="P(failure in 24h)", height=280)
        st.markdown("**Event log** (newest first)")
        with st.container(height=260):
            st.markdown("\n".join(f"- {x}" for x in live["feed"]) or "No events yet.")
        if not live["running"]:
            st.success(f"Stream finished. Machines swapped: {len(live['swapped'])}. Logs saved to "
                       "outputs/realtime_predictions.csv and realtime_anomalies.csv")
            st.download_button("Download prediction log (CSV)", df.to_csv(index=False), "realtime_predictions.csv")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Predictive Maintenance")
if MODEL_PATH.exists():
    art = get_model()
    st.sidebar.success(f"Model loaded: **{art['model_name']}**")
    st.sidebar.caption(f"Predicts failure within {art['horizon_hours']}h.  \n"
                       f"Critical if P >= {art['critical_threshold']:.2f}  \n"
                       f"Warning if P >= {art['warning_threshold']:.2f} or anomaly")
else:
    st.sidebar.warning("No trained model yet.")

if st.sidebar.button("Retrain model (2-3 min)"):
    with st.spinner("Training ... see the terminal for progress"):
        run = subprocess.run([sys.executable, "-u", str(ROOT / "src" / "train.py")],
                             cwd=ROOT, capture_output=True, text=True)
    st.cache_resource.clear()
    score_machine.clear()
    st.session_state["train_log"] = run.stdout + run.stderr
    st.session_state["train_ok"] = run.returncode == 0
    st.rerun()

if "train_log" in st.session_state:
    (st.sidebar.success if st.session_state["train_ok"] else st.sidebar.error)(
        "Training finished" if st.session_state["train_ok"] else "Training failed - see log")

if not MODEL_PATH.exists():
    st.info("Click **Retrain model** in the sidebar to train the first model.")
    st.stop()

telemetry, failures, maint = get_data()
test_start = pd.Timestamp(art["test_start"])
tabs = st.tabs(["Overview", "Live monitor", "Machine explorer"] + (["What-if predictor"] if SHOW_WHAT_IF else []))
tab_overview, tab_live, tab_machine = tabs[:3]

# ------------------------------------------------------------------ overview
with tab_overview:
    st.header("Model performance on unseen data (Oct-Dec 2015)")
    m = json.loads((OUTPUT_DIR / "metrics.json").read_text())
    best = m["selected_model"]
    r, ev = m["models_test"][best], m["evaluation"]
    c = st.columns(5)
    c[0].metric("Precision", f"{r['precision']:.2f}")
    c[1].metric("Recall", f"{r['recall']:.2f}")
    c[2].metric("F1-score", f"{r['f1']:.2f}")
    c[3].metric("Failures caught", f"{ev['events_detected']}/{ev['events_total']}")
    c[4].metric("Median warning", f"{ev['median_lead_time_h']:.0f} h before")

    st.subheader("Model comparison")
    table = pd.DataFrame(m["models_test"]).T[["precision", "recall", "f1", "pr_auc"]].astype(float)
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"), width="stretch")

    st.subheader("Charts")
    figs = sorted(FIG_DIR.glob("*.png"))
    fig = st.selectbox("Chart", figs, format_func=lambda p: p.stem[3:].replace("_", " ").capitalize())
    st.image(str(fig), width="stretch")

    if "train_log" in st.session_state:
        with st.expander("Last training log"):
            st.code(st.session_state["train_log"])

# ------------------------------------------------------------------ live monitor
with tab_live:
    st.header("Live monitor")
    source = st.radio("Data source", ["Replay real unseen data (Oct-Dec 2015)", "Generate synthetic machines"],
                      horizontal=True)
    synthetic = source.startswith("Generate")
    if synthetic:
        st.caption("Creates brand-new machines in Python: 4 that slowly degrade (one per failure type) and 1 healthy. "
                   "The degradation patterns are calibrated from the real data.")
    else:
        st.caption("Replays sensor data the model never saw, hour by hour, as if it were streaming from the machines. "
                   "Leave machines empty to auto-pick machines that really fail in the window, plus one healthy one.")
    c = st.columns(4)
    if synthetic:
        seed = c[0].number_input("Random seed", 0, 9999, 7)
    else:
        chosen = c[0].multiselect("Machines", list(range(1, 101)), max_selections=6)
    hours = c[1].slider("Hours to stream", 24, 240, 96, step=24)
    delay = c[2].slider("Seconds per simulated hour", 0.0, 1.0, 0.1, step=0.05)
    dropout = c[3].slider("Sensor dropout rate", 0.0, 0.2, 0.02, step=0.01)

    if st.button("Start live stream", type="primary"):
        if synthetic:
            sc = make_scenario(telemetry, seed=int(seed))
            machines, start, fail_events = sc.machines, sc.start, sc.failures
            monitors = build_monitors(machines, start, sc.history, sc.maint, art)
            stream = sc.stream(hours, dropout)
        else:
            start, machines = choose_window(failures, test_start, hours, chosen)
            monitors = build_monitors(machines, start, telemetry, maint, art)
            fail_events = failures[failures.machineID.isin(machines)]
            stream = sensor_stream(telemetry, machines, start, hours, dropout, np.random.default_rng(7))
        # The stream lives in the session so buttons (Swap / Stop) work while it runs.
        st.session_state["live"] = {
            "monitors": monitors, "stream": stream, "machines": machines, "start": start,
            "fail_events": fail_events, "log": [], "feed": [], "latest": {}, "swapped": {},
            "running": True, "ts": None, "rng": np.random.default_rng(1),
        }

    if "live" in st.session_state:
        live = st.session_state["live"]
        st.write(f"Streaming from **{live['start']:%Y-%m-%d %H:%M}** for machines "
                 f"**{', '.join(map(str, live['machines']))}**")
        st.fragment(live_panel, run_every=max(delay, 0.1) if live["running"] else None)()

# ------------------------------------------------------------------ machine explorer
@st.fragment  # widgets here only rerun this tab, keeping the dashboard responsive
def machine_explorer():
    st.header("Machine explorer")
    c = st.columns([1, 3])
    mid = c[0].selectbox("Machine", list(range(1, 101)), index=16)
    df = score_machine(mid)
    lo, hi = df.datetime.min().to_pydatetime(), df.datetime.max().to_pydatetime()
    period = c[1].slider("Period", lo, hi, (test_start.to_pydatetime(), hi), format="YYYY-MM-DD")
    sensors = st.multiselect("Sensors (24h average)", SENSORS, default=SENSORS)
    st.caption("Data before October 2015 was used for training; from October on the model has never seen it.")

    view = df[(df.datetime >= period[0]) & (df.datetime <= period[1])]
    fails = failures[(failures.machineID == mid) & failures.datetime.between(*period)]
    n = len(sensors) + 2
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.7 * n), sharex=True)
    for ax, s in zip(axes, sensors):
        ax.plot(view.datetime, view[s], color="lightgray", lw=0.5)
        ax.plot(view.datetime, view[f"{s}_mean_24h"], lw=1.2)
        ax.set_ylabel(s)
    axes[-2].fill_between(view.datetime, view.prob, alpha=0.4, color="tab:red")
    axes[-2].axhline(art["critical_threshold"], color="red", ls="--", lw=0.8)
    axes[-2].axhline(art["warning_threshold"], color="orange", ls="--", lw=0.8)
    axes[-2].set_ylabel("P(fail 24h)")
    axes[-1].plot(view.datetime, view.anomaly_score, lw=0.6, color="tab:purple")
    axes[-1].axhline(art["anomaly_threshold"], color="purple", ls="--", lw=0.8)
    axes[-1].set_ylabel("anomaly")
    for ax in axes:
        for t in fails.datetime:
            ax.axvline(t, color="black", lw=1)
    axes[0].set_title("Black lines = actual failures | dashed = alert thresholds")
    plt.tight_layout()
    st.pyplot(fig)

    rows = []
    for f in fails.itertuples():
        before = df[(df.datetime <= f.datetime) & (df.datetime > f.datetime - pd.Timedelta(hours=48))]
        crit = before[before.prob >= art["critical_threshold"]]
        lead = (f.datetime - crit.datetime.min()).total_seconds() / 3600 if len(crit) else None
        rows.append({"failure time": f.datetime, "component": f.failure,
                     "predicted": "yes" if lead is not None else "no",
                     "first Critical alert (h before)": lead})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.write("No failures in this period.")

# ------------------------------------------------------------------ what-if
@st.fragment
def what_if():
    st.header("What-if predictor")
    st.caption("Set a machine's average sensor levels over the last 24 hours and its component ages, "
               "and see what the model predicts. Try vibration = 50: Critical while comp4 is 60 days old, "
               "but Normal if comp4 was replaced 10 days ago - the model combines sensor drift with component wear.")
    noise = typical_noise()
    c1, c2, c3 = st.columns([2, 2, 2])
    values = {}
    with c1:
        st.subheader("Sensors (24h average)")
        for s in SENSORS:
            q = telemetry[s].quantile([0.001, 0.5, 0.999])
            values[s] = st.slider(s, float(round(q.iloc[0])), float(round(q.iloc[2])), float(round(q.iloc[1])),
                                  step=1.0, format="%.0f")
    with c2:
        st.subheader("Days since component replaced")
        ages = {c: st.slider(c, 0, 365, 60) for c in COMPONENTS}

    row = {}
    for s in SENSORS:
        for k in ("raw", "mean_3h", "mean_24h"):
            row[f"{s}_{k}"] = values[s]
        row[f"{s}_std_24h"] = noise[s]
        row[f"{s}_trend"] = 0.0
    for comp in COMPONENTS:
        row[f"{comp}_age_h"] = ages[comp] * 24.0
    X = pd.DataFrame([row])
    p = float(art["classifier"].predict_proba(X[art["features"]])[0, 1])
    a = float(-art["anomaly_model"].score_samples(X[art["anomaly_features"]])[0])
    anomaly = a >= art["anomaly_threshold"]
    status = classify(art, p, anomaly)
    with c3:
        st.subheader("Prediction")
        st.markdown(card("Machine status", status, [
            f"P(failure in 24h) = {p:.1%}",
            f"Anomaly score {a:.3f} ({'ANOMALY' if anomaly else 'normal'})",
        ]), unsafe_allow_html=True)
        st.progress(min(p, 1.0))


with tab_machine:
    machine_explorer()
if SHOW_WHAT_IF:
    with tabs[3]:
        what_if()
