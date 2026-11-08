"""Synthetic sensor data generator.

Creates brand-new machines: healthy ones produce normal noisy readings, faulty ones slowly drift
in the sensor linked to the failing component (the signatures found in the real data) and then fail.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from data import COMPONENTS, SENSORS
from features import LONG

# Calibrated from the real data: the fault shows up ~48h before failure as a sustained shift of one sensor.
# component -> (sensor that drifts, size of the shift)
SIGNATURES = {
    "comp1": ("volt", +20.0),
    "comp2": ("rotate", -80.0),
    "comp3": ("pressure", +25.0),
    "comp4": ("vibration", +10.0),
}
DRIFT_HOURS = 48   # degradation starts this long before the failure
RAMP_HOURS = 6     # time for the drift to reach full size


def healthy_readings(base, noise, machine_id, end, hours, rng):
    """`hours` normal readings for one machine, ending at `end` (e.g. a freshly installed unit)."""
    return pd.DataFrame([
        {"datetime": end - pd.Timedelta(hours=h), "machineID": machine_id,
         **dict(zip(SENSORS, base.values + noise.values * rng.standard_normal(len(SENSORS))))}
        for h in range(hours - 1, -1, -1)
    ])


@dataclass
class Scenario:
    machines: list
    start: pd.Timestamp
    history: pd.DataFrame      # 23h of normal readings before start, to prime the buffers
    maint: pd.DataFrame        # synthetic maintenance log (incl. the replacement at each failure)
    failures: pd.DataFrame     # planned failures: machineID, datetime, failure
    stream: Callable           # stream(hours, dropout) -> yields (timestamp, readings)


def make_scenario(telemetry, faults=("comp2", "comp4", "comp1", "comp3"), n_healthy=1, seed=7):
    rng = np.random.default_rng(seed)
    base = telemetry[SENSORS].median()
    noise = telemetry[SENSORS].std()
    start = pd.Timestamp.now().floor("h")
    machines = list(range(101, 101 + len(faults) + n_healthy))
    offset = {m: rng.normal(0, 0.03, len(SENSORS)) * base.values for m in machines}  # each machine a bit different

    plan = {}
    for i, comp in enumerate(faults):
        plan[machines[i]] = (comp, start + pd.Timedelta(hours=56 + 10 * i))
    failures = pd.DataFrame([{"machineID": m, "datetime": t, "failure": c} for m, (c, t) in plan.items()],
                            columns=["machineID", "datetime", "failure"])

    # Matches the real fleet: maintenance runs on a 15-day cycle, parts fail at 45-120 days old, and a
    # failure practically never comes within 15 days of a maintenance visit (0.5% of real failures).
    maint = []
    for m in machines:
        for c in COMPONENTS:
            if m in plan and plan[m][0] == c:
                last = plan[m][1] - pd.Timedelta(days=int(rng.choice([45, 60, 75, 90, 105, 120])))
            elif m in plan:
                last = plan[m][1] - pd.Timedelta(days=15 * int(rng.integers(1, 5)))
            else:
                last = start - pd.Timedelta(days=float(np.clip(rng.exponential(45), 1, 300)))
            maint.append({"machineID": m, "datetime": last, "comp": c})
    maint += [{"machineID": m, "datetime": t, "comp": c} for m, (c, t) in plan.items()]
    maint = pd.DataFrame(maint)

    def reading(m, ts):
        values = base.values + offset[m] + noise.values * rng.standard_normal(len(SENSORS))
        if m in plan:
            comp, fail_at = plan[m]
            onset = fail_at - pd.Timedelta(hours=DRIFT_HOURS)
            if onset <= ts <= fail_at:
                sensor, size = SIGNATURES[comp]
                ramp = min(1.0, (ts - onset) / pd.Timedelta(hours=RAMP_HOURS))
                values[SENSORS.index(sensor)] += size * ramp
        return {"datetime": ts, "machineID": m, **dict(zip(SENSORS, values))}

    history = pd.DataFrame([reading(m, start - pd.Timedelta(hours=h))
                            for m in machines for h in range(LONG - 1, 0, -1)])

    def stream(hours, dropout):
        for h in range(hours):
            ts = start + pd.Timedelta(hours=h)
            readings = []
            for m in machines:
                r = reading(m, ts)
                for s in SENSORS:
                    if rng.random() < dropout:
                        r[s] = np.nan
                readings.append(r)
            yield ts, readings

    return Scenario(machines, start, history, maint, failures, stream)
