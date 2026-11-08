"""Data loading, cleaning and labelling.

Dataset: Microsoft Azure Predictive Maintenance (public, 100 machines, hourly, 2015).
Sensors: voltage, rotation (RPM), pressure, vibration.
Failures: unplanned component replacements. Maintenance: every component replacement (planned + failures).
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

SENSORS = ["volt", "rotate", "pressure", "vibration"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
HORIZON_HOURS = 24  # predict "will this machine fail within the next 24 hours?"

MIRRORS = [
    "https://raw.githubusercontent.com/microsoft/sqlworkshops/master/"
    "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data",
    "https://raw.githubusercontent.com/ashishpatel26/"
    "Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data",
]
FILES = ["PdM_telemetry.csv", "PdM_failures.csv", "PdM_maint.csv"]


def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for base in MIRRORS:
            try:
                print(f"Downloading {name} (first run only) ...")
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(f"{base}/{name}", tmp)
                tmp.replace(dest)
                break
            except Exception as exc:
                print(f"  mirror failed: {exc}")
        else:
            raise RuntimeError(f"Could not download {name}. Place it manually in {DATA_DIR}")


def load_raw():
    download_data()
    telemetry = pd.read_csv(DATA_DIR / "PdM_telemetry.csv")
    failures = pd.read_csv(DATA_DIR / "PdM_failures.csv")
    maint = pd.read_csv(DATA_DIR / "PdM_maint.csv")
    for df in (telemetry, failures, maint):
        df["datetime"] = pd.to_datetime(df["datetime"])
    return telemetry, failures, maint.drop_duplicates()


def sensor_bounds(df):
    """Plausible range per sensor: median +/- 6 IQR. Wide on purpose so real fault signals are kept;
    only sensor glitches fall outside."""
    bounds = {}
    for s in SENSORS:
        q1, med, q3 = df[s].quantile([0.25, 0.5, 0.75])
        bounds[s] = (float(med - 6 * (q3 - q1)), float(med + 6 * (q3 - q1)))
    return bounds


def clean_telemetry(df):
    """Remove duplicates, fill gaps in the hourly grid, and remove impossible sensor values."""
    report = {"rows_raw": len(df)}
    df = df.copy()
    for s in SENSORS:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    before = len(df)
    df = df.drop_duplicates().drop_duplicates(["machineID", "datetime"], keep="first")
    report["duplicates_removed"] = before - len(df)

    # Reindex every machine onto a complete hourly grid so missing hours become explicit NaNs.
    parts = []
    for mid, g in df.groupby("machineID"):
        g = g.set_index("datetime").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(grid)
        g["machineID"] = mid
        parts.append(g.rename_axis("datetime").reset_index())
    df = pd.concat(parts, ignore_index=True)
    report["missing_hours_added"] = len(df) - (before - report["duplicates_removed"])

    # Noise: impossible readings become NaN and are interpolated below.
    outliers = 0
    for s, (lo, hi) in sensor_bounds(df).items():
        bad = (df[s] < lo) | (df[s] > hi)
        outliers += int(bad.sum())
        df.loc[bad, s] = np.nan
    report["outliers_set_to_nan"] = outliers

    report["missing_values_filled"] = int(df[SENSORS].isna().sum().sum())
    df[SENSORS] = (
        df.groupby("machineID")[SENSORS]
        .transform(lambda x: x.interpolate(limit_direction="both"))
    )
    report["rows_clean"] = len(df)
    return df.sort_values(["machineID", "datetime"]).reset_index(drop=True), report


def add_labels(df, failures):
    """Label each hour with whether the machine fails within the next HORIZON_HOURS."""
    events = (
        failures.groupby(["machineID", "datetime"])["failure"]
        .agg(lambda x: "+".join(sorted(x)))
        .reset_index()
        .rename(columns={"datetime": "next_failure", "failure": "next_failure_comp"})
        .sort_values("next_failure")
    )
    out = pd.merge_asof(
        df.sort_values("datetime"), events,
        left_on="datetime", right_on="next_failure", by="machineID", direction="forward",
    )
    out["hours_to_failure"] = (out["next_failure"] - out["datetime"]).dt.total_seconds() / 3600
    out["failure_soon"] = (out["hours_to_failure"] < HORIZON_HOURS).astype(int)
    return out.sort_values(["machineID", "datetime"]).reset_index(drop=True)
