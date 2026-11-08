"""Feature engineering shared by training and the real-time simulator (same code = no skew)."""
import pandas as pd

from data import COMPONENTS, SENSORS

SHORT, LONG = 3, 24  # hours
SENSOR_FEATURES = [
    f"{s}_{kind}"
    for s in SENSORS
    for kind in ("raw", "mean_3h", "mean_24h", "std_24h", "trend")
]
AGE_FEATURES = [f"{c}_age_h" for c in COMPONENTS]
FEATURES = SENSOR_FEATURES + AGE_FEATURES


def add_sensor_features(df):
    """Rolling statistics per machine. Expects rows sorted by machineID, datetime."""
    df = df.copy()
    g = df.groupby("machineID", sort=False)
    for s in SENSORS:
        roll_short = g[s].rolling(SHORT, min_periods=SHORT)
        roll_long = g[s].rolling(LONG, min_periods=LONG)
        df[f"{s}_raw"] = df[s]
        df[f"{s}_mean_3h"] = roll_short.mean().reset_index(level=0, drop=True)
        df[f"{s}_mean_24h"] = roll_long.mean().reset_index(level=0, drop=True)
        df[f"{s}_std_24h"] = roll_long.std().reset_index(level=0, drop=True)
        # Positive trend = the last 3 hours are running above the daily level.
        df[f"{s}_trend"] = df[f"{s}_mean_3h"] - df[f"{s}_mean_24h"]
    return df


def add_component_age(df, maint):
    """Hours since each component was last replaced (wear), using only maintenance logged before t."""
    out = df.sort_values("datetime").astype({"datetime": "datetime64[ns]"})
    maint = maint.astype({"datetime": "datetime64[ns]"})
    log_start = maint["datetime"].min()
    for c in COMPONENTS:
        events = (maint.loc[maint["comp"] == c, ["machineID", "datetime"]]
                  .rename(columns={"datetime": "_last"}).sort_values("_last"))
        # allow_exact_matches=False: a replacement made *because* of a failure must not leak into that hour.
        out = pd.merge_asof(out, events, left_on="datetime", right_on="_last", by="machineID",
                            direction="backward", allow_exact_matches=False)
        never = out["datetime"] - log_start
        out[f"{c}_age_h"] = (out["datetime"] - out["_last"]).fillna(never).dt.total_seconds() / 3600
        out = out.drop(columns="_last")
    return out.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def build_features(df, maint):
    return add_component_age(add_sensor_features(df), maint)


def latest_features(buffer: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """Features for the newest reading in a single machine's rolling buffer."""
    return build_features(buffer, maint)[FEATURES].tail(1)
