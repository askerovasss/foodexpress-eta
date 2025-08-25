import pandas as pd
import numpy as np

TARGET_COL = "time_taken_min_clean"

CATEGORICAL = [
    "city",
    "road_traffic_density",
    "weather_conditions",
    "type_of_order",
    "type_of_vehicle",
    "festival",
]
NUMERIC_BASE = [
    "dist_km",
    "delivery_person_age",
    "delivery_person_ratings",
    "multiple_deliveries",
    "time_to_pick_min",
    "order_hour",
    "order_dow",
]

def _parse_ts(col):
    return pd.to_datetime(col, errors="coerce", dayfirst=True)

def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Час заказа / день недели
    if "order_hour" not in df.columns:
        if "datetime_order" in df.columns:
            ts = _parse_ts(df["datetime_order"])
        elif "order_date" in df.columns and "time_ordered" in df.columns:
            ts = pd.to_datetime(
                df["order_date"].astype(str) + " " + df["time_ordered"].astype(str),
                errors="coerce"
            )
        else:
            ts = pd.Series(pd.NaT, index=df.index)
        df["order_hour"] = ts.dt.hour.fillna(0).astype(int)
        df["order_dow"] = ts.dt.dayofweek.fillna(0).astype(int)
    else:
        if "order_dow" not in df.columns:
            df["order_dow"] = 0

    if "time_to_pick_min" not in df.columns:
        if "datetime_order" in df.columns and "datetime_picked" in df.columns:
            ts_o = _parse_ts(df["datetime_order"])
            ts_p = _parse_ts(df["datetime_picked"])
            df["time_to_pick_min"] = (ts_p - ts_o).dt.total_seconds() / 60.0
        elif "time_ordered" in df.columns and "time_picked" in df.columns and "order_date" in df.columns:
            ts_o = _parse_ts(df["order_date"].astype(str) + " " + df["time_ordered"].astype(str))
            ts_p = _parse_ts(df["order_date"].astype(str) + " " + df["time_picked"].astype(str))
            df["time_to_pick_min"] = (ts_p - ts_o).dt.total_seconds() / 60.0
        else:
            df["time_to_pick_min"] = np.nan

    for c in ["festival", "city", "road_traffic_density", "weather_conditions",
              "type_of_order", "type_of_vehicle"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    for c in ["delivery_person_age", "delivery_person_ratings", "multiple_deliveries", "dist_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def build_frame(df: pd.DataFrame):
    df = _ensure_derived(df)
    numeric = [c for c in NUMERIC_BASE if c in df.columns]
    categorical = [c for c in CATEGORICAL if c in df.columns]
    X = df[numeric + categorical].copy()
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, numeric, categorical, df
