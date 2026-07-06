"""
SmartFareAI – Feature Engineering (Phase 3)
Derives new predictive features from cleaned taxi trip data.
"""

import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────
#  Feature Engineering Functions
# ─────────────────────────────────────────────

def add_average_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Average speed in km/h = distance / (duration / 60)"""
    df["avg_speed_kmh"] = np.where(
        df["trip_duration"] > 0,
        np.round(df["trip_distance"] / (df["trip_duration"] / 60), 2),
        0
    )
    return df


def add_peak_hour_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """1 if pickup during morning (7–9) or evening (17–20) rush hours"""
    df["is_peak_hour"] = np.where(
        ((df["pickup_hour"] >= 7)  & (df["pickup_hour"] <= 9)) |
        ((df["pickup_hour"] >= 17) & (df["pickup_hour"] <= 20)),
        1, 0
    )
    return df


def add_night_ride_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """1 if pickup between 10 PM and 5 AM"""
    df["is_night_ride"] = np.where(
        (df["pickup_hour"] >= 22) | (df["pickup_hour"] <= 5),
        1, 0
    )
    return df


def add_fare_per_km(df: pd.DataFrame) -> pd.DataFrame:
    """Fare per kilometre (useful for model calibration)"""
    df["fare_per_km"] = np.where(
        df["trip_distance"] > 0,
        np.round(df["fare_amount"] / df["trip_distance"], 4),
        0
    )
    return df


def add_duration_per_km(df: pd.DataFrame) -> pd.DataFrame:
    """Duration per kilometre → proxy for congestion"""
    df["duration_per_km"] = np.where(
        df["trip_distance"] > 0,
        np.round(df["trip_duration"] / df["trip_distance"], 4),
        0
    )
    return df


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucketed distance: Short / Medium / Long / Very Long"""
    bins   = [0, 5, 15, 30, np.inf]
    labels = [0, 1, 2, 3]           # 0=Short, 1=Medium, 2=Long, 3=VeryLong
    df["distance_bucket"] = pd.cut(
        df["trip_distance"], bins=bins, labels=labels
    ).astype(int)
    return df


def add_hour_period(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical time of day: Night / Morning / Afternoon / Evening"""
    conditions = [
        (df["pickup_hour"] >= 0)  & (df["pickup_hour"] < 6),   # Night
        (df["pickup_hour"] >= 6)  & (df["pickup_hour"] < 12),  # Morning
        (df["pickup_hour"] >= 12) & (df["pickup_hour"] < 18),  # Afternoon
        (df["pickup_hour"] >= 18) & (df["pickup_hour"] <= 23), # Evening
    ]
    labels = [0, 1, 2, 3]
    df["hour_period"] = np.select(conditions, labels, default=0)
    return df


# ─────────────────────────────────────────────
#  Master pipeline
# ─────────────────────────────────────────────
def run_feature_engineering(input_path: str = "data/taxi_data_cleaned.csv") -> pd.DataFrame:
    """
    Loads cleaned (pre-encoding) data, engineers features,
    saves result, and returns the enriched DataFrame.
    """
    print(f"\n{'='*55}")
    print(f"  PHASE 3 – Feature Engineering")
    print(f"{'='*55}")

    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")

    df = add_average_speed(df)
    df = add_peak_hour_indicator(df)
    df = add_night_ride_indicator(df)
    df = add_fare_per_km(df)
    df = add_duration_per_km(df)
    df = add_distance_bucket(df)
    df = add_hour_period(df)

    new_features = [
        "avg_speed_kmh", "is_peak_hour", "is_night_ride",
        "fare_per_km",   "duration_per_km",
        "distance_bucket", "hour_period"
    ]
    print(f"\n  ✅ New features added ({len(new_features)}):")
    for f in new_features:
        stats = df[f].describe()
        print(f"     {f:22s}  mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    os.makedirs("data", exist_ok=True)
    out_path = "data/taxi_data_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Feature-engineered data saved: {out_path}")
    print(f"     Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*55}\n")

    return df


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_feature_engineering()
    print(df[["trip_distance", "trip_duration", "fare_amount",
              "avg_speed_kmh", "is_peak_hour", "is_night_ride",
              "fare_per_km"]].head(10).to_string())
