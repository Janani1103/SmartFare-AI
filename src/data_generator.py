"""
SmartFareAI – Synthetic Taxi Dataset Generator
Generates a realistic 10,000-row taxi trip dataset with correlated features.
"""

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)
N = 10_000

# ─────────────────────────────────────────────
#  Base trip features
# ─────────────────────────────────────────────
trip_distance = np.round(np.random.exponential(scale=8, size=N).clip(0.5, 60), 2)

pickup_hour   = np.random.randint(0, 24, size=N)
is_weekend    = np.random.choice([0, 1], size=N, p=[0.71, 0.29])
passenger_count = np.random.choice([1, 2, 3, 4, 5, 6], size=N,
                                   p=[0.55, 0.20, 0.10, 0.08, 0.05, 0.02])

traffic_options = ["Low", "Medium", "High", "Very High"]
traffic_probs   = [0.25, 0.40, 0.25, 0.10]
traffic_condition = np.random.choice(traffic_options, size=N, p=traffic_probs)

weather_options = ["Clear", "Cloudy", "Rainy", "Foggy", "Snowy"]
weather_probs   = [0.45, 0.25, 0.18, 0.08, 0.04]
weather_condition = np.random.choice(weather_options, size=N, p=weather_probs)

# ─────────────────────────────────────────────
#  Traffic multiplier → affects duration & fare
# ─────────────────────────────────────────────
traffic_map = {"Low": 1.0, "Medium": 1.25, "High": 1.55, "Very High": 1.90}
traffic_mult = np.array([traffic_map[t] for t in traffic_condition])

# ─────────────────────────────────────────────
#  Weather multiplier → affects fare surcharge
# ─────────────────────────────────────────────
weather_map = {"Clear": 1.0, "Cloudy": 1.03, "Rainy": 1.12, "Foggy": 1.08, "Snowy": 1.20}
weather_mult = np.array([weather_map[w] for w in weather_condition])

# ─────────────────────────────────────────────
#  Trip duration (minutes): distance × traffic + noise
# ─────────────────────────────────────────────
base_speed = 40  # km/h
trip_duration = np.round(
    (trip_distance / base_speed) * 60 * traffic_mult + np.random.normal(0, 2, N), 2
).clip(1, 180)

# ─────────────────────────────────────────────
#  Peak hour flag (7-9 AM, 5-8 PM)
# ─────────────────────────────────────────────
is_peak_hour = np.where(
    ((pickup_hour >= 7) & (pickup_hour <= 9)) |
    ((pickup_hour >= 17) & (pickup_hour <= 20)), 1, 0
)

# ─────────────────────────────────────────────
#  Night ride flag (10 PM – 5 AM)
# ─────────────────────────────────────────────
is_night_ride = np.where(
    (pickup_hour >= 22) | (pickup_hour <= 5), 1, 0
)

# ─────────────────────────────────────────────
#  Fare calculation (realistic formula)
# ─────────────────────────────────────────────
base_fare     = 2.50                         # flag-fall
per_km_rate   = 1.80
per_min_rate  = 0.35
peak_surcharge = is_peak_hour * 2.00
night_surcharge = is_night_ride * 1.50
passenger_surcharge = np.where(passenger_count > 4, 1.00, 0.00)

fare_amount = np.round(
    (base_fare
     + per_km_rate * trip_distance
     + per_min_rate * trip_duration
     + peak_surcharge
     + night_surcharge
     + passenger_surcharge)
    * traffic_mult
    * weather_mult
    + np.random.normal(0, 1.2, N),   # measurement noise
    2
).clip(3.00, 200.00)

# ─────────────────────────────────────────────
#  Assemble DataFrame
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "trip_distance":     trip_distance,
    "trip_duration":     trip_duration,
    "passenger_count":   passenger_count,
    "pickup_hour":       pickup_hour,
    "is_weekend":        is_weekend,
    "traffic_condition": traffic_condition,
    "weather_condition": weather_condition,
    "fare_amount":       fare_amount
})

# ─────────────────────────────────────────────
#  Inject ~3% missing values realistically
# ─────────────────────────────────────────────
for col in ["trip_duration", "passenger_count", "weather_condition"]:
    mask = np.random.choice([True, False], size=N, p=[0.03, 0.97])
    df.loc[mask, col] = np.nan

# ─────────────────────────────────────────────
#  Inject ~1% duplicates
# ─────────────────────────────────────────────
dup_idx = np.random.choice(N, size=100, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

# ─────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/taxi_data.csv", index=False)

print(f"[OK] Dataset generated: {len(df):,} rows x {df.shape[1]} columns")
print(f"   Saved to: data/taxi_data.csv")
print(f"\nColumn summary:")
print(df.dtypes)
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFare stats:\n{df['fare_amount'].describe()}")
