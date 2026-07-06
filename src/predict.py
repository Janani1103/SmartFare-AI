"""
SmartFareAI – Prediction Utility
Loads the trained best model and generates fare predictions with confidence intervals.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────
#  Load artifacts
# ─────────────────────────────────────────────
def load_model_artifacts(models_dir: str = "models"):
    model   = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
    columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    with open(os.path.join(models_dir, "model_metrics.json")) as f:
        metrics = json.load(f)
    return model, scaler, columns, metrics


# ─────────────────────────────────────────────
#  Input → Feature vector
# ─────────────────────────────────────────────
def build_feature_vector(
    trip_distance:     float,
    trip_duration:     float,
    passenger_count:   int,
    pickup_hour:       int,
    is_weekend:        int,
    traffic_condition: str,
    weather_condition: str,
    expected_columns:  list
) -> pd.DataFrame:
    """
    Constructs a 1-row DataFrame matching the training feature space.
    """
    # Peak / night indicators
    is_peak_hour  = 1 if (7 <= pickup_hour <= 9) or (17 <= pickup_hour <= 20) else 0
    is_night_ride = 1 if (pickup_hour >= 22) or (pickup_hour <= 5) else 0

    # Average speed (approximate – traffic-adjusted)
    traffic_speed = {"Low": 40, "Medium": 32, "High": 26, "Very High": 21}
    speed = traffic_speed.get(traffic_condition, 32)
    avg_speed_kmh = round(trip_distance / (trip_duration / 60), 2) if trip_duration > 0 else speed

    # Duration per km
    duration_per_km = round(trip_duration / trip_distance, 4) if trip_distance > 0 else 0

    # Distance bucket
    if trip_distance <= 5:   distance_bucket = 0
    elif trip_distance <= 15: distance_bucket = 1
    elif trip_distance <= 30: distance_bucket = 2
    else:                    distance_bucket = 3

    # Hour period
    if   0  <= pickup_hour < 6:  hour_period = 0   # Night
    elif 6  <= pickup_hour < 12: hour_period = 1   # Morning
    elif 12 <= pickup_hour < 18: hour_period = 2   # Afternoon
    else:                        hour_period = 3   # Evening

    # Base row
    row = {
        "trip_distance":   trip_distance,
        "trip_duration":   trip_duration,
        "passenger_count": passenger_count,
        "pickup_hour":     pickup_hour,
        "is_weekend":      is_weekend,
        "avg_speed_kmh":   avg_speed_kmh,
        "is_peak_hour":    is_peak_hour,
        "is_night_ride":   is_night_ride,
        "duration_per_km": duration_per_km,
        "distance_bucket": distance_bucket,
        "hour_period":     hour_period,
    }

    # One-hot encode traffic
    for tc in ["Low", "Medium", "High", "Very High"]:
        col = f"traffic_condition_{tc}"
        row[col] = 1 if traffic_condition == tc else 0

    # One-hot encode weather
    for wc in ["Clear", "Cloudy", "Foggy", "Rainy", "Snowy"]:
        col = f"weather_condition_{wc}"
        row[col] = 1 if weather_condition == wc else 0

    df_input = pd.DataFrame([row])

    # Align with training columns (add missing cols as 0, drop extra)
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_columns]

    return df_input


# ─────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────
def predict_fare(
    trip_distance:     float,
    trip_duration:     float,
    passenger_count:   int   = 1,
    pickup_hour:       int   = 12,
    is_weekend:        int   = 0,
    traffic_condition: str   = "Medium",
    weather_condition: str   = "Clear",
    models_dir:        str   = "models"
) -> dict:
    """
    Returns a dictionary with:
        - predicted_fare      (float)
        - fare_range_low      (float)
        - fare_range_high     (float)
        - confidence_pct      (int)
        - estimated_duration  (float)
    """
    model, scaler, columns, metrics = load_model_artifacts(models_dir)
    best_model_name = metrics.get("_best_model", "Best Model")
    best_rmse = metrics.get(best_model_name, {}).get("RMSE", 2.0)
    best_r2   = metrics.get(best_model_name, {}).get("R2", 0.95)

    df_input = build_feature_vector(
        trip_distance, trip_duration, passenger_count,
        pickup_hour, is_weekend, traffic_condition, weather_condition, columns
    )

    # Scale numeric features
    num_cols = [c for c in ["trip_distance", "trip_duration", "avg_speed_kmh",
                             "duration_per_km", "passenger_count"]
                if c in df_input.columns]
    df_scaled = df_input.copy()
    df_scaled[num_cols] = scaler.transform(df_input[num_cols])

    predicted_fare = float(model.predict(df_scaled)[0])
    predicted_fare = max(3.0, round(predicted_fare, 2))

    # Confidence interval: ±1.5 * RMSE (≈ 87% CI)
    margin = round(1.5 * best_rmse, 2)
    fare_low  = round(max(3.0, predicted_fare - margin), 2)
    fare_high = round(predicted_fare + margin, 2)

    # Confidence % from R²
    confidence_pct = min(99, max(50, int(best_r2 * 100)))

    return {
        "predicted_fare":     predicted_fare,
        "fare_range_low":     fare_low,
        "fare_range_high":    fare_high,
        "confidence_pct":     confidence_pct,
        "estimated_duration": round(trip_duration, 1),
        "model_used":         best_model_name,
        "r2_score":           best_r2,
    }


# ─────────────────────────────────────────────
#  CLI demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    result = predict_fare(
        trip_distance=12.5,
        trip_duration=28.0,
        passenger_count=2,
        pickup_hour=8,
        is_weekend=0,
        traffic_condition="High",
        weather_condition="Rainy"
    )
    print("\n" + "="*45)
    print("  SmartFareAI – Fare Prediction")
    print("="*45)
    print(f"  Estimated Fare    : ${result['predicted_fare']:.2f}")
    print(f"  Fare Range        : ${result['fare_range_low']:.2f} – ${result['fare_range_high']:.2f}")
    print(f"  Confidence        : {result['confidence_pct']}%")
    print(f"  Est. Duration     : {result['estimated_duration']} min")
    print(f"  Model Used        : {result['model_used']} (R²={result['r2_score']:.4f})")
    print("="*45)
