"""
SmartFareAI – Data Preprocessing Pipeline (Phase 1)
Steps: Load → Explore → Missing Values → Duplicates → Outliers → Encode → Scale
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
#  Theme constants
# ─────────────────────────────────────────────
FOREST_GREEN = "#228B22"
DARK_GREEN   = "#006400"
LIME_GREEN   = "#32CD32"
GOLD         = "#FFD700"

# ─────────────────────────────────────────────
#  Step 1 – Load Dataset
# ─────────────────────────────────────────────
def load_data(path: str = "data/taxi_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"  STEP 1 – Dataset Loaded")
    print(f"{'='*55}")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns      : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
#  Step 2 – Explore Dataset
# ─────────────────────────────────────────────
def explore(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"  STEP 2 – Dataset Exploration")
    print(f"{'='*55}")
    print("\n[Data Types]")
    print(df.dtypes.to_string())
    print("\n[Summary Statistics]")
    print(df.describe(include="all").T.to_string())


# ─────────────────────────────────────────────
#  Step 3 – Missing Value Analysis & Imputation
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 3 – Missing Value Analysis")
    print(f"{'='*55}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if len(missing) else "  No missing values found.")

    # Numeric → median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            print(f"  Filled '{col}' with median = {med:.2f}")

    # Categorical → mode
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' with mode = '{mode_val}'")

    return df


# ─────────────────────────────────────────────
#  Step 4 – Duplicate Removal
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 4 – Duplicate Removal")
    print(f"{'='*55}")
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    print(f"  Removed {before - after:,} duplicates → {after:,} rows remain")
    return df


# ─────────────────────────────────────────────
#  Step 5 – Outlier Detection (IQR)
# ─────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame,
                    cols: list = ["fare_amount", "trip_distance", "trip_duration"]
                    ) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 5 – Outlier Detection (IQR)")
    print(f"{'='*55}")
    before = len(df)
    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"  {col:20s} → removed {outliers:4d} outliers "
              f"(range: {lower:.2f} – {upper:.2f})")
    df.reset_index(drop=True, inplace=True)
    print(f"  Rows after outlier removal: {len(df):,} (removed {before - len(df):,})")
    return df


# ─────────────────────────────────────────────
#  Step 6 – Categorical Encoding
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 6 – Categorical Encoding")
    print(f"{'='*55}")

    # One-hot encode traffic_condition and weather_condition
    df = pd.get_dummies(df, columns=["traffic_condition", "weather_condition"], drop_first=False)

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"  Columns after encoding: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
#  Step 7 – Feature Scaling
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   scale_cols: list = ["trip_distance", "trip_duration", "fare_amount"]
                   ) -> tuple[pd.DataFrame, StandardScaler]:
    print(f"\n{'='*55}")
    print(f"  STEP 7 – Feature Scaling (StandardScaler)")
    print(f"{'='*55}")
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"  Scaled columns: {scale_cols}")
    return df_scaled, scaler


# ─────────────────────────────────────────────
#  Full Pipeline
# ─────────────────────────────────────────────
def run_preprocessing(raw_path: str = "data/taxi_data.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_encoded  – encoded but NOT scaled (used for EDA & feature engineering)
        df_scaled   – fully scaled (ready for ML)
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data(raw_path)
    explore(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)

    # Save cleaned (pre-encoding) version
    df.to_csv("data/taxi_data_cleaned.csv", index=False)
    print(f"\n  [OK] Cleaned data saved: data/taxi_data_cleaned.csv")

    df_encoded = encode_categoricals(df.copy())
    df_encoded.to_csv("data/taxi_data_encoded.csv", index=False)
    print(f"  [OK] Encoded data saved: data/taxi_data_encoded.csv")

    scale_cols = [c for c in ["trip_distance", "trip_duration"] if c in df_encoded.columns]
    df_scaled, scaler = scale_features(df_encoded.copy(), scale_cols=scale_cols)
    df_scaled.to_csv("data/taxi_data_scaled.csv", index=False)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"  [OK] Scaled data saved: data/taxi_data_scaled.csv")
    print(f"  [OK] Scaler saved:      models/scaler.pkl")

    print(f"\n{'='*55}")
    print(f"  PREPROCESSING COMPLETE  ->  {len(df_scaled):,} clean rows")
    print(f"{'='*55}\n")

    return df_encoded, df_scaled


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_encoded, df_scaled = run_preprocessing()
