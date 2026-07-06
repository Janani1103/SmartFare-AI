"""
SmartFareAI – Machine Learning Model Training (Phase 4)
Trains 5 regression models, applies hyperparameter tuning,
saves the best model and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import json
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  ⚠️  XGBoost not found. Install with: pip install xgboost")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Color palette
# ─────────────────────────────────────────────
FOREST_GREEN = "#228B22"
DARK_GREEN   = "#006400"
LIME_GREEN   = "#32CD32"
GOLD         = "#FFD700"
BG_COLOR     = "#F5F5F5"
TEXT_COLOR   = "#2F4F4F"
WHITE        = "#FFFFFF"

MODEL_COLORS = [FOREST_GREEN, DARK_GREEN, LIME_GREEN, "#4CAF50", GOLD]


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   WHITE,
        "axes.edgecolor":   DARK_GREEN,
        "axes.labelcolor":  TEXT_COLOR,
        "axes.titlecolor":  DARK_GREEN,
        "axes.titlesize":   13,
        "axes.labelsize":   10,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "grid.color":       "#D0E8D0",
        "font.family":      "DejaVu Sans",
        "text.color":       TEXT_COLOR,
    })


def save_fig(name: str, outdir: str = "outputs/model_plots"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✅ Saved: {path}")


# ─────────────────────────────────────────────
#  Load & Prepare Data
# ─────────────────────────────────────────────
def load_and_prepare(path: str = "data/taxi_data_features.csv"):
    df = pd.read_csv(path)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

    # Drop EDA-only derived columns that would cause leakage
    drop_cols = [c for c in ["fare_per_km"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Remove any remaining NaN
    df.dropna(inplace=True)

    target = "fare_amount"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scale numeric features (except target)
    scaler = StandardScaler()
    num_cols = [c for c in ["trip_distance", "trip_duration", "avg_speed_kmh",
                            "duration_per_km", "passenger_count"] if c in X_train.columns]
    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_s[num_cols]  = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, "models/feature_scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")

    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"  Features: {X.shape[1]}")
    return X_train_s, X_test_s, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
#  Evaluation Helper
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4),
            "y_pred": y_pred}


# ─────────────────────────────────────────────
#  Model 1 – Linear Regression
# ─────────────────────────────────────────────
def train_linear(X_train, y_train, X_test, y_test):
    print("\n  [Model 1] Linear Regression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────
#  Model 2 – Decision Tree
# ─────────────────────────────────────────────
def train_decision_tree(X_train, y_train, X_test, y_test):
    print("\n  [Model 2] Decision Tree Regressor")
    model = DecisionTreeRegressor(random_state=42, max_depth=12)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────
#  Model 3 – Random Forest (with tuning)
# ─────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n  [Model 3] Random Forest Regressor (RandomizedSearchCV)")
    param_dist = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(base, param_dist, n_iter=15, cv=3,
                                scoring="r2", random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"     Best params: {search.best_params_}")
    metrics = evaluate(model, X_test, y_test)
    print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────
#  Model 4 – Gradient Boosting
# ─────────────────────────────────────────────
def train_gradient_boosting(X_train, y_train, X_test, y_test):
    print("\n  [Model 4] Gradient Boosting Regressor")
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────
#  Model 5 – XGBoost (with tuning)
# ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n  [Model 5] XGBoost Regressor (RandomizedSearchCV) ⭐")
    if not XGBOOST_AVAILABLE:
        print("     Skipped – XGBoost not installed.")
        return None, None

    param_dist = {
        "n_estimators":    [200, 400, 600],
        "learning_rate":   [0.03, 0.05, 0.08, 0.10],
        "max_depth":       [4, 5, 6, 7],
        "subsample":       [0.7, 0.8, 0.9],
        "colsample_bytree":[0.7, 0.8, 1.0],
        "reg_alpha":       [0, 0.1, 0.5],
        "reg_lambda":      [1, 1.5, 2],
    }
    base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(base, param_dist, n_iter=20, cv=3,
                                scoring="r2", random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"     Best params: {search.best_params_}")
    metrics = evaluate(model, X_test, y_test)
    print(f"     MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────
#  Comparison Charts
# ─────────────────────────────────────────────
def plot_model_comparison(results: dict, y_test, feature_names):
    apply_theme()

    model_names = list(results.keys())
    maes  = [results[m]["MAE"]  for m in model_names]
    rmses = [results[m]["RMSE"] for m in model_names]
    r2s   = [results[m]["R2"]   for m in model_names]

    # --- R² Comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, r2s, color=MODEL_COLORS[:len(model_names)],
                  edgecolor=WHITE, linewidth=0.8)
    ax.set_title("Model Comparison – R² Score", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_ylabel("R² Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, color=GOLD, linestyle="--", linewidth=1.5, label="Target R² = 0.90")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=9, color=DARK_GREEN, fontweight="bold")
    plt.tight_layout()
    save_fig("12_r2_comparison.png")

    # --- RMSE + MAE Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, title in zip(axes, [rmses, maes], ["RMSE", "MAE"]):
        bars = ax.bar(model_names, vals, color=MODEL_COLORS[:len(model_names)],
                      edgecolor=WHITE, linewidth=0.8)
        ax.set_title(f"Model Comparison – {title}", color=DARK_GREEN, fontweight="bold")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", fontsize=9, color=DARK_GREEN)
    plt.tight_layout()
    save_fig("13_rmse_mae_comparison.png")

    # --- Actual vs Predicted (best model) ---
    best_name = max(results, key=lambda m: results[m]["R2"])
    y_pred = results[best_name]["y_pred"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test, y_pred, alpha=0.25, c=FOREST_GREEN, s=10, edgecolors="none")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=GOLD, linewidth=2, linestyle="--", label="Perfect Prediction")
    ax.set_title(f"Actual vs Predicted Fare – {best_name}", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Actual Fare ($)")
    ax.set_ylabel("Predicted Fare ($)")
    ax.legend()
    ax.grid(alpha=0.35)
    plt.tight_layout()
    save_fig("14_actual_vs_predicted.png")

    # --- Residual Plot ---
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, residuals, alpha=0.25, c=DARK_GREEN, s=8, edgecolors="none")
    ax.axhline(0, color=GOLD, linewidth=2, linestyle="--")
    ax.set_title(f"Residual Plot – {best_name}", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Predicted Fare ($)")
    ax.set_ylabel("Residuals ($)")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    save_fig("15_residual_plot.png")

    # --- Error Distribution ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=50, color=FOREST_GREEN, edgecolor=WHITE, alpha=0.85)
    ax.axvline(0, color=GOLD, linewidth=2, linestyle="--", label="Zero Error")
    ax.axvline(np.mean(residuals), color=DARK_GREEN, linewidth=1.8,
               linestyle="-.", label=f"Mean Error: {np.mean(residuals):.2f}")
    ax.set_title("Error Distribution", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Prediction Error ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_fig("16_error_distribution.png")


# ─────────────────────────────────────────────
#  Feature Importance Chart
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name: str):
    apply_theme()
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [GOLD if i == fi.index[-1] else FOREST_GREEN for i in fi.index]
    ax.barh(fi.index, fi.values, color=colors, edgecolor=WHITE, linewidth=0.6)
    ax.set_title(f"Top 15 Feature Importances – {model_name}",
                 color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    save_fig("17_feature_importance.png")


# ─────────────────────────────────────────────
#  Print Results Table
# ─────────────────────────────────────────────
def print_results_table(results: dict):
    print(f"\n{'='*65}")
    print(f"  MODEL EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'MAE':>8} {'RMSE':>8} {'R² Score':>10}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        marker = " ⭐" if m["R2"] == max(r["R2"] for r in results.values()) else ""
        print(f"  {name:<30} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>10.4f}{marker}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────
#  Full Training Pipeline
# ─────────────────────────────────────────────
def run_model_training():
    print(f"\n{'='*55}")
    print(f"  PHASE 4 – Machine Learning Model Training")
    print(f"{'='*55}")

    os.makedirs("models", exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    results = {}
    trained_models = {}

    # Train all models
    model, metrics = train_linear(X_train, y_train, X_test, y_test)
    results["Linear Regression"] = metrics
    trained_models["Linear Regression"] = model

    model, metrics = train_decision_tree(X_train, y_train, X_test, y_test)
    results["Decision Tree"] = metrics
    trained_models["Decision Tree"] = model

    model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results["Random Forest"] = metrics
    trained_models["Random Forest"] = model

    model, metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results["Gradient Boosting"] = metrics
    trained_models["Gradient Boosting"] = model

    model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    if model is not None:
        results["XGBoost"] = metrics
        trained_models["XGBoost"] = model

    # Print summary
    print_results_table(results)

    # Save best model
    best_name = max(results, key=lambda m: results[m]["R2"])
    best_model = trained_models[best_name]
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"  🏆 Best Model: {best_name} (R²={results[best_name]['R2']:.4f})")
    print(f"  ✅ Saved: models/best_model.pkl")

    # Save metrics to JSON
    metrics_to_save = {k: {mk: mv for mk, mv in v.items() if mk != "y_pred"}
                       for k, v in results.items()}
    metrics_to_save["_best_model"] = best_name
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  ✅ Saved: models/model_metrics.json")

    # Save all individual models
    for name, m in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(m, f"models/{fname}")

    # Generate charts
    plot_model_comparison(results, y_test, feature_names)
    plot_feature_importance(best_model, feature_names, best_name)

    return results, trained_models


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_model_training()
