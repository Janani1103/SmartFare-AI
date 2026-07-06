"""
SmartFareAI – Exploratory Data Analysis (Phase 2)
Generates ForestGreen-themed charts and answers business questions.
All plots saved to outputs/eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

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

GREEN_PALETTE = [FOREST_GREEN, DARK_GREEN, LIME_GREEN, "#4CAF50",
                 "#81C784", "#A5D6A7", "#C8E6C9", GOLD]

def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   WHITE,
        "axes.edgecolor":   DARK_GREEN,
        "axes.labelcolor":  TEXT_COLOR,
        "axes.titlecolor":  DARK_GREEN,
        "axes.titlesize":   14,
        "axes.labelsize":   11,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "grid.color":       "#D0E8D0",
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
        "text.color":       TEXT_COLOR,
    })

def save_fig(name: str, outdir: str = "outputs/eda_plots"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
#  UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────
def plot_univariate(df: pd.DataFrame):
    print("\n  [Univariate Analysis]")
    apply_theme()

    numeric_cols = ["fare_amount", "trip_distance", "trip_duration", "passenger_count"]
    cat_cols     = ["weather_condition", "traffic_condition"]

    # --- Histograms ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Fare, Distance, Duration & Passenger Distribution",
                 fontsize=16, color=DARK_GREEN, fontweight="bold", y=1.01)
    for ax, col in zip(axes.flatten(), numeric_cols):
        ax.hist(df[col].dropna(), bins=40, color=FOREST_GREEN, edgecolor=WHITE, alpha=0.85)
        ax.set_title(col.replace("_", " ").title(), color=DARK_GREEN, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.4)
        mean_val = df[col].mean()
        ax.axvline(mean_val, color=GOLD, linestyle="--", linewidth=1.8,
                   label=f"Mean: {mean_val:.1f}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig("01_histograms.png")

    # --- Box Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Outlier Detection – Fare & Distance Box Plots",
                 fontsize=15, color=DARK_GREEN, fontweight="bold")
    for ax, col in zip(axes, ["fare_amount", "trip_distance"]):
        bp = ax.boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=LIME_GREEN, color=DARK_GREEN),
                        medianprops=dict(color=GOLD, linewidth=2.5),
                        whiskerprops=dict(color=DARK_GREEN),
                        capprops=dict(color=DARK_GREEN),
                        flierprops=dict(marker="o", markerfacecolor=GOLD,
                                        markersize=4, alpha=0.5))
        ax.set_title(col.replace("_", " ").title(), color=DARK_GREEN, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_fig("02_boxplots.png")

    # --- Count Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trip Frequency by Weather & Traffic Condition",
                 fontsize=15, color=DARK_GREEN, fontweight="bold")
    for ax, col in zip(axes, cat_cols):
        counts = df[col].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=GREEN_PALETTE[:len(counts)], edgecolor=WHITE, linewidth=0.8)
        ax.set_title(col.replace("_", " ").title(), color=DARK_GREEN, fontweight="bold")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.4)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{v:,}", ha="center", fontsize=9, color=DARK_GREEN)
    plt.tight_layout()
    save_fig("03_count_plots.png")


# ─────────────────────────────────────────────
#  BIVARIATE ANALYSIS
# ─────────────────────────────────────────────
def plot_bivariate(df: pd.DataFrame):
    print("\n  [Bivariate Analysis]")
    apply_theme()

    # --- Scatter: Distance vs Fare ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["trip_distance"], df["fare_amount"],
                    alpha=0.25, c=FOREST_GREEN, s=8, edgecolors="none")
    m, b = np.polyfit(df["trip_distance"].dropna(), df["fare_amount"].dropna(), 1)
    x_line = np.linspace(df["trip_distance"].min(), df["trip_distance"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=GOLD, linewidth=2.2, label=f"Trend (slope={m:.2f})")
    ax.set_title("Trip Distance vs Fare Amount", color=DARK_GREEN, fontweight="bold", fontsize=15)
    ax.set_xlabel("Trip Distance (km)")
    ax.set_ylabel("Fare Amount ($)")
    ax.legend()
    ax.grid(alpha=0.35)
    plt.tight_layout()
    save_fig("04_distance_vs_fare.png")

    # --- Scatter: Duration vs Fare ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["trip_duration"], df["fare_amount"],
               alpha=0.25, c=DARK_GREEN, s=8, edgecolors="none")
    m2, b2 = np.polyfit(df["trip_duration"].dropna(), df["fare_amount"].dropna(), 1)
    x2 = np.linspace(df["trip_duration"].min(), df["trip_duration"].max(), 100)
    ax.plot(x2, m2 * x2 + b2, color=GOLD, linewidth=2.2, label=f"Trend (slope={m2:.2f})")
    ax.set_title("Trip Duration vs Fare Amount", color=DARK_GREEN, fontweight="bold", fontsize=15)
    ax.set_xlabel("Trip Duration (minutes)")
    ax.set_ylabel("Fare Amount ($)")
    ax.legend()
    ax.grid(alpha=0.35)
    plt.tight_layout()
    save_fig("05_duration_vs_fare.png")

    # --- Bar: Avg Fare by Traffic ---
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Low", "Medium", "High", "Very High"]
    order = [o for o in order if o in df["traffic_condition"].unique()]
    avg_fare = df.groupby("traffic_condition")["fare_amount"].mean().reindex(order)
    bars = ax.bar(avg_fare.index, avg_fare.values,
                  color=GREEN_PALETTE[:len(avg_fare)], edgecolor=WHITE, linewidth=0.8)
    ax.set_title("Average Fare by Traffic Condition", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Traffic Condition")
    ax.set_ylabel("Average Fare ($)")
    ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, avg_fare.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"${v:.2f}", ha="center", fontsize=10, color=DARK_GREEN, fontweight="bold")
    plt.tight_layout()
    save_fig("06_fare_by_traffic.png")

    # --- Violin: Fare by Weather ---
    fig, ax = plt.subplots(figsize=(11, 6))
    weather_order = [w for w in ["Clear", "Cloudy", "Rainy", "Foggy", "Snowy"]
                     if w in df["weather_condition"].unique()]
    parts = ax.violinplot(
        [df[df["weather_condition"] == w]["fare_amount"].dropna().values for w in weather_order],
        showmedians=True
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(GREEN_PALETTE[i % len(GREEN_PALETTE)])
        pc.set_edgecolor(DARK_GREEN)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color(GOLD)
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(weather_order) + 1))
    ax.set_xticklabels(weather_order)
    ax.set_title("Fare Distribution by Weather Condition", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Fare Amount ($)")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_fig("07_fare_by_weather.png")

    # --- Bar: Avg Fare by Passenger Count ---
    fig, ax = plt.subplots(figsize=(9, 5))
    avg_pass = df.groupby("passenger_count")["fare_amount"].mean()
    bars = ax.bar(avg_pass.index.astype(str), avg_pass.values,
                  color=GREEN_PALETTE[:len(avg_pass)], edgecolor=WHITE, linewidth=0.8)
    ax.set_title("Average Fare by Passenger Count", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Passenger Count")
    ax.set_ylabel("Average Fare ($)")
    ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, avg_pass.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"${v:.2f}", ha="center", fontsize=9, color=DARK_GREEN)
    plt.tight_layout()
    save_fig("08_fare_by_passenger.png")

    # --- Bar: Avg Fare by Hour ---
    fig, ax = plt.subplots(figsize=(13, 5))
    avg_hour = df.groupby("pickup_hour")["fare_amount"].mean()
    colors_hour = [GOLD if h in [7,8,9,17,18,19,20] else
                   DARK_GREEN if h in [22,23,0,1,2,3,4,5] else
                   FOREST_GREEN for h in avg_hour.index]
    bars = ax.bar(avg_hour.index, avg_hour.values, color=colors_hour, edgecolor=WHITE, linewidth=0.5)
    ax.set_title("Average Fare by Pickup Hour", color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Pickup Hour (0–23)")
    ax.set_ylabel("Average Fare ($)")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.4)
    legend_patches = [
        mpatches.Patch(color=GOLD,         label="Peak Hours"),
        mpatches.Patch(color=DARK_GREEN,   label="Night Hours"),
        mpatches.Patch(color=FOREST_GREEN, label="Regular Hours"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()
    save_fig("09_fare_by_hour.png")


# ─────────────────────────────────────────────
#  MULTIVARIATE ANALYSIS – Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    print("\n  [Multivariate – Correlation Heatmap]")
    apply_theme()

    num_df = df[["fare_amount", "trip_distance", "trip_duration",
                 "passenger_count", "pickup_hour", "is_weekend"]].dropna()
    corr = num_df.corr()

    cmap = sns.diverging_palette(145, 10, as_cmap=True)   # green ↔ red
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.6, linecolor=BG_COLOR,
                ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 10, "color": TEXT_COLOR},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", color=DARK_GREEN, fontweight="bold", fontsize=15)
    plt.tight_layout()
    save_fig("10_correlation_heatmap.png")


# ─────────────────────────────────────────────
#  BUSINESS QUESTIONS SUMMARY CHART
# ─────────────────────────────────────────────
def plot_business_questions(df: pd.DataFrame):
    print("\n  [Business Questions]")
    apply_theme()

    correlations = {
        "Distance vs Fare":        df["trip_distance"].corr(df["fare_amount"]),
        "Duration vs Fare":        df["trip_duration"].corr(df["fare_amount"]),
        "Passengers vs Fare":      df["passenger_count"].corr(df["fare_amount"]),
        "Pickup Hour vs Fare":     df["pickup_hour"].corr(df["fare_amount"]),
        "Is Weekend vs Fare":      df["is_weekend"].corr(df["fare_amount"]),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    factors = list(correlations.keys())
    values  = list(correlations.values())
    colors  = [FOREST_GREEN if v >= 0 else "#E57373" for v in values]
    bars    = ax.barh(factors, values, color=colors, edgecolor=WHITE, linewidth=0.8)
    ax.axvline(0, color=DARK_GREEN, linewidth=1.2)
    ax.set_title("Correlation with Fare Amount (Business Q&A)",
                 color=DARK_GREEN, fontweight="bold", fontsize=14)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.grid(axis="x", alpha=0.4)
    for bar, v in zip(bars, values):
        ax.text(v + 0.005 if v >= 0 else v - 0.005,
                bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10, color=DARK_GREEN, fontweight="bold")
    plt.tight_layout()
    save_fig("11_business_questions.png")


# ─────────────────────────────────────────────
#  Full EDA pipeline
# ─────────────────────────────────────────────
def run_eda(input_path: str = "data/taxi_data_cleaned.csv"):
    print(f"\n{'='*55}")
    print(f"  PHASE 2 – Exploratory Data Analysis")
    print(f"{'='*55}")

    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows")

    plot_univariate(df)
    plot_bivariate(df)
    plot_correlation_heatmap(df)
    plot_business_questions(df)

    print(f"\n{'='*55}")
    print(f"  EDA COMPLETE  →  11 charts saved to outputs/eda_plots/")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()
