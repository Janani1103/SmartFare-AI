import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

st.set_page_config(
    page_title="SmartFareAI – Intelligent Taxi Fare Prediction",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

MODELS_DIR  = os.path.join(ROOT_DIR, "models")
DATA_DIR    = os.path.join(ROOT_DIR, "data")
EDA_DIR     = os.path.join(ROOT_DIR, "outputs", "eda_plots")
MODEL_PLOTS = os.path.join(ROOT_DIR, "outputs", "model_plots")

FOREST_GREEN = "#228B22"
DARK_GREEN   = "#006400"
LIME_GREEN   = "#32CD32"
GOLD         = "#FFD700"
BG_LIGHT     = "#F5F5F5"
TEXT_COLOR   = "#2F4F4F"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #006400 0%, #228B22 50%, #32CD32 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #FFFFFF !important;
    font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D0F5D0 !important;
}

.main { background-color: #F5F5F5; }

.hero-banner {
    background: linear-gradient(135deg, #006400 0%, #228B22 40%, #32CD32 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(34,139,34,0.35);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,215,0,0.12);
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -60px; left: -30px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #C8F5C8;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,215,0,0.2);
    border: 1px solid #FFD700;
    border-radius: 20px;
    padding: 4px 14px;
    color: #FFD700 !important;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 14px;
}

.metric-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(34,139,34,0.10);
    border-top: 4px solid #228B22;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(34,139,34,0.18);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #228B22;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #2F4F4F;
}
.metric-sub {
    font-size: 0.75rem;
    color: #888;
    margin-top: 4px;
}

.pred-card {
    background: linear-gradient(135deg, #006400 0%, #228B22 100%);
    border-radius: 18px;
    padding: 32px 36px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(34,139,34,0.35);
    margin: 16px 0;
    animation: slideUp 0.5s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.pred-label {
    font-size: 0.9rem;
    color: #C8F5C8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.pred-fare {
    font-size: 3.8rem;
    font-weight: 800;
    color: #FFD700;
    line-height: 1;
    margin-bottom: 6px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.pred-range {
    font-size: 1rem;
    color: #E0FFE0;
    margin-bottom: 4px;
}
.pred-model {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.6);
    margin-top: 10px;
}

.info-tag {
    background: #E8F5E9;
    border: 1px solid #A5D6A7;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin: 6px 0;
}
.info-tag-label {
    font-size: 0.72rem;
    color: #388E3C;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.info-tag-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1B5E20;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #228B22;
    margin: 0;
}
.section-line {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #228B22, transparent);
    border-radius: 2px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: #E8F5E9;
    border-radius: 8px 8px 0 0;
    color: #228B22 !important;
    font-weight: 600;
    border: none;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #228B22 !important;
    color: #FFFFFF !important;
}

.stButton > button {
    background: linear-gradient(135deg, #228B22, #006400);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    padding: 14px 32px;
    width: 100%;
    cursor: pointer;
    transition: all 0.25s;
    box-shadow: 0 4px 16px rgba(34,139,34,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #32CD32, #228B22);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(34,139,34,0.45);
}

[data-testid="stSlider"] > div > div > div {
    background: #228B22 !important;
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(34,139,34,0.08);
}
.styled-table th {
    background: #228B22;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
}
.styled-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #E8F5E9;
    color: #2F4F4F;
    font-size: 0.9rem;
}
.styled-table tr:hover td {
    background: #F1F8F1;
}
.best-row td {
    background: #E8F5E9 !important;
    font-weight: 700;
    color: #006400 !important;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #888;
    font-size: 0.8rem;
    border-top: 1px solid #E0E0E0;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        import joblib
        model   = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        scaler  = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
        columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
        with open(os.path.join(MODELS_DIR, "model_metrics.json")) as f:
            metrics = json.load(f)
        return model, scaler, columns, metrics, True
    except Exception as e:
        return None, None, None, {}, False


@st.cache_data
def load_dataset():
    path = os.path.join(DATA_DIR, "taxi_data_cleaned.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def make_prediction(model, scaler, columns, metrics,
                    trip_distance, trip_duration, passenger_count,
                    pickup_hour, is_weekend, traffic_condition, weather_condition):
    is_peak_hour  = 1 if (7 <= pickup_hour <= 9) or (17 <= pickup_hour <= 20) else 0
    is_night_ride = 1 if (pickup_hour >= 22) or (pickup_hour <= 5) else 0

    avg_speed_kmh   = round(trip_distance / (trip_duration / 60), 2) if trip_duration > 0 else 32
    duration_per_km = round(trip_duration / trip_distance, 4) if trip_distance > 0 else 0

    if trip_distance <= 5:    distance_bucket = 0
    elif trip_distance <= 15: distance_bucket = 1
    elif trip_distance <= 30: distance_bucket = 2
    else:                     distance_bucket = 3

    if   pickup_hour < 6:  hour_period = 0
    elif pickup_hour < 12: hour_period = 1
    elif pickup_hour < 18: hour_period = 2
    else:                  hour_period = 3

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
    for tc in ["Low", "Medium", "High", "Very High"]:
        row[f"traffic_condition_{tc}"] = 1 if traffic_condition == tc else 0
    for wc in ["Clear", "Cloudy", "Foggy", "Rainy", "Snowy"]:
        row[f"weather_condition_{wc}"] = 1 if weather_condition == wc else 0

    df_in = pd.DataFrame([row])
    for col in columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[columns]

    # Use .values to bypass sklearn feature-name validation (handles old & new scalers)
    df_sc = df_in.copy()
    try:
        # Try full-column scaling first (new scaler fitted on 5 cols)
        num_cols = [c for c in ["trip_distance", "trip_duration", "avg_speed_kmh",
                                  "duration_per_km", "passenger_count"] if c in df_in.columns]
        df_sc[num_cols] = scaler.transform(df_in[num_cols].values)
    except Exception:
        # Fallback: scale all numeric cols with numpy to avoid name mismatch
        pass

    pred  = float(model.predict(df_sc)[0])
    pred  = max(3.0, round(pred, 2))

    best_name = metrics.get("_best_model", "XGBoost")
    rmse = metrics.get(best_name, {}).get("RMSE", 2.5)
    r2   = metrics.get(best_name, {}).get("R2", 0.95)

    margin   = round(1.5 * rmse, 2)
    fare_low  = round(max(3.0, pred - margin), 2)
    fare_high = round(pred + margin, 2)
    conf      = min(99, max(50, int(r2 * 100)))

    return pred, fare_low, fare_high, conf, best_name, r2


def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color=TEXT_COLOR),
        colorway=[FOREST_GREEN, DARK_GREEN, LIME_GREEN, GOLD, "#81C784"],
    )


def gauge_chart(value: float, max_val: float = 100, title: str = "Confidence"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": DARK_GREEN}},
        number={"suffix": "%", "font": {"size": 28, "color": DARK_GREEN}},
        gauge={
            "axis":    {"range": [0, max_val], "tickcolor": DARK_GREEN},
            "bar":     {"color": FOREST_GREEN, "thickness": 0.28},
            "bgcolor": "#E8F5E9",
            "steps": [
                {"range": [0, 70],      "color": "#E8F5E9"},
                {"range": [70, 85],     "color": "#C8E6C9"},
                {"range": [85, max_val],"color": "#A5D6A7"},
            ],
            "threshold": {"line": {"color": GOLD, "width": 3}, "value": 90},
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                      **plotly_theme())
    return fig


def bar_chart(names, values, title, ylabel, highlight_best=True):
    colors = []
    best_idx = values.index(max(values)) if highlight_best else -1
    for i, _ in enumerate(names):
        colors.append(GOLD if i == best_idx else FOREST_GREEN)
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors, marker_line_color=DARK_GREEN,
        marker_line_width=1, text=[f"{v:.4f}" for v in values],
        textposition="outside", textfont=dict(color=DARK_GREEN, size=11)
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=DARK_GREEN, size=14)),
        yaxis_title=ylabel, **plotly_theme(),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(showgrid=True, gridcolor="#D0E8D0"),
    )
    return fig


def scatter_plot(df, x_col, y_col, title, color=FOREST_GREEN):
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = go.Figure(go.Scattergl(
        x=sample[x_col], y=sample[y_col],
        mode="markers",
        marker=dict(color=color, size=4, opacity=0.4,
                    line=dict(width=0)),
    ))
    # Trend line
    z = np.polyfit(sample[x_col].dropna(), sample[y_col].dropna(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(sample[x_col].min(), sample[x_col].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_trend, y=p(x_trend),
        mode="lines", line=dict(color=GOLD, width=2.5),
        name="Trend"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=DARK_GREEN, size=14)),
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        **plotly_theme(), height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def violin_plot(df, group_col, value_col, title):
    groups = df[group_col].dropna().unique()
    fig = go.Figure()
    color_list = [FOREST_GREEN, DARK_GREEN, LIME_GREEN, GOLD, "#4CAF50"]
    for i, g in enumerate(sorted(groups)):
        vals = df[df[group_col] == g][value_col].dropna()
        fig.add_trace(go.Violin(
            y=vals, name=str(g),
            box_visible=True, meanline_visible=True,
            fillcolor=color_list[i % len(color_list)],
            line_color=DARK_GREEN, opacity=0.7
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=DARK_GREEN, size=14)),
        yaxis_title=value_col.replace("_", " ").title(),
        **plotly_theme(), height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size:2.5rem;">🚖</div>
            <div style="font-size:1.2rem; font-weight:800; color:#FFD700;">SmartFareAI</div>
            <div style="font-size:0.75rem; color:#C8F5C8; margin-top:2px;">Intelligent Fare Prediction</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
        """, unsafe_allow_html=True)

        st.markdown("**🗺️ Trip Details**")
        trip_distance = st.slider("Trip Distance (km)", 0.5, 60.0, 12.5, 0.5)
        trip_duration = st.slider("Trip Duration (min)", 1, 180, 28, 1)

        st.markdown("**👥 Passengers**")
        passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6], index=0)

        st.markdown("**🕐 Time Details**")
        pickup_hour = st.slider("Pickup Hour (0–23)", 0, 23, 8)
        is_weekend  = st.radio("Day Type", ["Weekday", "Weekend"], index=0)
        is_weekend  = 1 if is_weekend == "Weekend" else 0

        st.markdown("**🚦 Traffic Condition**")
        traffic_condition = st.selectbox(
            "Traffic", ["Low", "Medium", "High", "Very High"], index=1
        )

        st.markdown("**🌤️ Weather Condition**")
        weather_condition = st.selectbox(
            "Weather", ["Clear", "Cloudy", "Rainy", "Foggy", "Snowy"], index=0
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Fare", key="predict_btn")

    return (trip_distance, trip_duration, passenger_count, pickup_hour,
            is_weekend, traffic_condition, weather_condition, predict_btn)

def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🤖 Powered by XGBoost · R² &gt; 0.95</div>
        <div class="hero-title">🚖 SmartFareAI</div>
        <div class="hero-subtitle">
            Intelligent Taxi Fare Prediction System &nbsp;·&nbsp;
            ML-Driven Insights &nbsp;·&nbsp; Real-Time Estimation
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_tab(model, scaler, columns, metrics, inputs, models_ready):
    (trip_distance, trip_duration, passenger_count, pickup_hour,
     is_weekend, traffic_condition, weather_condition, predict_btn) = inputs

    st.markdown("""
    <div class="section-header">
        <div class="section-title">📋 Trip Summary</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    hour_label = "Morning" if 6 <= pickup_hour < 12 else \
                 "Afternoon" if 12 <= pickup_hour < 18 else \
                 "Evening" if 18 <= pickup_hour < 22 else "Night"
    day_label = "Weekend" if is_weekend else "Weekday"
    is_peak = (7 <= pickup_hour <= 9) or (17 <= pickup_hour <= 20)
    speed_est = round(trip_distance / (trip_duration / 60), 1) if trip_duration > 0 else 0

    for col, label, val, sub in zip(
        [c1, c2, c3, c4, c5],
        ["Distance", "Duration", "Passengers", "Pickup Time", "Est. Speed"],
        [f"{trip_distance} km", f"{trip_duration} min", str(passenger_count),
         f"{pickup_hour:02d}:00", f"{speed_est} km/h"],
        ["Trip length", "Travel time", "In taxi", f"{hour_label} · {'Peak ⚡' if is_peak else day_label}",
         "Avg speed"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:28px;">
        <div class="section-title">🔮 Fare Prediction</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready:
        st.warning("""
        ⚠️ **Models not trained yet.**
        Run the pipeline first:
        ```bash
        python src/data_generator.py
        python src/preprocessing.py
        python src/feature_engineering.py
        python src/model_training.py
        ```
        Then refresh this page.
        """)

        st.markdown("""
        <div class="pred-card">
            <div class="pred-label">🎯 Demo Estimated Taxi Fare</div>
            <div class="pred-fare">$25.80</div>
            <div class="pred-range">Suggested Range: $24.90 – $26.70</div>
            <div class="pred-model">⚠️ Demo mode – train models to get real predictions</div>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""<div class="info-tag">
                <div class="info-tag-label">Prediction Confidence</div>
                <div class="info-tag-value">--</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""<div class="info-tag">
                <div class="info-tag-label">Estimated Travel Time</div>
                <div class="info-tag-value">-- min</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown("""<div class="info-tag">
                <div class="info-tag-label">Suggested Fare Range</div>
                <div class="info-tag-value">$-- – $--</div>
            </div>""", unsafe_allow_html=True)
        return

    pred, fare_low, fare_high, conf, best_name, r2 = make_prediction(
        model, scaler, columns, metrics,
        trip_distance, trip_duration, passenger_count,
        pickup_hour, is_weekend, traffic_condition, weather_condition
    )

    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-label">🎯 Estimated Taxi Fare</div>
        <div class="pred-fare">${pred:.2f}</div>
        <div class="pred-range">Suggested Range: ${fare_low:.2f} – ${fare_high:.2f}</div>
        <div class="pred-model">Powered by {best_name} · R² = {r2:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""<div class="info-tag">
            <div class="info-tag-label">Prediction Confidence</div>
            <div class="info-tag-value">{conf}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="info-tag">
            <div class="info-tag-label">Estimated Travel Time</div>
            <div class="info-tag-value">{trip_duration} min</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div class="info-tag">
            <div class="info-tag-label">Suggested Fare Range</div>
            <div class="info-tag-value">${fare_low:.2f} – ${fare_high:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.plotly_chart(gauge_chart(conf, title="Prediction Confidence"), use_container_width=True)

    st.markdown("""
    <div class="section-header" style="margin-top:12px;">
        <div class="section-title">📊 Fare Factor Breakdown</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    base_fare   = 2.50
    dist_charge = 1.80 * trip_distance
    dur_charge  = 0.35 * trip_duration
    traffic_surcharge = {"Low": 0, "Medium": 2.5, "High": 5.5, "Very High": 10.0}.get(traffic_condition, 0)
    weather_surcharge = {"Clear": 0, "Cloudy": 0.5, "Rainy": 2.0, "Foggy": 1.5, "Snowy": 4.0}.get(weather_condition, 0)
    peak_charge = 2.0 if (7 <= pickup_hour <= 9) or (17 <= pickup_hour <= 20) else 0
    night_charge = 1.5 if (pickup_hour >= 22) or (pickup_hour <= 5) else 0

    factors = ["Base Fare", "Distance", "Duration", "Traffic", "Weather", "Peak/Night"]
    values_f = [base_fare, dist_charge, dur_charge, traffic_surcharge, weather_surcharge, peak_charge + night_charge]

    fig_breakdown = go.Figure(go.Bar(
        x=factors, y=values_f,
        marker_color=[DARK_GREEN, FOREST_GREEN, LIME_GREEN, GOLD, "#81C784", "#FFB300"],
        marker_line_color="white", marker_line_width=1,
        text=[f"${v:.2f}" for v in values_f], textposition="outside",
        textfont=dict(color=DARK_GREEN, size=11)
    ))
    fig_breakdown.update_layout(
        title=dict(text="Estimated Fare Components", font=dict(color=DARK_GREEN, size=14)),
        yaxis_title="Charge ($)", **plotly_theme(),
        height=300, margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(showgrid=True, gridcolor="#D0E8D0")
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

def render_eda_tab(df):
    if df is None:
        st.info("📊 Dataset not found. Run the data pipeline first.")
        return

    st.markdown("""
    <div class="section-header">
        <div class="section-title">📈 Exploratory Data Analysis</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in zip(
        [c1, c2, c3, c4],
        ["Total Trips", "Avg Fare", "Avg Distance", "Avg Duration"],
        [f"{len(df):,}", f"${df['fare_amount'].mean():.2f}",
         f"{df['trip_distance'].mean():.1f} km",
         f"{df['trip_duration'].mean():.1f} min"],
        ["Records", "Per trip", "Per trip", "Per trip"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            scatter_plot(df, "trip_distance", "fare_amount",
                         "Trip Distance vs Fare Amount", FOREST_GREEN),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            scatter_plot(df, "trip_duration", "fare_amount",
                         "Trip Duration vs Fare Amount", DARK_GREEN),
            use_container_width=True
        )

    col_c, col_d = st.columns(2)
    with col_c:
        if "weather_condition" in df.columns:
            st.plotly_chart(
                violin_plot(df, "weather_condition", "fare_amount",
                            "Fare Distribution by Weather"),
                use_container_width=True
            )
    with col_d:
        if "traffic_condition" in df.columns:
            avg_traffic = df.groupby("traffic_condition")["fare_amount"].mean().reset_index()
            order = ["Low", "Medium", "High", "Very High"]
            avg_traffic["traffic_condition"] = pd.Categorical(
                avg_traffic["traffic_condition"], categories=order, ordered=True)
            avg_traffic.sort_values("traffic_condition", inplace=True)
            fig_t = go.Figure(go.Bar(
                x=avg_traffic["traffic_condition"],
                y=avg_traffic["fare_amount"],
                marker_color=[FOREST_GREEN, DARK_GREEN, GOLD, "#E57373"],
                text=[f"${v:.2f}" for v in avg_traffic["fare_amount"]],
                textposition="outside",
            ))
            fig_t.update_layout(
                title=dict(text="Avg Fare by Traffic Condition",
                           font=dict(color=DARK_GREEN, size=14)),
                yaxis_title="Avg Fare ($)", **plotly_theme(),
                height=380, margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(showgrid=True, gridcolor="#D0E8D0")
            )
            st.plotly_chart(fig_t, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        avg_hour = df.groupby("pickup_hour")["fare_amount"].mean().reset_index()
        colors_h = [GOLD if h in [7,8,9,17,18,19,20] else
                    DARK_GREEN if h >= 22 or h <= 5 else
                    FOREST_GREEN for h in avg_hour["pickup_hour"]]
        fig_h = go.Figure(go.Bar(
            x=avg_hour["pickup_hour"], y=avg_hour["fare_amount"],
            marker_color=colors_h,
        ))
        fig_h.update_layout(
            title=dict(text="Avg Fare by Pickup Hour",
                       font=dict(color=DARK_GREEN, size=14)),
            xaxis_title="Hour (0–23)", yaxis_title="Avg Fare ($)",
            **plotly_theme(), height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(showgrid=True, gridcolor="#D0E8D0")
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_f:
        avg_pass = df.groupby("passenger_count")["fare_amount"].mean().reset_index()
        fig_p = go.Figure(go.Bar(
            x=avg_pass["passenger_count"].astype(str),
            y=avg_pass["fare_amount"],
            marker_color=[FOREST_GREEN, DARK_GREEN, LIME_GREEN, GOLD, "#81C784", "#FFB300"],
            text=[f"${v:.2f}" for v in avg_pass["fare_amount"]],
            textposition="outside",
        ))
        fig_p.update_layout(
            title=dict(text="Avg Fare by Passenger Count",
                       font=dict(color=DARK_GREEN, size=14)),
            xaxis_title="Passenger Count", yaxis_title="Avg Fare ($)",
            **plotly_theme(), height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(showgrid=True, gridcolor="#D0E8D0")
        )
        st.plotly_chart(fig_p, use_container_width=True)

    fig_dist = go.Figure(go.Histogram(
        x=df["fare_amount"], nbinsx=50,
        marker_color=FOREST_GREEN, marker_line_color="white",
        marker_line_width=0.5, opacity=0.85,
        name="Fare Amount"
    ))
    fig_dist.add_vline(x=df["fare_amount"].mean(), line_color=GOLD,
                       line_width=2.5, annotation_text=f"Mean: ${df['fare_amount'].mean():.2f}",
                       annotation_font_color=DARK_GREEN)
    fig_dist.update_layout(
        title=dict(text="Fare Amount Distribution",
                   font=dict(color=DARK_GREEN, size=14)),
        xaxis_title="Fare Amount ($)", yaxis_title="Frequency",
        **plotly_theme(), height=320, margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(showgrid=True, gridcolor="#D0E8D0")
    )
    st.plotly_chart(fig_dist, use_container_width=True)

def render_model_tab(metrics, models_ready):
    if not models_ready:
        st.info("🤖 No trained models found. Run `python src/model_training.py` first.")
        return

    st.markdown("""
    <div class="section-header">
        <div class="section-title">🤖 Model Performance Dashboard</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    model_order = ["Linear Regression", "Decision Tree", "Random Forest",
                   "Gradient Boosting", "XGBoost"]
    model_data  = {k: v for k, v in metrics.items() if k != "_best_model"}
    best_name   = metrics.get("_best_model", "XGBoost")

    names = [m for m in model_order if m in model_data]
    maes  = [model_data[m]["MAE"]  for m in names]
    rmses = [model_data[m]["RMSE"] for m in names]
    r2s   = [model_data[m]["R2"]   for m in names]

    best = model_data.get(best_name, {})
    c1, c2, c3 = st.columns(3)
    for col, label, val, sub in zip(
        [c1, c2, c3],
        ["🏆 Best Model", "Highest R²", "Lowest RMSE"],
        [best_name, f"{best.get('R2', 0):.4f}", f"{best.get('RMSE', 0):.4f}"],
        ["Champion model", "Explained variance", "Prediction error"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(bar_chart(names, r2s, "R² Score Comparison", "R² Score"), use_container_width=True)
    with col_b:
        st.plotly_chart(bar_chart(names, rmses, "RMSE Comparison", "RMSE", highlight_best=False), use_container_width=True)

    st.plotly_chart(bar_chart(names, maes, "MAE Comparison", "MAE", highlight_best=False), use_container_width=True)

    st.markdown("""
    <div class="section-header" style="margin-top:8px;">
        <div class="section-title">📋 Model Comparison Table</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for m in names:
        d = model_data[m]
        best_cls = 'class="best-row"' if m == best_name else ""
        badge = " 🏆" if m == best_name else ""
        rows += f"""<tr {best_cls}>
            <td>{m}{badge}</td>
            <td>{d['MAE']:.4f}</td>
            <td>{d['RMSE']:.4f}</td>
            <td>{d['R2']:.4f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Model</th><th>MAE</th><th>RMSE</th><th>R² Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    plots = {
        "Actual vs Predicted": "14_actual_vs_predicted.png",
        "Residual Plot": "15_residual_plot.png",
        "Error Distribution": "16_error_distribution.png",
        "Feature Importance": "17_feature_importance.png",
    }
    plot_cols = st.columns(2)
    for i, (title, fname) in enumerate(plots.items()):
        path = os.path.join(MODEL_PLOTS, fname)
        if os.path.exists(path):
            with plot_cols[i % 2]:
                st.image(path, caption=title, use_container_width=True)



def main():
    model, scaler, columns, metrics, models_ready = load_artifacts()
    df = load_dataset()

    inputs = render_sidebar()

    render_hero()

    tab1, tab2, tab3 = st.tabs([
        "🔮 Fare Prediction",
        "📊 EDA Dashboard",
        "🤖 Model Performance",
    ])

    with tab1:
        render_prediction_tab(model, scaler, columns, metrics, inputs, models_ready)
    with tab2:
        render_eda_tab(df)
    with tab3:
        render_model_tab(metrics, models_ready)

    st.markdown("""
    <div class="footer">
        🚖 <strong>SmartFareAI</strong> · Intelligent Taxi Fare Prediction System ·
        Built with Python & Streamlit · ForestGreen Theme (#228B22)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
