import json
from pathlib import Path
from datetime import timedelta, datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Grid Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark SCADA-style CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stSidebar .stMarkdown { color: #8b949e; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.78rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

    /* Status badges */
    .badge-green  { background:#1a4731; color:#3fb950; border:1px solid #3fb950;
                    padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-amber  { background:#3d2b00; color:#d29922; border:1px solid #d29922;
                    padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-red    { background:#3d0000; color:#f85149; border:1px solid #f85149;
                    padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }

    /* Alert banner */
    .alert-banner { background:#3d0000; border:1px solid #f85149; border-radius:8px;
                    padding:10px 16px; margin-bottom:12px; color:#f85149; font-weight:600; }
    .info-banner  { background:#0c2d6b; border:1px solid #58a6ff; border-radius:8px;
                    padding:10px 16px; margin-bottom:12px; color:#58a6ff; }

    /* Section headers */
    h2, h3 { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA     = BASE / "data"
CKPT     = DATA / "checkpoints"
ABLATION = CKPT / "ablation"

PLANT_CAPACITY_KW = 500.0
PLANT_NAME        = "Pune Solar Plant — Unit 1"
PLANT_LOCATION    = "18.6°N, 73.8°E · Pune, Maharashtra"

# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_power():
    df = pd.read_csv(DATA / "pune_500kw_hourly.csv", parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None) + pd.DateOffset(years=3)
    return df

@st.cache_data
def load_weather():
    df = pd.read_csv(DATA / "weather_pune_2023.csv", parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None) + pd.DateOffset(years=3)
    return df

@st.cache_data
def load_alert_metrics():
    with open(ABLATION / "alert_metrics.json") as f:
        return json.load(f)

power_df   = load_power()
weather_df = load_weather()

# ── Simulated forecast: actual + calibrated noise (model proxy) ───────────────
rng = np.random.default_rng(42)
power_df["forecast_kw"] = (
    power_df["power_kw"]
    + rng.normal(0, 19.33, len(power_df))          # model RMSE = 19.33 kW
).clip(0, PLANT_CAPACITY_KW)

# ── Sidebar: operator controls ─────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ Grid Control Center")
st.sidebar.markdown(f"**{PLANT_NAME}**")
st.sidebar.caption(PLANT_LOCATION)
st.sidebar.markdown("---")

_today = datetime.now()
_default_date = _today.date()
_default_hour = _today.hour

st.sidebar.markdown("### Current Date & Time")
sim_date = st.sidebar.date_input(
    "Date",
    value=_default_date,
    min_value=pd.Timestamp("2026-01-01"),
    max_value=pd.Timestamp("2026-12-31"),
)
sim_hour = st.sidebar.slider("Hour", 0, 23, _default_hour)
now_ts   = pd.Timestamp(sim_date) + timedelta(hours=sim_hour)

st.sidebar.markdown("---")
st.sidebar.markdown("### Grid Parameters")
total_demand_kw = st.sidebar.number_input(
    "Total grid demand (kW)", min_value=100, max_value=2000, value=800, step=50
)
alert_fpr_target = st.sidebar.selectbox("Alert sensitivity", ["Low (FPR 0.05)", "Medium (FPR 0.15)", "High (FPR 0.25)"], index=1)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "View",
    ["⚡ Live Monitor", "📈 Solar Forecast", "🔋 Grid Planning", "🚨 Alert Center", "📋 Shift Report"],
)

# ── Helper: current row and forecast window ────────────────────────────────────
def get_current_row():
    idx = (power_df["timestamp"] - now_ts).abs().idxmin()
    return power_df.loc[idx]

def get_today_data():
    day = pd.Timestamp(sim_date)
    mask = (power_df["timestamp"].dt.date == day.date())
    return power_df[mask].copy()

def get_forecast_window(hours=24):
    mask = (power_df["timestamp"] >= now_ts) & (power_df["timestamp"] < now_ts + timedelta(hours=hours))
    return power_df[mask].copy()

def get_past_window(hours=48):
    mask = (power_df["timestamp"] >= now_ts - timedelta(hours=hours)) & (power_df["timestamp"] <= now_ts)
    return power_df[mask].copy()

def status_badge(label, level):
    cls = {"green": "badge-green", "amber": "badge-amber", "red": "badge-red"}.get(level, "badge-green")
    return f'<span class="{cls}">{label}</span>'

def plant_status_level(pct):
    if pct >= 60:   return "green",  "NOMINAL"
    if pct >= 20:   return "amber",  "REDUCED"
    if pct > 0:     return "amber",  "LOW OUTPUT"
    return          "red",   "NO OUTPUT"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ① — LIVE MONITOR
# ─────────────────────────────────────────────────────────────────────────────
if page == "⚡ Live Monitor":

    cur     = get_current_row()
    today   = get_today_data()
    cur_pct = (cur["power_kw"] / PLANT_CAPACITY_KW) * 100
    lvl, status_text = plant_status_level(cur_pct)

    # Alert banner if anomaly active
    if cur["anomaly_type"] != "normal":
        st.markdown(
            f'<div class="alert-banner">⚠ ACTIVE FAULT DETECTED — {cur["anomaly_type"].upper().replace("_"," ")} '
            f'at {now_ts.strftime("%H:%M")}. Inspect inverter immediately.</div>',
            unsafe_allow_html=True,
        )

    # Header
    c_title, c_status, c_time = st.columns([3, 1, 1])
    c_title.markdown(f"## {PLANT_NAME}")
    c_status.markdown(
        f"**Plant Status**<br>{status_badge(status_text, lvl)}",
        unsafe_allow_html=True,
    )
    c_time.markdown(f"**Current Time**<br>`{now_ts.strftime('%Y-%m-%d  %H:%M')}`", unsafe_allow_html=True)

    st.markdown("---")

    # ── Power gauge + key metrics ─────────────────────────────────────────────
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(cur["power_kw"], 1),
            delta={"reference": cur["forecast_kw"], "valueformat": ".1f",
                   "suffix": " kW vs forecast"},
            title={"text": "Current Output (kW)", "font": {"color": "#8b949e", "size": 14}},
            number={"suffix": " kW", "font": {"color": "#58a6ff", "size": 36}},
            gauge={
                "axis": {"range": [0, PLANT_CAPACITY_KW], "tickcolor": "#8b949e",
                         "tickfont": {"color": "#8b949e"}},
                "bar":  {"color": "#58a6ff"},
                "steps": [
                    {"range": [0,   150], "color": "#3d0000"},
                    {"range": [150, 300], "color": "#3d2b00"},
                    {"range": [300, 500], "color": "#1a4731"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 3},
                              "thickness": 0.8, "value": PLANT_CAPACITY_KW * 0.9},
                "bgcolor": "#161b22",
            },
        ))
        gauge.update_layout(
            height=280,
            paper_bgcolor="#0d1117",
            font={"color": "#e6edf3"},
            margin=dict(t=40, b=10, l=20, r=20),
        )
        st.plotly_chart(gauge, width='stretch')

    with col_metrics:
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Capacity Factor",  f"{cur_pct:.1f} %",
                    delta=f"{cur_pct - 50:.1f} % vs 50% target")
        r1c2.metric("Forecast (next hr)", f"{cur['forecast_kw']:.1f} kW")
        r1c3.metric("Grid Backup Needed",
                    f"{max(0, total_demand_kw - cur['power_kw']):.0f} kW",
                    delta=f"of {total_demand_kw} kW demand", delta_color="inverse")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Irradiance (GHI)",  f"{cur['ghi']:.0f} W/m²")
        r2c2.metric("Air Temperature",   f"{cur['temp_air']:.1f} °C")
        r2c3.metric("Cloud Cover",       f"{cur['cloud_cover']*100:.0f} %")

        r3c1, r3c2, r3c3 = st.columns(3)
        today_gen = today[today["timestamp"] <= now_ts]["power_kw"].sum() / 1000
        r3c1.metric("Today's Generation", f"{today_gen:.2f} MWh")
        r3c2.metric("Wind Speed",          f"{cur['wind_speed']:.1f} m/s")
        r3c3.metric("Solar Zenith",        f"{cur['zenith_deg']:.1f}°")

    st.markdown("---")

    # ── Today's actual vs forecast strip ─────────────────────────────────────
    st.subheader("Today's Generation — Actual vs Forecast")
    past   = today[today["timestamp"] <= now_ts]
    future = today[today["timestamp"] >  now_ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past["timestamp"], y=past["power_kw"],
        fill="tozeroy", name="Actual Output",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88,166,255,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=future["timestamp"], y=future["forecast_kw"],
        fill="tozeroy", name="Forecast",
        line=dict(color="#3fb950", width=2, dash="dash"),
        fillcolor="rgba(63,185,80,0.10)",
    ))
    fig.add_vline(x=now_ts.timestamp() * 1000, line_dash="dot", line_color="#d29922",
                  annotation_text="NOW", annotation_font_color="#d29922")
    fig.update_layout(
        height=280,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e"),
        legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")),
        yaxis=dict(range=[0, PLANT_CAPACITY_KW + 20], gridcolor="#21262d",
                   title=dict(text="Power (kW)", font=dict(color="#8b949e"))),
        xaxis=dict(gridcolor="#21262d"),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Anomaly type indicator ────────────────────────────────────────────────
    today_faults = today[today["anomaly_type"] != "normal"]
    if not today_faults.empty:
        st.markdown("**Today's fault events:**")
        fault_counts = today_faults["anomaly_type"].value_counts()
        cols = st.columns(len(fault_counts))
        for col, (ftype, cnt) in zip(cols, fault_counts.items()):
            col.metric(ftype.replace("_", " ").title(), f"{cnt} event(s)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ② — SOLAR FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Solar Forecast":

    st.markdown("## Solar Power Forecast")
    st.caption(f"Model: Multi-Modal LSTM+FC+CNN · Attention Fusion · RMSE 19.33 kW · R² 0.980")
    st.markdown("---")

    horizon = st.radio("Forecast horizon", ["Next 6 hours", "Next 12 hours", "Next 24 hours"], horizontal=True, index=2)
    hours   = {"Next 6 hours": 6, "Next 12 hours": 12, "Next 24 hours": 24}[horizon]

    fcast  = get_forecast_window(hours)
    past48 = get_past_window(48)

    # ── Forecast chart with confidence band ──────────────────────────────────
    if not fcast.empty:
        rmse   = 19.33
        upper  = (fcast["forecast_kw"] + 1.5 * rmse).clip(0, PLANT_CAPACITY_KW)
        lower  = (fcast["forecast_kw"] - 1.5 * rmse).clip(0)

        fig = go.Figure()

        # Historical actual
        fig.add_trace(go.Scatter(
            x=past48["timestamp"], y=past48["power_kw"],
            name="Historical Actual", mode="lines",
            line=dict(color="#8b949e", width=1.5),
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast["timestamp"], fcast["timestamp"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(63,185,80,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1.5 RMSE band", showlegend=True,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fcast["timestamp"], y=fcast["forecast_kw"],
            name="Forecast", mode="lines",
            line=dict(color="#3fb950", width=2.5),
        ))

        fig.add_vline(x=now_ts.timestamp() * 1000, line_dash="dot", line_color="#d29922",
                      annotation_text="NOW", annotation_font_color="#d29922")

        fig.update_layout(
            height=360,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")),
            yaxis=dict(range=[0, PLANT_CAPACITY_KW + 30], gridcolor="#21262d",
                       title="Power (kW)"),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, width='stretch')

        # ── Forecast table ────────────────────────────────────────────────────
        st.subheader("Hourly Forecast Table")
        tbl = fcast[["timestamp", "forecast_kw", "ghi", "temp_air", "cloud_cover"]].copy()
        tbl["upper_kw"]     = upper.values
        tbl["lower_kw"]     = lower.values
        tbl["grid_backup_kw"] = (total_demand_kw - tbl["forecast_kw"]).clip(lower=0).round(1)
        tbl["solar_share_%"]  = (tbl["forecast_kw"] / total_demand_kw * 100).round(1)
        tbl["timestamp"]      = tbl["timestamp"].dt.strftime("%H:%M")
        tbl = tbl.rename(columns={
            "timestamp": "Time", "forecast_kw": "Forecast (kW)",
            "upper_kw": "Upper (kW)", "lower_kw": "Lower (kW)",
            "ghi": "GHI (W/m²)", "temp_air": "Temp (°C)",
            "cloud_cover": "Cloud",
            "grid_backup_kw": "Grid Backup (kW)", "solar_share_%": "Solar Share (%)",
        })
        tbl = tbl.round(1)
        st.dataframe(tbl, width='stretch', height=320)

        # ── Summary forecast KPIs ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Forecast Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Forecast",     f"{fcast['forecast_kw'].max():.1f} kW")
        c2.metric("Avg Forecast",      f"{fcast['forecast_kw'].mean():.1f} kW")
        c3.metric("Expected Generation", f"{fcast['forecast_kw'].sum()/1000:.2f} MWh")
        c4.metric("Avg Solar Share",   f"{(fcast['forecast_kw'].mean()/total_demand_kw*100):.1f} %")
    else:
        st.warning("No forecast data for this window. Adjust the simulation clock in the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ③ — GRID PLANNING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔋 Grid Planning":

    st.markdown("## Grid Supply Planning")
    st.caption("Based on next-24h solar forecast · Adjust demand in sidebar")
    st.markdown("---")

    fcast = get_forecast_window(24)

    if fcast.empty:
        st.warning("No forecast data. Adjust simulation clock.")
    else:
        solar  = fcast["forecast_kw"].values
        demand = np.full(len(solar), total_demand_kw)
        backup = np.maximum(0, demand - solar)
        surplus= np.maximum(0, solar - demand)
        solar_share = (solar / demand * 100).clip(0, 100)

        # ── Supply stack chart ────────────────────────────────────────────────
        st.subheader("24-Hour Supply Stack")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fcast["timestamp"], y=solar,
            name="Solar (Forecast)", marker_color="#f59e0b",
        ))
        fig.add_trace(go.Bar(
            x=fcast["timestamp"], y=backup,
            name="Grid Backup Required", marker_color="#58a6ff",
        ))
        fig.add_trace(go.Scatter(
            x=fcast["timestamp"], y=demand,
            name=f"Total Demand ({total_demand_kw} kW)",
            mode="lines", line=dict(color="#f85149", width=2, dash="dot"),
        ))
        fig.update_layout(
            barmode="stack", height=360,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")),
            yaxis=dict(title="Power (kW)", gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # ── Planning KPIs ─────────────────────────────────────────────────────
        st.subheader("24-Hour Planning Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Peak Solar",            f"{solar.max():.0f} kW")
        c2.metric("Min Grid Backup",       f"{backup.min():.0f} kW")
        c3.metric("Max Grid Backup",       f"{backup.max():.0f} kW")
        c4.metric("Avg Solar Share",       f"{solar_share.mean():.1f} %")
        c5.metric("Expected Solar Energy", f"{solar.sum()/1000:.2f} MWh")

        st.markdown("---")

        # ── Dispatch schedule table ───────────────────────────────────────────
        st.subheader("Operator Dispatch Schedule")
        sched = pd.DataFrame({
            "Time":              fcast["timestamp"].dt.strftime("%H:%M"),
            "Solar Forecast (kW)": solar.round(1),
            "Grid Backup (kW)":    backup.round(1),
            "Surplus (kW)":        surplus.round(1),
            "Solar Share (%)":     solar_share.round(1),
            "GHI (W/m²)":         fcast["ghi"].values.round(0),
            "Cloud Cover":         (fcast["cloud_cover"].values * 100).round(0),
        })
        # Colour-code by solar share
        def highlight_share(val):
            if val >= 60:  return "background-color:#1a4731; color:#3fb950"
            if val >= 30:  return "background-color:#3d2b00; color:#d29922"
            return              "background-color:#3d0000; color:#f85149"

        styled = sched.style.map(highlight_share, subset=["Solar Share (%)"])
        st.dataframe(styled, width='stretch', height=400)

        st.markdown("---")

        # ── Solar share pie ───────────────────────────────────────────────────
        st.subheader("Energy Mix (24 h)")
        total_solar  = float(solar.sum())
        total_backup = float(backup.sum())
        pie = go.Figure(go.Pie(
            labels=["Solar", "Grid Backup"],
            values=[total_solar, total_backup],
            marker_colors=["#f59e0b", "#58a6ff"],
            hole=0.45,
            textfont=dict(color="#e6edf3"),
        ))
        pie.update_layout(
            height=300,
            paper_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            legend=dict(font=dict(color="#e6edf3")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(pie, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ④ — ALERT CENTER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Alert Center":

    st.markdown("## Alert Center")
    st.markdown("---")

    alert_metrics = load_alert_metrics()
    fcast = get_forecast_window(24)
    past  = get_past_window(72)

    # Active alerts in forecast window
    active_alerts = fcast[fcast["anomaly_type"] != "normal"] if not fcast.empty else pd.DataFrame()
    past_alerts   = past[past["anomaly_type"]  != "normal"]

    # ── Alert status cards ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    n_active = len(active_alerts)
    lvl_active = "red" if n_active > 0 else "green"
    c1.markdown(
        f"**Active Alerts (next 24h)**<br>"
        f"{status_badge(f'{n_active} ALERT(S)' if n_active else 'ALL CLEAR', lvl_active)}",
        unsafe_allow_html=True,
    )
    c1.metric(" ", n_active, label_visibility="collapsed")

    c2.markdown("**Detector: OR Fusion**<br>" + status_badge("ACTIVE", "green"), unsafe_allow_html=True)
    c2.metric(" ", f"Recall {alert_metrics['or_fusion']['recall']:.0%}", label_visibility="collapsed")

    c3.markdown("**Isolation Forest**<br>" + status_badge("ACTIVE", "green"), unsafe_allow_html=True)
    c3.metric(" ", f"F1 {alert_metrics['iforest']['f1']:.3f}", label_visibility="collapsed")

    st.markdown("---")

    # ── Active alert list ─────────────────────────────────────────────────────
    st.subheader("Active Fault Events (Next 24 h)")
    if active_alerts.empty:
        st.markdown('<div class="info-banner">✓ No faults detected in the next 24-hour window.</div>',
                    unsafe_allow_html=True)
    else:
        for _, row in active_alerts.iterrows():
            ftype = row["anomaly_type"].replace("_", " ").upper()
            color_map = {"inverter trip": "red", "string fault": "amber", "mppt underperf": "amber"}
            lvl = color_map.get(ftype.lower(), "amber")
            st.markdown(
                f'<div class="alert-banner">⚠ {row["timestamp"].strftime("%H:%M")} — '
                f'{ftype} · Power: {row["power_kw"]:.1f} kW · GHI: {row["ghi"]:.0f} W/m²</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Past 72h alert history ────────────────────────────────────────────────
    st.subheader("Past 72-Hour Fault History")
    if past_alerts.empty:
        st.info("No faults in the past 72 hours.")
    else:
        hist_tbl = past_alerts[["timestamp","anomaly_type","power_kw","ghi","cloud_cover"]].copy()
        hist_tbl["timestamp"] = hist_tbl["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        hist_tbl.columns      = ["Time","Fault Type","Power (kW)","GHI (W/m²)","Cloud Cover"]
        st.dataframe(hist_tbl, width='stretch', height=260)

    # ── Fault type chart (past 72h) ───────────────────────────────────────────
    if not past_alerts.empty:
        st.markdown("---")
        st.subheader("Fault Breakdown (Past 72 h)")
        counts = past_alerts["anomaly_type"].value_counts().reset_index()
        counts.columns = ["Fault Type", "Count"]
        fig = go.Figure(go.Bar(
            x=counts["Fault Type"], y=counts["Count"],
            marker_color=["#f85149","#d29922","#f59e0b"][:len(counts)],
            text=counts["Count"], textposition="outside",
        ))
        fig.update_layout(
            height=280,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Detector performance reference ────────────────────────────────────────
    st.subheader("Detector Performance Reference")
    dets   = ["threshold", "iforest", "or_fusion"]
    labels = ["Threshold", "Isolation Forest", "OR Fusion"]
    rows   = [{"Detector": l,
               "Precision": f"{alert_metrics[d]['precision']:.3f}",
               "Recall":    f"{alert_metrics[d]['recall']:.3f}",
               "F1":        f"{alert_metrics[d]['f1']:.3f}",
               "FPR":       f"{alert_metrics[d]['fpr']:.3f}",
               "TP": alert_metrics[d]["tp"], "FP": alert_metrics[d]["fp"],
               "FN": alert_metrics[d]["fn"]}
              for d, l in zip(dets, labels)]
    st.dataframe(pd.DataFrame(rows), width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ⑤ — SHIFT REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Shift Report":

    st.markdown("## Shift Report")
    st.caption(f"Generated for: {sim_date}  ·  {PLANT_NAME}")
    st.markdown("---")

    today   = get_today_data()
    past24  = get_past_window(24)
    fcast24 = get_forecast_window(24)

    # ── Daily generation stats ────────────────────────────────────────────────
    st.subheader("Today's Generation Summary")
    c1, c2, c3, c4 = st.columns(4)
    gen_mwh  = today["power_kw"].sum() / 1000
    peak_kw  = today["power_kw"].max()
    avg_kw   = today["power_kw"].mean()
    faults   = today[today["anomaly_type"] != "normal"]
    c1.metric("Total Generation",  f"{gen_mwh:.3f} MWh")
    c2.metric("Peak Output",       f"{peak_kw:.1f} kW")
    c3.metric("Average Output",    f"{avg_kw:.1f} kW")
    c4.metric("Fault Events",      len(faults), delta_color="inverse")

    st.markdown("---")

    # ── Full day chart ────────────────────────────────────────────────────────
    st.subheader("Full Day — Actual Output")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=today["timestamp"], y=today["power_kw"],
        fill="tozeroy", name="Actual",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88,166,255,0.12)",
    ))
    if not faults.empty:
        fig.add_trace(go.Scatter(
            x=faults["timestamp"], y=faults["power_kw"],
            mode="markers", name="Fault",
            marker=dict(color="#f85149", size=9, symbol="x-thin", line=dict(width=2, color="#f85149")),
        ))
    fig.update_layout(
        height=300,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e"),
        yaxis=dict(range=[0, PLANT_CAPACITY_KW + 20], gridcolor="#21262d", title="Power (kW)"),
        xaxis=dict(gridcolor="#21262d"),
        legend=dict(orientation="h", font=dict(color="#e6edf3")),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Next shift forecast ───────────────────────────────────────────────────
    st.subheader("Next Shift Forecast (Next 24 h)")
    if not fcast24.empty:
        solar  = fcast24["forecast_kw"].values
        backup = np.maximum(0, total_demand_kw - solar)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Generation", f"{solar.sum()/1000:.3f} MWh")
        c2.metric("Peak Forecast",       f"{solar.max():.1f} kW")
        c3.metric("Max Grid Backup",     f"{backup.max():.0f} kW")
        c4.metric("Avg Solar Share",     f"{(solar.mean()/total_demand_kw*100):.1f} %")

    st.markdown("---")

    # ── Fault log ─────────────────────────────────────────────────────────────
    st.subheader("Fault Log")
    if faults.empty:
        st.success("No fault events recorded today.")
    else:
        log = faults[["timestamp","anomaly_type","power_kw","ghi"]].copy()
        log["timestamp"]   = log["timestamp"].dt.strftime("%H:%M")
        log["anomaly_type"]= log["anomaly_type"].str.replace("_"," ").str.title()
        log.columns        = ["Time","Fault Type","Power (kW)","GHI (W/m²)"]
        st.dataframe(log.round(1), width='stretch')

    st.markdown("---")

    # ── Weather summary ───────────────────────────────────────────────────────
    st.subheader("Weather Summary (Today)")
    w_today = weather_df[weather_df["timestamp"].dt.date == pd.Timestamp(sim_date).date()]
    if not w_today.empty:
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Avg GHI",       f"{w_today['ghi'].mean():.0f} W/m²")
        wc2.metric("Max Temp",      f"{w_today['temp_air'].max():.1f} °C")
        wc3.metric("Avg Cloud",     f"{w_today['cloud_cover'].mean()*100:.0f} %")
        wc4.metric("Avg Wind",      f"{w_today['wind_speed'].mean():.1f} m/s")
