# Solar Grid Control Center — Full Project

This repo contains the complete project:
- **`app.py`, `requirements.txt`, `data/`** — the live Streamlit dashboard (deployed at your app's URL)
- **`notebooks/`** — the 7-notebook research walkthrough (weather EDA → power EDA → satellite imagery → dataset pipeline → model training → baselines comparison → alert module)
- **`solar_forecast/`** — the full training pipeline: model architecture, dataset loader, training loop, baselines, ablations, alert module, and trained checkpoints
- **`PROJECT_EXPLANATION.md`** — a plain-language walkthrough of every notebook and term used

---

# Solar Grid Control Center — UI Code Reference

Operational dashboard for the Multi-Modal Solar PV Forecasting system.
Deployed as a Streamlit app. Simulates a grid control center that monitors
a 500 kW solar plant in Pune, India, forecasts next-hour power output, and
plans grid backup supply accordingly.

---

## Folder Structure

```
UI_Code/
├── app.py                          # Main Streamlit dashboard (single file)
├── requirements.txt                # Python dependencies
├── run.sh                          # Quick-start script
├── ui_code_readme.md               # This file
└── data/
    ├── pune_500kw_hourly.csv       # Hourly power + anomaly labels (8 760 rows)
    ├── weather_pune_2023.csv       # ERA5 hourly weather (8 760 rows)
    ├── norm_stats.json             # Z-score parameters from training
    ├── sample_seasons.png          # Satellite patch season preview
    ├── checkpoints/
    │   ├── history.csv             # Per-epoch training log (main model)
    │   ├── training_curves.png     # Training curves figure
    │   └── ablation/
    │       ├── results.csv         # 8-config ablation metrics
    │       ├── alert_metrics.json  # Alert detector P/R/F1/FPR
    │       ├── *_history.csv       # Per-config training histories
    │       └── *.metrics.json      # Per-config final test metrics
    └── figures/
        ├── fig5_scatter.png        # Predicted vs actual scatter
        ├── fig6_ablation.png       # RMSE by modality config
        ├── fig7_alerts.png         # Alert P/R/F1 bars
        ├── fig8_diurnal.png        # Diurnal RMSE profile
        ├── attention_diurnal.png   # Attention weights by hour
        └── weekly_strip.png        # Weekly time-series strip
```

**Total size: ~2.8 MB** (no model .pt weights or satellite .npz patches needed)

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open browser at: **http://localhost:8501**

Or use the convenience script:
```bash
bash run.sh
```

## Deployed on Streamlit Community Cloud

This repo is deployable as-is on [share.streamlit.io](https://share.streamlit.io) (free):
1. Push this repo to GitHub (root must contain `app.py` and `requirements.txt` — it already does)
2. Sign in to Streamlit Community Cloud with GitHub
3. "New app" → pick this repo → branch `main` → main file path `app.py`
4. Deploy

### Requirements

- Python 3.9+
- No GPU, no PyTorch, no CUDA needed
- Dependencies installed by `requirements.txt`:
  - `streamlit >= 1.32`
  - `pandas >= 2.0`
  - `plotly >= 5.18`
  - `Pillow >= 10.0`
  - `numpy` (comes with pandas)

---

## Dashboard Pages

### ⚡ Live Monitor
Real-time plant status view simulated from historical data.

- **Power gauge** — current output vs capacity (0–500 kW), colour-coded zones
- **Status badge** — NOMINAL / REDUCED / LOW OUTPUT / NO OUTPUT
- **Active fault banner** — red alert banner if anomaly detected at current timestamp
- **Key metrics** — capacity factor, next-hour forecast, grid backup needed, GHI,
  temperature, cloud cover, today's generation, wind speed, solar zenith
- **Today's strip chart** — actual (blue filled) vs forecast (green dashed), NOW marker

### 📈 Solar Forecast
Next 6 / 12 / 24 hour solar power forecast.

- **Forecast chart** — historical actual + forecast line + ±1.5 RMSE confidence band
- **Hourly forecast table** — time, forecast kW, upper/lower bounds, GHI, temp,
  cloud cover, grid backup kW, solar share %
- **Forecast KPIs** — peak, average, expected MWh, average solar share

Model used: Attention Fusion (LSTM + FC + CNN), RMSE = 19.33 kW, R² = 0.980.
Forecast is simulated as actual + Gaussian noise calibrated to model RMSE.

### 🔋 Grid Planning
Supply planning based on solar forecast vs configurable grid demand.

- **Supply stack chart** — stacked bars: solar (amber) + grid backup (blue),
  demand line (red dashed)
- **Planning KPIs** — peak solar, min/max grid backup, avg solar share, expected MWh
- **Dispatch schedule table** — colour-coded by solar share
  (green ≥ 60%, amber ≥ 30%, red < 30%)
- **Energy mix pie** — solar vs grid backup for the 24-hour window

### 🚨 Alert Center
Fault detection and anomaly management.

- **Status cards** — active alert count, OR Fusion status, Isolation Forest status
- **Active fault list** — timestamped fault banners for next 24h window
- **72-hour fault history** — table of past faults with power and weather context
- **Fault breakdown chart** — count by fault type (past 72h)
- **Detector reference table** — Precision / Recall / F1 / FPR for all three
  detectors (Threshold, Isolation Forest, OR Fusion)

Fault types in data:
| Type | Effect | Detectability |
|---|---|---|
| `inverter_trip` | Full zero output | 100% (median residual 203 kW) |
| `string_fault` | 15–35% drop | 71% (median residual 56 kW) |
| `mppt_underperf` | 5–15% drop | 0% (inside noise floor at 14 kW) |

### 📋 Shift Report
End-of-shift summary for handover.

- **Daily generation summary** — total MWh, peak kW, average kW, fault event count
- **Full day chart** — actual output with fault markers
- **Next shift forecast** — 24h ahead expected generation and max grid backup
- **Fault log** — timestamped table of all fault events for the day
- **Weather summary** — avg GHI, max temp, avg cloud cover, avg wind speed

---

## Sidebar Controls

| Control | Description | Default |
|---|---|---|
| **Current date** | Sets the simulation "now" date | 2023-09-15 |
| **Current hour** | Sets the simulation "now" hour (0–23) | 10 |
| **Total grid demand (kW)** | Demand against which solar share and backup are calculated | 800 kW |
| **Alert sensitivity** | FPR target label (display only) | Medium (FPR 0.15) |

The simulation clock lets you scrub through the full year (Jan–Dec 2023) to
see any time period as if it were live.

---

## Key Model Facts (for context)

| Metric | Value |
|---|---|
| Model | Attention Fusion (LSTM + FC + CNN) |
| Test RMSE | 19.33 kW |
| Test R² | 0.980 |
| nRMSE | 3.87% of 500 kW nameplate |
| Attention weights | TS=0.783, Img=0.148, Met=0.070 |
| Plant capacity | 500 kW |
| Location | 18.6°N, 73.8°E, Pune, India |
| Data year | 2023 (ERA5 + pvlib digital twin) |
| Train/Val/Test split | Jan–Aug / Sep–Oct / Nov–Dec |

---

## Updating the App

Push changes to `main` on GitHub — Streamlit Community Cloud auto-redeploys on every push.
