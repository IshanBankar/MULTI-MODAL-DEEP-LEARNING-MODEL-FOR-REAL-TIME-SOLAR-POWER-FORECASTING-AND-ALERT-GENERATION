#!/usr/bin/env bash
# Quick-start script — run from inside the UI_Code folder
set -e

echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "Launching Solar PV Forecast Dashboard..."
streamlit run app.py --server.port 8501 --server.headless true
