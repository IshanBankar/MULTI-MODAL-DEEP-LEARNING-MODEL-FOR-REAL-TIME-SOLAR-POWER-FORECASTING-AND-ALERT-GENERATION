"""
Logs the 8 already-trained ablation runs (results.csv + per-run .metrics.json,
_history.csv, and .pt files) into MLflow so they show up in the MLflow UI as
one experiment with 8 comparable runs.

This does NOT retrain anything. It only reads files that already exist under
data/checkpoints/ablation/ and records their results.

Run this from inside the solar_forecast/ folder:
    python log_to_mlflow.py

Then view the results:
    mlflow ui
    (open http://127.0.0.1:5000 in your browser)
"""

import json
from pathlib import Path

import pandas as pd
import mlflow

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
ABLATION_DIR = BASE / "data" / "checkpoints" / "ablation"
RESULTS_CSV = ABLATION_DIR / "results.csv"

# ── MLflow setup ─────────────────────────────────────────────────────────
# Stores everything in a local SQLite file (mlflow.db) next to this script.
# This is MLflow's current recommended local setup (the older plain-folder
# store is deprecated as of MLflow 3.x).
mlflow.set_tracking_uri(f"sqlite:///{BASE / 'mlflow.db'}")
mlflow.set_experiment("solar-forecast-ablation")


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS_CSV}. "
            "Run this script from inside the solar_forecast/ folder, "
            "with data/checkpoints/ablation/results.csv present."
        )

    df = pd.read_csv(RESULTS_CSV)
    print(f"Found {len(df)} runs in results.csv. Logging each to MLflow...\n")

    for _, row in df.iterrows():
        name = row["name"]

        with mlflow.start_run(run_name=name):
            # ── Parameters: the settings that produced this result ──────
            mlflow.log_params({
                "fusion": row["fusion"],
                "use_ts": bool(row["use_ts"]),
                "use_met": bool(row["use_met"]),
                "use_img": bool(row["use_img"]),
                "params": int(row["params"]),
                "best_epoch": int(row["best_epoch"]),
                "epochs_run": int(row["epochs_run"]),
            })

            # ── Metrics: the numeric results for this run ────────────────
            mlflow.log_metrics({
                "test_rmse_kw": float(row["test_rmse_kw"]),
                "test_mae_kw": float(row["test_mae_kw"]),
                "test_r2": float(row["test_r2"]),
                "test_nrmse_pct": float(row["test_nrmse_pct"]),
            })

            # ── Tags: extra searchable labels ────────────────────────────
            mlflow.set_tag("model_family", "multi-modal-solar-forecast")
            mlflow.set_tag("modalities", "+".join(
                m for m, used in
                [("ts", row["use_ts"]), ("met", row["use_met"]), ("img", row["use_img"])]
                if used
            ))

            # ── Artifacts: attach the actual files for this run ──────────
            metrics_json = ABLATION_DIR / f"{name}.metrics.json"
            if metrics_json.exists():
                mlflow.log_artifact(str(metrics_json))

            history_csv = ABLATION_DIR / f"{name}_history.csv"
            if history_csv.exists():
                mlflow.log_artifact(str(history_csv))

            weights_pt = ABLATION_DIR / f"{name}.pt"
            if weights_pt.exists():
                mlflow.log_artifact(str(weights_pt))

            print(f"  logged: {name}  (RMSE={row['test_rmse_kw']} kW, R²={row['test_r2']})")

    print("\nDone. Run `mlflow ui` and open http://127.0.0.1:5000 to view all 8 runs.")


if __name__ == "__main__":
    main()
