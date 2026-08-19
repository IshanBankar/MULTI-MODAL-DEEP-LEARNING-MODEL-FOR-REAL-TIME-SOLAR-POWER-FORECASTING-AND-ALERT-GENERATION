"""Run the 7-config ablation study + the attention-fusion variant.

Reproduces the structure of the paper:
  - Table I: best fusion strategy comparison (concat vs attention)
  - Table II: 7 modality combinations (TS-only, Met-only, Img-only, pairs, full)

Runs `train.py` as a subprocess for each config so each gets a clean,
seeded, isolated training run. Aggregates the final test metrics from each
run's metrics.json into a single CSV + a printed Markdown table.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONFIGS = [
    # name,          fusion,      use_ts, use_met, use_img
    ("TS_only",      "concat",    True,   False,   False),
    ("Met_only",     "concat",    False,  True,    False),
    ("Img_only",     "concat",    False,  False,   True),
    ("TS_Met",       "concat",    True,   True,    False),
    ("TS_Img",       "concat",    True,   False,   True),
    ("Met_Img",      "concat",    False,  True,    True),
    ("EarlyFusion",  "concat",    True,   True,    True),
    ("AttnFusion",   "attention", True,   True,    True),
]


def run_one(name: str, fusion: str, use_ts: bool, use_met: bool, use_img: bool,
            args, log) -> dict | None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{name}.pt"
    history = ckpt_dir / f"{name}_history.csv"

    cmd = [
        sys.executable, "train.py",
        "--fusion", fusion,
        "--ckpt", str(ckpt),
        "--history", str(history),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    if not use_ts: cmd.append("--no-ts")
    if not use_met: cmd.append("--no-met")
    if not use_img: cmd.append("--no-img")

    log.info(">>> %s  (%s)", name, " ".join(cmd[2:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("FAILED (%s) stderr: %s", name, proc.stderr[-1000:])
        return None

    metrics_path = ckpt.with_suffix(".metrics.json")
    if not metrics_path.exists():
        log.error("No metrics.json for %s", name); return None
    m = json.loads(metrics_path.read_text())
    m["name"] = name
    log.info("    -> RMSE=%.2f kW  MAE=%.2f kW  R^2=%.3f  nRMSE=%.2f%%",
             m["test_rmse_kw"], m["test_mae_kw"], m["test_r2"], m["test_nrmse_pct"])
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="data/checkpoints/ablation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20,
                    help="Bumped from train.py default 12 to give shorter "
                         "single-modality models more chances to recover.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/checkpoints/ablation/results.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ablation")

    results = []
    for name, fusion, use_ts, use_met, use_img in CONFIGS:
        r = run_one(name, fusion, use_ts, use_met, use_img, args, log)
        if r is not None:
            results.append(r)

    if not results:
        log.error("No successful runs"); sys.exit(1)

    df = pd.DataFrame(results)[
        ["name", "fusion", "use_ts", "use_met", "use_img",
         "params", "best_epoch", "epochs_run",
         "test_rmse_kw", "test_mae_kw", "test_r2", "test_nrmse_pct"]
    ]
    df.to_csv(args.out, index=False)
    log.info("Saved -> %s", args.out)

    # Compute the % degradation vs the full-fusion model
    base = df[df.name == "EarlyFusion"]["test_rmse_kw"]
    if len(base) == 1:
        df["rmse_vs_full_pct"] = (
            (df["test_rmse_kw"] - float(base.values[0])) / float(base.values[0]) * 100
        ).round(1)

    # Pretty markdown table to stdout
    print("\n=== Ablation Results (paper Table II structure) ===\n")
    cols_show = ["name", "fusion", "test_rmse_kw", "test_mae_kw", "test_r2",
                 "test_nrmse_pct"]
    if "rmse_vs_full_pct" in df.columns:
        cols_show.append("rmse_vs_full_pct")
    print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
