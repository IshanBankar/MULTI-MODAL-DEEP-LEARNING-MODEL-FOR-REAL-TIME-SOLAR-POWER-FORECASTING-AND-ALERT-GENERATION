"""Generate paper-style evaluation plots from a trained checkpoint.

Reproduces:
  - Fig 5: Predicted vs Actual Power scatter (test set, colour = absolute error)
  - Fig 6: Ablation bar chart over the 7 modality combinations
  - Fig 8: Diurnal RMSE profile (RMSE by hour of day)
  - Bonus: Predicted vs Actual time-series strip for one representative week

Default checkpoint is the best-attention-fusion model (lowest test RMSE).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dataset import build_datasets
from model import MultiModalSolarModel


def load_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["args"]
    model = MultiModalSolarModel(
        fusion=cfg["fusion"],
        use_ts=not cfg.get("no_ts", False),
        use_met=not cfg.get("no_met", False),
        use_img=not cfg.get("no_img", False),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def predict_test(model, loader, device, target_mean, target_std):
    preds_kw, targets_kw, alphas = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_ts, x_met, x_img, _, y_kw = [b.to(device) for b in batch]
            kw_args = dict(
                x_ts=x_ts if model.use_ts else None,
                x_met=x_met if model.use_met else None,
                x_img=x_img if model.use_img else None,
            )
            if model.fusion_type == "attention" and len(model.encoder_order) > 1:
                pred, alpha = model(**kw_args, return_alpha=True)
                alphas.append(alpha.cpu().numpy())
            else:
                pred = model(**kw_args)
            pred_kw = pred * target_std + target_mean
            preds_kw.append(pred_kw.cpu().numpy())
            targets_kw.append(y_kw.cpu().numpy())
    preds_kw = np.concatenate(preds_kw)
    targets_kw = np.concatenate(targets_kw)
    alpha = np.concatenate(alphas) if alphas else None
    return preds_kw, targets_kw, alpha


def fig_scatter(preds, targets, out_path: Path, title_prefix: str):
    err = np.abs(preds - targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(err))
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(targets, preds, c=err, cmap="RdYlGn_r",
                    s=10, alpha=0.65, edgecolors="none")
    lim = max(targets.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="perfect prediction (y=x)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Power Output (kW)")
    ax.set_ylabel("Predicted Power Output (kW)")
    ax.set_title(f"Fig 5 reproduction: {title_prefix}\n"
                 f"RMSE={rmse:.2f} kW  MAE={mae:.2f} kW  R^2={r2:.3f}")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Error (kW)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def fig_diurnal(preds, targets, timestamps, lstm_preds, lstm_targets,
                out_path: Path):
    """Hourly RMSE profile, proposed vs LSTM baseline (paper Fig 8)."""
    df_full = pd.DataFrame({"hour": timestamps.hour,
                            "err2": (preds - targets) ** 2}).groupby("hour")["err2"]
    rmse_full = np.sqrt(df_full.mean()).reindex(range(24), fill_value=np.nan)
    df_lstm = pd.DataFrame({"hour": timestamps.hour,
                            "err2": (lstm_preds - lstm_targets) ** 2}).groupby("hour")["err2"]
    rmse_lstm = np.sqrt(df_lstm.mean()).reindex(range(24), fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    h = np.arange(24)
    ax.plot(h, rmse_lstm.values, "-o", label="Standalone LSTM", color="C1")
    ax.plot(h, rmse_full.values, "-o", label="Proposed (multi-modal)", color="C0")
    ax.fill_between(h, rmse_full.values, rmse_lstm.values,
                    where=(rmse_lstm.values > rmse_full.values),
                    alpha=0.2, color="C0", label="Improvement")
    ax.set_xlabel("Hour of Day (IST)")
    ax.set_ylabel("Hourly RMSE (kW)")
    ax.set_title("Fig 8 reproduction: Diurnal RMSE Profile  (test set)")
    ax.set_xticks(range(24))
    ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def fig_ablation(results_csv: str, out_path: Path):
    """Reproduce Fig 6 - ablation by modality combination."""
    df = pd.read_csv(results_csv)
    order = ["TS_only", "Met_only", "Img_only",
             "TS_Met", "TS_Img", "Met_Img",
             "EarlyFusion", "AttnFusion"]
    df = df.set_index("name").reindex(order).reset_index()
    colors = ["#5b9bd5"]*3 + ["#ed7d31"]*3 + ["#2e75b6", "#548235"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["name"], df["test_rmse_kw"], color=colors)
    for b, v in zip(bars, df["test_rmse_kw"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_title("Fig 6 reproduction: RMSE by modality combination")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def fig_weekly_strip(preds, targets, timestamps, out_path: Path):
    df = pd.DataFrame({"actual": targets, "pred": preds}, index=timestamps)
    # First full week of December
    week = df.loc["2023-12-04":"2023-12-10"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week["actual"], "-", lw=1.4, label="Actual", color="C0")
    ax.plot(week.index, week["pred"],   "--", lw=1.2, label="Predicted", color="C3")
    ax.set_ylabel("Power (kW)"); ax.set_xlabel("Time (IST)")
    ax.set_title("Bonus: Predicted vs Actual - First week of December 2023")
    ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/checkpoints/ablation/AttnFusion.pt")
    ap.add_argument("--lstm-ckpt", default="data/checkpoints/ablation/TS_only.pt",
                    help="Standalone LSTM checkpoint for diurnal-comparison plot")
    ap.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    ap.add_argument("--npz", default="data/satellite_patches.npz")
    ap.add_argument("--ablation-csv", default="data/checkpoints/ablation/results.csv")
    ap.add_argument("--out-dir", default="data/figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("eval")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log.info("Building test loader")
    ds_train, ds_val, ds_test, stats, df_full = build_datasets(args.csv, args.npz)
    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False, num_workers=2)
    test_indices = ds_test.indices  # array of t in df_full
    test_timestamps = df_full.index[test_indices + 1]  # target times = t+1

    log.info("Loading main checkpoint %s", args.ckpt)
    model, state = load_checkpoint(args.ckpt, device)
    title = f"{state['args']['fusion']} fusion, " \
            f"{'+'.join(model.encoder_order)}"
    preds, targets, alpha = predict_test(model, test_loader, device,
                                         stats.target_mean, stats.target_std)

    log.info("Plotting scatter")
    fig_scatter(preds, targets, out_dir / "fig5_scatter.png",
                title_prefix=title)

    log.info("Loading LSTM-only checkpoint %s", args.lstm_ckpt)
    lstm_model, _ = load_checkpoint(args.lstm_ckpt, device)
    lstm_preds, lstm_targets, _ = predict_test(lstm_model, test_loader, device,
                                               stats.target_mean, stats.target_std)
    log.info("Plotting diurnal profile")
    fig_diurnal(preds, targets, test_timestamps, lstm_preds, lstm_targets,
                out_dir / "fig8_diurnal.png")

    if Path(args.ablation_csv).exists():
        log.info("Plotting ablation bars")
        fig_ablation(args.ablation_csv, out_dir / "fig6_ablation.png")

    log.info("Plotting weekly time series strip")
    fig_weekly_strip(preds, targets, test_timestamps, out_dir / "weekly_strip.png")

    if alpha is not None:
        log.info("Mean attention weights on test set: ts=%.3f met=%.3f img=%.3f",
                 *alpha.mean(axis=0))
        # Diurnal alpha plot
        ah = pd.DataFrame(alpha, columns=model.encoder_order,
                          index=test_timestamps).groupby(test_timestamps.hour).mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ah.columns:
            ax.plot(ah.index, ah[col], "-o", label=col)
        ax.set_xlabel("Hour of Day (IST)"); ax.set_ylabel("Attention weight")
        ax.set_title("Diurnal attention weights (Attn-Fusion model)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "attention_diurnal.png", dpi=130); plt.close()

    log.info("All figures -> %s", out_dir)


if __name__ == "__main__":
    main()
