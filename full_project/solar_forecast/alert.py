"""Alert pipeline: threshold rule + IsolationForest + OR-fusion.

Implements the paper's Section III-E:
  delta_t = |y_pred - y_true| / max(y_true, 1 kW)
  A_thresh : delta_t > 0.15
  A_iforest: IsolationForest on [residual, MA3(residual), met_vector_at_t]
             trained on TRAIN-set residuals only (no leakage)
  Alert    : A_thresh OR A_iforest  (OR fusion)

Evaluates on the TEST split against the `is_anomaly` ground truth from
data_generator.py's labelled injection. Reports precision / recall / F1 / FPR
per detector and saves a Fig 7 reproduction.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from dataset import MET_FEATURES, build_datasets
from evaluate import load_checkpoint, predict_test


# ---------------------------------------------------------------------------
def compute_features(preds_kw, targets_kw, met_array):
    """Build per-sample features for the IsolationForest detector."""
    residual = preds_kw - targets_kw
    ma3 = pd.Series(residual).rolling(3, min_periods=1).mean().to_numpy()
    return np.column_stack([residual, ma3, met_array]).astype(np.float32)


def metrics_from_alerts(alert: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(((alert == 1) & (gt == 1)).sum())
    fp = int(((alert == 1) & (gt == 0)).sum())
    fn = int(((alert == 0) & (gt == 1)).sum())
    tn = int(((alert == 0) & (gt == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return dict(precision=p, recall=r, f1=f1, fpr=fpr,
                tp=tp, fp=fp, fn=fn, tn=tn)


def fig_alert_bars(metrics: dict, out_path: Path):
    detectors = ["threshold", "iforest", "or_fusion"]
    labels = ["Threshold Only", "Isolation Forest Only", "OR Fusion (Proposed)"]
    metric_names = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(detectors))
    width = 0.25
    for i, m in enumerate(metric_names):
        vals = [metrics[d][m] for d in detectors]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 7 reproduction: Alert detection precision / recall / F1")
    ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/checkpoints/ablation/AttnFusion.pt")
    ap.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    ap.add_argument("--npz", default="data/satellite_patches.npz")
    ap.add_argument("--out-dir", default="data/figures")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="Absolute relative-deviation threshold (paper: 0.15)")
    ap.add_argument("--auto-threshold", action="store_true",
                    help="Override threshold by picking the (1-fpr) quantile "
                         "of the training-set residual distribution. Recommended "
                         "for the digital twin since paper's 0.15 is calibrated "
                         "to a narrower error distribution.")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="Target false-positive rate when --auto-threshold is set")
    ap.add_argument("--iforest-n", type=int, default=200)
    ap.add_argument("--iforest-contamination", type=float, default=0.05)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("alert")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    log.info("Building datasets")
    ds_train, ds_val, ds_test, stats, df_full = build_datasets(args.csv, args.npz)
    # NOTE: shuffle=False so prediction order matches ds.indices order
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False, num_workers=2)

    log.info("Loading checkpoint %s", args.ckpt)
    model, state = load_checkpoint(args.ckpt, device)

    log.info("Predicting on train (for IForest fitting)")
    preds_train, targets_train, _ = predict_test(
        model, train_loader, device, stats.target_mean, stats.target_std)
    log.info("Predicting on test")
    preds_test, targets_test, _ = predict_test(
        model, test_loader, device, stats.target_mean, stats.target_std)

    # Met vectors at prediction-time t (NOT t+1) - matches paper "current meteo state"
    train_met = df_full.iloc[ds_train.indices][MET_FEATURES].to_numpy(np.float32)
    test_met = df_full.iloc[ds_test.indices][MET_FEATURES].to_numpy(np.float32)

    # Restrict alert evaluation to daytime samples (zenith < 85 deg).
    # Anomalies are only injected during daylight; the threshold rule is
    # only meaningful when actual power is non-trivial. This matches the
    # paper's operational framing.
    target_idx = ds_test.indices + 1
    train_target_idx = ds_train.indices + 1
    train_zenith = df_full.iloc[train_target_idx]["zenith_deg"].to_numpy()
    test_zenith = df_full.iloc[target_idx]["zenith_deg"].to_numpy()
    train_day = train_zenith < 85
    test_day = test_zenith < 85
    log.info("Daytime filter: train %d/%d  test %d/%d",
             int(train_day.sum()), len(train_day),
             int(test_day.sum()), len(test_day))

    X_train = compute_features(preds_train[train_day], targets_train[train_day],
                               train_met[train_day])
    X_test_full = compute_features(preds_test, targets_test, test_met)
    X_test = X_test_full[test_day]
    log.info("Feature matrices: X_train=%s X_test=%s", X_train.shape, X_test.shape)

    log.info("Fitting IsolationForest (n=%d, cont=%.3f) on TRAIN residuals",
             args.iforest_n, args.iforest_contamination)
    clf = IsolationForest(
        n_estimators=args.iforest_n,
        contamination=args.iforest_contamination,
        random_state=42,
    )
    clf.fit(X_train)
    iforest_alert_day = (clf.predict(X_test) == -1).astype(int)

    # Threshold rule on daytime samples only
    preds_day = preds_test[test_day]
    targets_day = targets_test[test_day]
    delta = np.abs(preds_day - targets_day) / np.maximum(targets_day, 1.0)

    threshold = args.threshold
    if args.auto_threshold:
        # Calibrate threshold from training residuals on non-anomalous samples
        train_targets_day = targets_train[train_day]
        train_preds_day = preds_train[train_day]
        train_delta = (np.abs(train_preds_day - train_targets_day)
                       / np.maximum(train_targets_day, 1.0))
        train_anom = df_full.iloc[train_target_idx]["is_anomaly"].to_numpy()[train_day]
        clean = train_delta[~train_anom]
        threshold = float(np.quantile(clean, 1 - args.target_fpr))
        log.info("Auto-calibrated threshold = %.3f "
                 "(quantile %.2f of clean training delta)",
                 threshold, 1 - args.target_fpr)
    thresh_alert = (delta > threshold).astype(int)
    iforest_alert = iforest_alert_day
    or_alert = (thresh_alert | iforest_alert).astype(int)

    # Ground truth: anomaly labels at TARGET time t+1, daytime only
    gt_full = df_full.iloc[target_idx]["is_anomaly"].astype(int).to_numpy()
    gt = gt_full[test_day]

    log.info("Test set: n=%d  GT anomalies=%d (%.2f%%)",
             len(gt), int(gt.sum()), 100 * gt.mean())
    log.info("Threshold alerts: %d   IForest alerts: %d   OR-fusion alerts: %d",
             int(thresh_alert.sum()), int(iforest_alert.sum()), int(or_alert.sum()))

    metrics = {
        "threshold": metrics_from_alerts(thresh_alert, gt),
        "iforest":   metrics_from_alerts(iforest_alert, gt),
        "or_fusion": metrics_from_alerts(or_alert, gt),
    }

    # Recall by anomaly type (diagnostic - which kinds of anomalies are catchable)
    anomaly_type = df_full.iloc[target_idx]["anomaly_type"].to_numpy()[test_day]
    log.info("Recall by anomaly type (OR-fusion):")
    for t in ["inverter_trip", "string_fault", "mppt_underperf"]:
        mask = anomaly_type == t
        n = int(mask.sum())
        caught = int(or_alert[mask].sum())
        log.info("  %-20s n=%3d caught=%3d recall=%s",
                 t, n, caught,
                 f"{caught/n:.3f}" if n > 0 else "n/a")
    # Median residual magnitude by type
    residual = preds_day - targets_day
    log.info("Median |residual| (kW) by type:")
    for t in ["normal", "inverter_trip", "string_fault", "mppt_underperf"]:
        mask = anomaly_type == t
        if mask.sum() > 0:
            log.info("  %-20s n=%4d median |residual|=%.1f kW",
                     t, int(mask.sum()),
                     float(np.median(np.abs(residual[mask]))))

    paper_ref = {
        "threshold": (0.950, 0.720, 0.819, 0.038),
        "iforest":   (0.830, 0.850, 0.840, 0.173),
        "or_fusion": (0.910, 0.880, 0.894, 0.089),
    }

    log.info("=" * 80)
    log.info("ALERT MODULE EVALUATION (paper Table III)")
    log.info("=" * 80)
    log.info(f"{'Detector':<22} {'Precision':>10} {'Recall':>10} "
             f"{'F1':>10} {'FPR':>10}    {'(paper P/R/F1/FPR)':>30}")
    for name, m in metrics.items():
        ref = paper_ref[name]
        log.info(f"{name:<22} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                 f"{m['f1']:>10.3f} {m['fpr']:>10.3f}   "
                 f"({ref[0]:.3f}/{ref[1]:.3f}/{ref[2]:.3f}/{ref[3]:.3f})")

    metrics_path = Path(args.ckpt).parent / "alert_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=int))
    log.info("Metrics -> %s", metrics_path)
    fig_alert_bars(metrics, out_dir / "fig7_alerts.png")
    log.info("Figure  -> %s", out_dir / "fig7_alerts.png")


if __name__ == "__main__":
    main()
