"""PyTorch dataset + dataloader for the multi-modal solar forecasting model.

Joins the three pre-built sources into samples ready for the LSTM, FC, and CNN
encoders described in the paper:

  - data/pune_500kw_hourly.csv      (power + meteo + anomaly labels)
  - data/satellite_patches.npz      (64x64x3 RGB patches per timestamp)

Per sample (prediction time = t):
  x_ts  : (24, 6)   24-hour window ending at t-1
                    [power_kw, ghi, temp_air, humidity, hour_sin, hour_cos]
  x_met : (5,)      meteorological scalars at t
                    [temp_air, humidity, wind_speed, cloud_cover, zenith_deg]
  x_img : (3,64,64) satellite patch at t (channel-first, float [0,1])
  y     : scalar    normalised power_kw at t+1 (training target)
  y_raw : scalar    raw power_kw at t+1 in kW (for metrics)

Chronological 70/15/15 split:
  Jan-Aug = train, Sep-Oct = val, Nov-Dec = test  (matches paper)

Normalisation: per-feature z-score fitted on training rows ONLY, then applied
to all splits. Target mean/std saved to data/norm_stats.json so inference can
denormalise predictions.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Feature lists from the paper
# ---------------------------------------------------------------------------
LSTM_FEATURES = ["power_kw", "ghi", "temp_air", "humidity"]
MET_FEATURES = ["temp_air", "humidity", "wind_speed", "cloud_cover", "zenith_deg"]
WINDOW = 24       # 24-hour input window
HORIZON = 1       # one-hour-ahead forecast

TRAIN_LAST_MONTH = 8   # Jan-Aug
VAL_LAST_MONTH = 10    # Sep-Oct (test = Nov-Dec)


# ---------------------------------------------------------------------------
# Normalisation stats container
# ---------------------------------------------------------------------------
@dataclass
class NormStats:
    lstm_mean: np.ndarray
    lstm_std: np.ndarray
    met_mean: np.ndarray
    met_std: np.ndarray
    target_mean: float
    target_std: float

    def to_dict(self):
        return {
            "lstm_features": LSTM_FEATURES,
            "lstm_mean": self.lstm_mean.tolist(),
            "lstm_std": self.lstm_std.tolist(),
            "met_features": MET_FEATURES,
            "met_mean": self.met_mean.tolist(),
            "met_std": self.met_std.tolist(),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
        }


def fit_norm_stats(df: pd.DataFrame) -> NormStats:
    """Fit z-score statistics on training rows only (Jan-Aug)."""
    train_df = df[df.index.month <= TRAIN_LAST_MONTH]
    return NormStats(
        lstm_mean=train_df[LSTM_FEATURES].to_numpy(np.float32).mean(axis=0),
        lstm_std=train_df[LSTM_FEATURES].to_numpy(np.float32).std(axis=0),
        met_mean=train_df[MET_FEATURES].to_numpy(np.float32).mean(axis=0),
        met_std=train_df[MET_FEATURES].to_numpy(np.float32).std(axis=0),
        target_mean=float(train_df["power_kw"].mean()),
        target_std=float(train_df["power_kw"].std()),
    )


# ---------------------------------------------------------------------------
# Index splitting
# ---------------------------------------------------------------------------
def make_split_indices(timestamps: pd.DatetimeIndex):
    """Valid prediction times t such that we have a full window before AND target after."""
    n = len(timestamps)
    months = timestamps.month.to_numpy()
    valid = np.arange(WINDOW, n - HORIZON)
    train = valid[months[valid] <= TRAIN_LAST_MONTH]
    val = valid[(months[valid] > TRAIN_LAST_MONTH) & (months[valid] <= VAL_LAST_MONTH)]
    test = valid[months[valid] > VAL_LAST_MONTH]
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SolarMultiModalDataset(Dataset):
    """Pre-computes all normalised feature arrays once; __getitem__ is just slicing."""

    def __init__(self, df: pd.DataFrame, images: np.ndarray,
                 stats: NormStats, indices: np.ndarray):
        self.indices = indices.astype(np.int64)
        self.images = images
        self.stats = stats

        # LSTM features: z-score the four core columns, append sin/cos hour
        lstm_raw = df[LSTM_FEATURES].to_numpy(np.float32)
        lstm_std_safe = np.where(stats.lstm_std > 1e-6, stats.lstm_std, 1.0)
        lstm_scaled = (lstm_raw - stats.lstm_mean) / lstm_std_safe

        hr = df.index.hour.to_numpy(np.float32) + df.index.minute.to_numpy(np.float32) / 60.0
        sin_h = np.sin(2 * np.pi * hr / 24.0).astype(np.float32)
        cos_h = np.cos(2 * np.pi * hr / 24.0).astype(np.float32)

        self._x_ts = np.concatenate(
            [lstm_scaled.astype(np.float32), sin_h[:, None], cos_h[:, None]], axis=1
        )  # (N, 6)

        # Meteorological scalars: z-score the five
        met_raw = df[MET_FEATURES].to_numpy(np.float32)
        met_std_safe = np.where(stats.met_std > 1e-6, stats.met_std, 1.0)
        self._x_met = ((met_raw - stats.met_mean) / met_std_safe).astype(np.float32)  # (N, 5)

        # Raw target in kW; normalise on the fly per __getitem__
        self._y_raw = df["power_kw"].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = int(self.indices[idx])
        x_ts = self._x_ts[t - WINDOW:t]                    # (24, 6)
        x_met = self._x_met[t]                              # (5,)
        img = self.images[t].astype(np.float32) / 255.0     # (64,64,3) in [0,1]
        x_img = np.transpose(img, (2, 0, 1)).copy()         # (3,64,64) channel-first

        y_kw = float(self._y_raw[t + HORIZON])
        y_norm = (y_kw - self.stats.target_mean) / max(self.stats.target_std, 1e-6)

        return (
            torch.from_numpy(x_ts),
            torch.from_numpy(x_met),
            torch.from_numpy(x_img),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(y_kw, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_datasets(csv_path: str, npz_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    with np.load(npz_path) as nz:
        images = nz["images"].copy()  # uint8 (N,64,64,3)

    if len(df) != len(images):
        raise ValueError(f"row count mismatch: csv={len(df)} npz={len(images)}")

    stats = fit_norm_stats(df)
    train_idx, val_idx, test_idx = make_split_indices(df.index)

    ds_train = SolarMultiModalDataset(df, images, stats, train_idx)
    ds_val = SolarMultiModalDataset(df, images, stats, val_idx)
    ds_test = SolarMultiModalDataset(df, images, stats, test_idx)
    return ds_train, ds_val, ds_test, stats, df


# ---------------------------------------------------------------------------
# Smoke test entrypoint
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    ap.add_argument("--npz", default="data/satellite_patches.npz")
    ap.add_argument("--save-stats", default="data/norm_stats.json")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dataset")

    log.info("Loading sources csv=%s npz=%s", args.csv, args.npz)
    ds_train, ds_val, ds_test, stats, df = build_datasets(args.csv, args.npz)
    log.info("Split sizes  train=%d  val=%d  test=%d",
             len(ds_train), len(ds_val), len(ds_test))
    log.info("Implied test n=%d  (paper reports n=1464)", len(ds_test))

    Path(args.save_stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_stats, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    log.info("Saved norm stats -> %s", args.save_stats)

    log.info("LSTM features %s  (+ hour_sin, hour_cos)", LSTM_FEATURES)
    log.info("MET features  %s", MET_FEATURES)
    log.info("Target mean=%.2f kW  std=%.2f kW",
             stats.target_mean, stats.target_std)
    log.info("LSTM mean=%s",
             np.round(stats.lstm_mean, 2).tolist())
    log.info("LSTM std =%s",
             np.round(stats.lstm_std, 2).tolist())

    # Smoke-test a batch from each split
    for name, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(name == "train"))
        x_ts, x_met, x_img, y, y_raw = next(iter(loader))
        log.info("%s batch shapes  x_ts=%s  x_met=%s  x_img=%s  y=%s  y_raw=%s",
                 name, tuple(x_ts.shape), tuple(x_met.shape),
                 tuple(x_img.shape), tuple(y.shape), tuple(y_raw.shape))
        log.info("    finite? x_ts=%s x_met=%s x_img=%s y=%s",
                 bool(torch.isfinite(x_ts).all()),
                 bool(torch.isfinite(x_met).all()),
                 bool(torch.isfinite(x_img).all()),
                 bool(torch.isfinite(y).all()))
        log.info("    y_raw range = [%.1f .. %.1f] kW",
                 float(y_raw.min()), float(y_raw.max()))


if __name__ == "__main__":
    main()
