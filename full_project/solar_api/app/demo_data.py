"""Loads the trimmed demo slice (Nov-Dec test period + 24h buffer) so the
/predict/sample/{index} endpoint can demo live inference against real
plant data without requiring the caller to construct a raw payload."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "demo_slice.csv"
NPZ_PATH = DATA_DIR / "demo_slice_images.npz"

WINDOW = 24
LSTM_FEATURES = ["power_kw", "ghi", "temp_air", "humidity"]
MET_FEATURES = ["temp_air", "humidity", "wind_speed", "cloud_cover", "zenith_deg"]


class DemoDataset:
    def __init__(self):
        self.df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], index_col="timestamp")
        with np.load(NPZ_PATH) as nz:
            self.images = nz["images"]
        if len(self.df) != len(self.images):
            raise ValueError("demo csv/npz row mismatch")
        # valid indices: need WINDOW rows before t, and t+1 (target) within range
        self.valid_indices = list(range(WINDOW, len(self.df) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def get_sample(self, position: int):
        """position indexes into self.valid_indices (0-based, demo-friendly)."""
        if position < 0 or position >= len(self.valid_indices):
            raise IndexError(f"position must be in [0, {len(self.valid_indices) - 1}]")
        t = self.valid_indices[position]

        window_df = self.df.iloc[t - WINDOW:t]
        x_ts_raw = window_df[LSTM_FEATURES].to_numpy(dtype=np.float32)
        hours = (window_df.index.hour + window_df.index.minute / 60.0).to_numpy(dtype=np.float32)

        met_row = self.df.iloc[t]
        x_met_raw = met_row[MET_FEATURES].to_numpy(dtype=np.float32)

        image_uint8 = self.images[t]

        actual_kw = float(self.df.iloc[t + 1]["power_kw"])
        target_timestamp = str(self.df.index[t + 1])
        current_timestamp = str(self.df.index[t])

        return {
            "x_ts_raw": x_ts_raw,
            "hours": hours,
            "x_met_raw": x_met_raw,
            "image_uint8": image_uint8,
            "actual_power_kw": actual_kw,
            "current_timestamp": current_timestamp,
            "target_timestamp": target_timestamp,
        }


@lru_cache(maxsize=1)
def get_demo_dataset() -> DemoDataset:
    return DemoDataset()
