"""Inference wrapper around the trained MultiModalSolarModel.

Loads the checkpoint once at process start and exposes a single `predict()`
function used by the FastAPI routes. Keeps normalisation logic in one place
so the training-time z-score stats (norm_stats.json) and the model
architecture (model.py) stay in sync with what was actually trained.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import MultiModalSolarModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_PATH = DATA_DIR / "best.pt"
NORM_STATS_PATH = DATA_DIR / "norm_stats.json"

LSTM_FEATURES = ["power_kw", "ghi", "temp_air", "humidity"]
MET_FEATURES = ["temp_air", "humidity", "wind_speed", "cloud_cover", "zenith_deg"]
WINDOW = 24


class SolarForecaster:
    """Wraps the trained model + normalisation stats for single-sample inference."""

    def __init__(self, ckpt_path: Path = CKPT_PATH, stats_path: Path = NORM_STATS_PATH,
                 device: str = "cpu"):
        self.device = torch.device(device)

        with open(stats_path) as f:
            self.stats = json.load(f)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = state["args"]
        self.model = MultiModalSolarModel(
            fusion=cfg["fusion"],
            use_ts=not cfg.get("no_ts", False),
            use_met=not cfg.get("no_met", False),
            use_img=not cfg.get("no_img", False),
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

        self.target_mean = state["target_mean"]
        self.target_std = state["target_std"]
        self.checkpoint_meta = {
            "epoch": state.get("epoch"),
            "val_rmse_kw": state.get("val_rmse_kw"),
            "fusion": cfg["fusion"],
            "encoder_order": state.get("encoder_order"),
        }

        self.lstm_mean = np.array(self.stats["lstm_mean"], dtype=np.float32)
        self.lstm_std = np.where(np.array(self.stats["lstm_std"], dtype=np.float32) > 1e-6,
                                  self.stats["lstm_std"], 1.0)
        self.met_mean = np.array(self.stats["met_mean"], dtype=np.float32)
        self.met_std = np.where(np.array(self.stats["met_std"], dtype=np.float32) > 1e-6,
                                 self.stats["met_std"], 1.0)

    def _build_ts_tensor(self, x_ts_raw: np.ndarray, hours: np.ndarray) -> torch.Tensor:
        """x_ts_raw: (24, 4) raw [power_kw, ghi, temp_air, humidity]. hours: (24,) hour-of-day."""
        scaled = (x_ts_raw - self.lstm_mean) / self.lstm_std
        sin_h = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        cos_h = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
        x_ts = np.concatenate([scaled.astype(np.float32), sin_h[:, None], cos_h[:, None]], axis=1)
        return torch.from_numpy(x_ts).unsqueeze(0).to(self.device)  # (1, 24, 6)

    def _build_met_tensor(self, x_met_raw: np.ndarray) -> torch.Tensor:
        scaled = ((x_met_raw - self.met_mean) / self.met_std).astype(np.float32)
        return torch.from_numpy(scaled).unsqueeze(0).to(self.device)  # (1, 5)

    def _build_img_tensor(self, image_uint8: np.ndarray) -> torch.Tensor:
        """image_uint8: (64, 64, 3) uint8 RGB."""
        img = image_uint8.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)).copy()  # (3, 64, 64)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, 3, 64, 64)

    def predict(self, x_ts_raw: np.ndarray, hours: np.ndarray, x_met_raw: np.ndarray,
                image_uint8: np.ndarray) -> dict:
        """Run one forward pass and return predicted kW + timing."""
        t0 = time.perf_counter()
        with torch.no_grad():
            x_ts = self._build_ts_tensor(x_ts_raw, hours)
            x_met = self._build_met_tensor(x_met_raw)
            x_img = self._build_img_tensor(image_uint8)
            pred_norm = self.model(x_ts=x_ts, x_met=x_met, x_img=x_img)
            pred_kw = float(pred_norm.item()) * self.target_std + self.target_mean
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predicted_power_kw": max(0.0, pred_kw),  # plant can't produce negative power
            "latency_ms": round(latency_ms, 3),
        }


_forecaster: SolarForecaster | None = None


def get_forecaster() -> SolarForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = SolarForecaster()
    return _forecaster
