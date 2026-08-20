"""FastAPI live-inference service for the multi-modal solar PV forecasting model.

Endpoints:
  GET  /health                    - liveness + model metadata
  GET  /predict/sample/{position} - run inference on a real held-out sample
                                     from the Nov-Dec test period, returns
                                     predicted vs actual kW
  POST /predict                   - run inference on a caller-supplied
                                     24h window + met scalars + satellite image
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from .demo_data import get_demo_dataset
from .inference import get_forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + demo data once at process start rather than on first request.
    get_forecaster()
    get_demo_dataset()
    yield


app = FastAPI(
    title="Solar PV Forecasting API",
    description="Live next-hour power forecast for a 500kW solar plant, "
                "served from a trained multi-modal LSTM+CNN+FC model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    forecaster = get_forecaster()
    dataset = get_demo_dataset()
    return {
        "status": "ok",
        "model": forecaster.checkpoint_meta,
        "demo_samples_available": len(dataset),
    }


@app.get("/predict/sample/{position}")
def predict_sample(position: int):
    """Run live inference on a real sample from the held-out test period.

    `position` is a 0-based index into the demo slice (Nov-Dec + buffer).
    Returns predicted vs actual next-hour power output.
    """
    dataset = get_demo_dataset()
    try:
        sample = dataset.get_sample(position)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))

    forecaster = get_forecaster()
    result = forecaster.predict(
        x_ts_raw=sample["x_ts_raw"],
        hours=sample["hours"],
        x_met_raw=sample["x_met_raw"],
        image_uint8=sample["image_uint8"],
    )

    predicted = result["predicted_power_kw"]
    actual = sample["actual_power_kw"]
    return {
        "current_timestamp": sample["current_timestamp"],
        "target_timestamp": sample["target_timestamp"],
        "predicted_power_kw": round(predicted, 2),
        "actual_power_kw": round(actual, 2),
        "absolute_error_kw": round(abs(predicted - actual), 2),
        "latency_ms": result["latency_ms"],
    }


class PredictRequest(BaseModel):
    x_ts: list[list[float]] = Field(
        ..., description="24 hourly rows, each [power_kw, ghi, temp_air, humidity], "
                          "oldest first, ending at the hour before the forecast target."
    )
    hours_of_day: list[float] = Field(
        ..., description="24 hour-of-day values (0-23.99) matching each row in x_ts."
    )
    x_met: list[float] = Field(
        ..., description="5 current meteorological scalars: "
                          "[temp_air, humidity, wind_speed, cloud_cover, zenith_deg]."
    )
    image_base64: str = Field(
        ..., description="Base64-encoded RGB image (any size, resized to 64x64) "
                          "of the satellite patch at the current hour."
    )


@app.post("/predict")
def predict(req: PredictRequest):
    """Run live inference on caller-supplied raw features."""
    if len(req.x_ts) != 24:
        raise HTTPException(status_code=422, detail="x_ts must contain exactly 24 rows")
    if any(len(row) != 4 for row in req.x_ts):
        raise HTTPException(status_code=422, detail="each x_ts row must have 4 values")
    if len(req.hours_of_day) != 24:
        raise HTTPException(status_code=422, detail="hours_of_day must contain 24 values")
    if len(req.x_met) != 5:
        raise HTTPException(status_code=422, detail="x_met must contain exactly 5 values")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((64, 64))
        image_uint8 = np.array(image, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not decode image_base64: {e}")

    forecaster = get_forecaster()
    result = forecaster.predict(
        x_ts_raw=np.array(req.x_ts, dtype=np.float32),
        hours=np.array(req.hours_of_day, dtype=np.float32),
        x_met_raw=np.array(req.x_met, dtype=np.float32),
        image_uint8=image_uint8,
    )
    return {
        "predicted_power_kw": round(result["predicted_power_kw"], 2),
        "latency_ms": result["latency_ms"],
    }
