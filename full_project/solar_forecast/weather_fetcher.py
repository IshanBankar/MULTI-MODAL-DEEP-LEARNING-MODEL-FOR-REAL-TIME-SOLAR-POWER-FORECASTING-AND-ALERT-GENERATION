"""Fetch real ERA5 hourly weather for Pune via Open-Meteo's archive API.

Open-Meteo's free archive serves ERA5 reanalysis directly - same underlying
dataset Copernicus CDS provides, but with no API key or queue.
Reference: https://open-meteo.com/en/docs/historical-weather-api

Output schema is aligned with data_generator.py so the file can be passed in
via --weather-csv to drive the digital-twin physics with real weather instead
of synthetic Pune climatology.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo variable names. See API docs for the full list.
VARIABLES = [
    "temperature_2m",            # degC
    "relative_humidity_2m",      # %
    "wind_speed_10m",            # m/s (with wind_speed_unit=ms)
    "cloud_cover",               # %
    "shortwave_radiation",       # GHI on horizontal, W/m^2
    "direct_radiation",          # direct on horizontal, W/m^2
    "diffuse_radiation",         # DHI on horizontal, W/m^2
    "direct_normal_irradiance",  # DNI on sun-normal, W/m^2
    "surface_pressure",          # hPa
    "precipitation",             # mm
]


def fetch(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": tz,
        "wind_speed_unit": "ms",
    }
    r = requests.get(URL, params=params, timeout=180)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Unexpected response shape: {json.dumps(data)[:400]}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    df = df.drop(columns=["time"]).set_index("timestamp")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=18.6)
    ap.add_argument("--lon", type=float, default=73.8)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--tz", default="Asia/Kolkata")
    ap.add_argument("--out", default="data/weather_pune_2023.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("weather")

    log.info("Open-Meteo ERA5 archive (%.3f, %.3f) %s..%s tz=%s",
             args.lat, args.lon, args.start, args.end, args.tz)
    raw = fetch(args.lat, args.lon, args.start, args.end, args.tz)
    log.info("Received %d hourly rows; %d variables", len(raw), len(raw.columns))

    # Convert to the schema data_generator.py expects, plus extras
    out = pd.DataFrame(index=raw.index)
    out["temp_air"] = raw["temperature_2m"]
    out["humidity"] = raw["relative_humidity_2m"]
    out["wind_speed"] = raw["wind_speed_10m"]
    out["cloud_cover"] = raw["cloud_cover"] / 100.0          # % -> fraction
    out["ghi"] = raw["shortwave_radiation"]
    out["dni_horizontal"] = raw["direct_radiation"]
    out["dhi"] = raw["diffuse_radiation"]
    out["dni"] = raw["direct_normal_irradiance"]
    out["pressure_pa"] = raw["surface_pressure"] * 100.0     # hPa -> Pa
    out["precip_mm"] = raw["precipitation"]
    out.index.name = "timestamp"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    log.info("Wrote %s (%d rows, %d cols)", out_path, len(out), len(out.columns))

    # Sanity stats so you can compare against the synthetic version
    daytime = out[out["ghi"] > 10]
    log.info("Daytime mean GHI: %.0f W/m^2 | peak: %.0f",
             daytime["ghi"].mean(), out["ghi"].max())
    log.info("Annual mean temperature: %.1f degC", out["temp_air"].mean())
    log.info("Annual mean humidity:    %.1f %%",  out["humidity"].mean())
    log.info("Annual mean wind:        %.2f m/s", out["wind_speed"].mean())
    log.info("Annual total precip:     %.0f mm",  out["precip_mm"].sum())
    log.info("Monthly mean cloud cover (fraction):")
    for ts, v in out["cloud_cover"].resample("ME").mean().items():
        log.info("  %s : %.2f", ts.strftime("%Y-%m"), v)


if __name__ == "__main__":
    main()
