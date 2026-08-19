"""Digital-twin data generator for a 500 kW PV plant in Pune, India.

Implements: synthetic Pune climatology -> pvlib physics -> realistic noise ->
labelled anomaly injection. Output mirrors what a real SCADA historian +
co-located weather station would record at hourly resolution.

Plant spec follows the paper: 18.6 deg N, 73.8 deg E, 500 kW utility-scale.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# ---------------------------------------------------------------------------
# Site + Plant configuration
# ---------------------------------------------------------------------------
LATITUDE = 18.6
LONGITUDE = 73.8
ALTITUDE = 560        # metres (Pune approx)
TZ = "Asia/Kolkata"

PLANT_DC_KW = 500.0   # nameplate DC capacity
INVERTER_AC_KW = 500.0
INVERTER_EFF = 0.97
TILT = 18.0           # near-latitude fixed tilt
AZIMUTH = 180.0       # south-facing
DC_LOSS = 0.05        # cabling, mismatch
GAMMA_PDC = -0.0035   # power temp coefficient (1/degC)
BASE_SOILING = 0.02   # 2% baseline soiling


# ---------------------------------------------------------------------------
# Step 1: Synthesise Pune-climatology weather
# ---------------------------------------------------------------------------
def synthesise_weather(start: str, end: str, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
    """Hourly weather following Pune seasonal patterns.

    Pune climate phases:
      - Winter (Dec-Feb): cool, dry, clear skies
      - Pre-monsoon (Mar-May): hot, dry
      - Monsoon (Jun-Sep): warm, humid, heavily clouded
      - Post-monsoon (Oct-Nov): mild, clearing
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq=freq, tz=TZ)
    doy = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    month = times.month.to_numpy()

    monsoon = ((month >= 6) & (month <= 9)).astype(float)

    # Air temperature (degC): seasonal peak ~May, diurnal peak ~14:00
    seasonal_T = 25.0 + 7.0 * np.sin(2 * np.pi * (doy - 100) / 365.0)
    diurnal_T = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temp_air = seasonal_T + diurnal_T + rng.normal(0, 1.0, len(times))

    # Relative humidity (%): monsoon spike, anti-correlated with mid-day temp
    humidity = 45.0 + 35.0 * monsoon - 8.0 * np.sin(2 * np.pi * (hour - 14) / 24.0)
    humidity = humidity + rng.normal(0, 5, len(times))
    humidity = np.clip(humidity, 10.0, 100.0)

    # Wind speed (m/s): light, slightly stronger pre-monsoon
    wind = np.abs(rng.normal(2.5, 1.0, len(times)))

    # Cloud cover fraction [0,1]: AR(1) for temporal persistence
    cc_target = 0.20 + 0.60 * monsoon + 0.10 * (month == 10)
    cloud = np.zeros(len(times))
    cloud[0] = cc_target[0]
    for i in range(1, len(times)):
        cloud[i] = 0.85 * cloud[i - 1] + 0.15 * cc_target[i] + rng.normal(0, 0.05)
    cloud = np.clip(cloud, 0.0, 1.0)

    return pd.DataFrame(
        {
            "temp_air": temp_air,
            "humidity": humidity,
            "wind_speed": wind,
            "cloud_cover": cloud,
        },
        index=times,
    )


# ---------------------------------------------------------------------------
# Step 2: Solar geometry + cloud-attenuated irradiance via pvlib
# ---------------------------------------------------------------------------
def compute_irradiance(weather: pd.DataFrame):
    """Clear-sky GHI/DNI/DHI, then attenuated by cloud cover and decomposed."""
    site = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TZ, altitude=ALTITUDE)
    times = weather.index

    solpos = site.get_solarposition(times)
    cs = site.get_clearsky(times, model="ineichen")

    # Kasten-Czeplak transmittance from cloud cover fraction
    cc = weather["cloud_cover"].to_numpy()
    transmittance = 1.0 - 0.75 * np.power(cc, 3.4)

    ghi = cs["ghi"] * transmittance
    # Erbs decomposes GHI into DNI/DHI given solar zenith
    erbs = pvlib.irradiance.erbs(ghi, solpos["zenith"], times.dayofyear)
    dni = pd.Series(np.clip(np.asarray(erbs["dni"]), 0, None), index=ghi.index)
    dhi = pd.Series(np.clip(np.asarray(erbs["dhi"]), 0, None), index=ghi.index)
    return ghi, dni, dhi, solpos


def compute_poa(ghi, dni, dhi, solpos) -> pd.Series:
    """Plane-of-array irradiance via Hay-Davies transposition."""
    dni_extra = pvlib.irradiance.get_extra_radiation(ghi.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT,
        surface_azimuth=AZIMUTH,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="haydavies",
    )
    return poa["poa_global"].fillna(0.0).clip(lower=0)


# ---------------------------------------------------------------------------
# Step 3: PV physics -> AC power
# ---------------------------------------------------------------------------
def compute_power(poa_global: pd.Series, weather: pd.DataFrame) -> pd.Series:
    """POA -> cell temp -> PVWatts DC -> inverter clip."""
    t_cell = pvlib.temperature.pvsyst_cell(
        poa_global, weather["temp_air"], weather["wind_speed"]
    )
    g_eff = poa_global * (1.0 - BASE_SOILING)
    dc_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=g_eff,
        temp_cell=t_cell,
        pdc0=PLANT_DC_KW * 1000.0,
        gamma_pdc=GAMMA_PDC,
    )
    dc_w = dc_w * (1.0 - DC_LOSS)
    ac_w = np.minimum(dc_w * INVERTER_EFF, INVERTER_AC_KW * 1000.0)
    return (ac_w / 1000.0).clip(lower=0)  # kW


# ---------------------------------------------------------------------------
# Step 4: Slow soiling drift (reset by rain) + measurement noise
# ---------------------------------------------------------------------------
def add_soiling_drift(power: pd.Series, humidity: pd.Series,
                      decay_per_day: float = 0.001, seed: int = 42) -> pd.Series:
    """Drift downward, reset on rainy hours (humidity > 90%)."""
    rng = np.random.default_rng(seed)
    factor = np.ones(len(power))
    decay_per_hour = decay_per_day / 24.0
    for i in range(1, len(power)):
        factor[i] = factor[i - 1] * (1.0 - decay_per_hour)
        if humidity.iloc[i] > 90.0 and rng.random() < 0.10:
            factor[i] = 1.0
    return power * factor


def add_measurement_noise(power: pd.Series, sigma_pct: float = 0.015,
                          seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0, sigma_pct, len(power)) * power.to_numpy()
    return (power + noise).clip(lower=0)


# ---------------------------------------------------------------------------
# Step 5: Labelled anomaly injection
# ---------------------------------------------------------------------------
def inject_anomalies(power: pd.Series, n_inverter_trips: int = 8,
                     n_string_faults: int = 18, n_mppt: int = 14,
                     seed: int = 42):
    """Insert realistic plant faults; return modified power and string labels.

    Priority order (highest first): inverter_trip > string_fault > mppt_underperf.
    A lower-priority fault never overwrites a higher-priority label, and only
    modifies power on samples not already claimed by a higher-priority fault.
    """
    rng = np.random.default_rng(seed + 2)
    power_arr = power.to_numpy().astype(float).copy()
    labels = np.full(len(power), "normal", dtype=object)

    daylight_idx = np.where(power.to_numpy() > 5.0)[0]
    if len(daylight_idx) == 0:
        return pd.Series(power_arr, index=power.index), pd.Series(labels, index=power.index)

    def _pick_segments(n_events, dur_range):
        for _ in range(n_events):
            start = int(rng.choice(daylight_idx))
            dur = int(rng.integers(*dur_range))
            end = min(start + dur, len(power))
            yield start, end

    # 1. Inverter trips - zero out everything in segment, claim labels
    for s, e in _pick_segments(n_inverter_trips, (2, 7)):
        power_arr[s:e] = 0.0
        labels[s:e] = "inverter_trip"

    # 2. String faults - 15-35% drop on samples still 'normal'
    for s, e in _pick_segments(n_string_faults, (3, 10)):
        drop = rng.uniform(0.15, 0.35)
        seg_labels = labels[s:e]
        free = seg_labels == "normal"
        power_arr[s:e][free] *= (1.0 - drop)
        seg_labels[free] = "string_fault"
        labels[s:e] = seg_labels

    # 3. MPPT underperformance - 5-15% drop on samples still 'normal'
    for s, e in _pick_segments(n_mppt, (6, 18)):
        drop = rng.uniform(0.05, 0.15)
        seg_labels = labels[s:e]
        free = seg_labels == "normal"
        power_arr[s:e][free] *= (1.0 - drop)
        seg_labels[free] = "mppt_underperf"
        labels[s:e] = seg_labels

    return (
        pd.Series(power_arr, index=power.index),
        pd.Series(labels, index=power.index),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_real_weather(path: str) -> pd.DataFrame:
    """Load real ERA5 weather (from weather_fetcher.py output).

    Required columns: temp_air, humidity, wind_speed, cloud_cover.
    Optional columns: ghi, dni, dhi - if present, they bypass the
    clear-sky + cloud-attenuation step and feed pvlib's POA transposition
    directly with measured irradiance.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    required = {"temp_air", "humidity", "wind_speed", "cloud_cover"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"weather CSV missing columns: {sorted(missing)}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01", help="ISO start datetime")
    parser.add_argument("--end", default="2023-12-31 23:00", help="ISO end datetime")
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument("--weather-csv", default=None,
                        help="Real weather CSV from weather_fetcher.py "
                             "(if given, overrides synthetic Pune climatology)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("data_gen")

    real_irradiance = False
    if args.weather_csv:
        log.info("Loading real weather from %s", args.weather_csv)
        weather = _load_real_weather(args.weather_csv)
        # Trim to requested window if possible
        try:
            weather = weather.loc[args.start:args.end]
        except Exception:
            pass
        site = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TZ, altitude=ALTITUDE)
        solpos = site.get_solarposition(weather.index)
        if {"ghi", "dni", "dhi"}.issubset(weather.columns):
            log.info("Using real ERA5 GHI/DNI/DHI directly (bypassing clear-sky model)")
            ghi = weather["ghi"]
            dni = weather["dni"]
            dhi = weather["dhi"]
            real_irradiance = True
        else:
            log.info("Real weather lacks irradiance columns; falling back to clear-sky + cloud attenuation")
            ghi, dni, dhi, solpos = compute_irradiance(weather)
    else:
        log.info("Synthesising Pune weather %s -> %s", args.start, args.end)
        weather = synthesise_weather(args.start, args.end, seed=args.seed)
        log.info("Computing solar geometry + cloud-attenuated irradiance")
        ghi, dni, dhi, solpos = compute_irradiance(weather)

    log.info("Transposing GHI to plane-of-array (tilt=%.0f, az=%.0f)", TILT, AZIMUTH)
    poa = compute_poa(ghi, dni, dhi, solpos)

    log.info("Running PV physics -> AC power for %.0f kW plant%s",
             PLANT_DC_KW, " [real weather]" if args.weather_csv else " [synthetic weather]")
    power = compute_power(poa, weather)

    log.info("Adding soiling drift + measurement noise")
    power = add_soiling_drift(power, weather["humidity"], seed=args.seed)
    power = add_measurement_noise(power, seed=args.seed)

    log.info("Injecting labelled anomalies")
    power, labels = inject_anomalies(power, seed=args.seed)

    df = pd.DataFrame(
        {
            "power_kw": power.values,
            "ghi": ghi.values,
            "dni": dni.values,
            "dhi": dhi.values,
            "poa_global": poa.values,
            "temp_air": weather["temp_air"].values,
            "humidity": weather["humidity"].values,
            "wind_speed": weather["wind_speed"].values,
            "cloud_cover": weather["cloud_cover"].values,
            "zenith_deg": solpos["zenith"].values,
            "azimuth_solar_deg": solpos["azimuth"].values,
            "is_anomaly": (labels != "normal").values,
            "anomaly_type": labels.values,
        },
        index=power.index,
    )
    df.index.name = "timestamp"

    out_csv = out_dir / "pune_500kw_hourly.csv"
    df.to_csv(out_csv)
    log.info("Wrote %s (%d rows)", out_csv, len(df))

    # Brief summary so the user sees something sane immediately
    daytime = df[df["zenith_deg"] < 85]
    log.info("Daytime mean power: %.1f kW | peak: %.1f kW",
             daytime["power_kw"].mean(), df["power_kw"].max())
    log.info("Anomaly rate: %.2f%% (%d events flagged)",
             100.0 * df["is_anomaly"].mean(), int(df["is_anomaly"].sum()))
    log.info("Anomaly breakdown:\n%s", df["anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
