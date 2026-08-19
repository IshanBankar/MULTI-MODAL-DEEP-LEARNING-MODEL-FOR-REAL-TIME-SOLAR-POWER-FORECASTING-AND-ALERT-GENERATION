"""Synthetic INSAT-3D-like satellite imagery generator for the Pune ROI.

Reads pune_500kw_hourly.csv and generates one 64x64x3 RGB patch per timestamp.
Each patch reflects:
  - that hour's cloud_cover (coverage fraction)
  - solar zenith (overall illumination; night = dark)
  - month (season-dependent cloud texture and atmospheric haze)

Background terrain is a fixed Pune ROI: Western Ghats (W) -> urban Pune ->
plain (E). It does not change between hours - only clouds, illumination, and
haze do.

Outputs:
  data/satellite_patches.npz  - images (N,64,64,3) uint8 + timestamps
  data/sample_seasons.png     - 4-season x 5-cloud_cover preview grid
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

H = W = 64

# ---------------------------------------------------------------------------
# Season parameters
#   sigma            - Gaussian blur radius for cloud-noise field.
#                      Larger sigma -> bigger, smoother cloud blobs (monsoon
#                      stratiform). Smaller sigma -> wispy fragmented clouds
#                      (winter cirrus).
#   cloud_brightness - peak reflectance of cloud tops (monsoon clouds are
#                      thicker so they hit higher peak brightness).
#   haze             - additive atmospheric haze (heavy in pre-monsoon dust).
# ---------------------------------------------------------------------------
SEASON_PARAMS = {
    "winter":       {"name": "Winter (Dec-Feb)",      "sigma": 1.5, "cloud_brightness": 0.85, "haze": 0.05},
    "pre_monsoon":  {"name": "Pre-monsoon (Mar-May)", "sigma": 2.5, "cloud_brightness": 0.92, "haze": 0.22},
    "monsoon":      {"name": "Monsoon (Jun-Sep)",     "sigma": 6.0, "cloud_brightness": 0.95, "haze": 0.12},
    "post_monsoon": {"name": "Post-monsoon (Oct-Nov)","sigma": 3.0, "cloud_brightness": 0.88, "haze": 0.05},
}


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def _smooth_noise(shape, sigma, rng):
    """Spatially-correlated noise field via Gaussian-blurred white noise."""
    raw = rng.standard_normal(shape)
    s = gaussian_filter(raw, sigma=sigma)
    s -= s.min()
    if s.max() > 0:
        s /= s.max()
    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixed terrain background for Pune ROI
# ---------------------------------------------------------------------------
def make_terrain_bg(seed: int = 0):
    """Deterministic 64x64x3 terrain background centred on Pune.

    Western Ghats ridge along x ~ 0.15 (forest, dark green).
    Pune urban cluster around (0.55, 0.45) (slightly brighter, less green).
    Eastern plain dominated by farmland (mid brightness).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)

    ghats = 0.7 * np.exp(-((xx - 0.15) ** 2) / 0.02)
    veg = _smooth_noise((H, W), sigma=2.5, rng=rng) * 0.18
    urban = 0.18 * np.exp(-((xx - 0.55) ** 2 + (yy - 0.45) ** 2) / 0.004)

    base = 0.22 + 0.12 * (1.0 - ghats) - 0.10 * ghats + veg + urban

    rgb = np.empty((H, W, 3), dtype=np.float32)
    rgb[..., 0] = base * 0.95                         # R
    rgb[..., 1] = base * 1.02 + (1.0 - ghats) * 0.02  # G - vegetation greener
    rgb[..., 2] = base * 0.88 + urban * 0.05          # B - urban a touch bluer
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-timestamp patch synthesis
# ---------------------------------------------------------------------------
def synth_patch(cloud_cover: float, zenith_deg: float, month: int,
                terrain_bg: np.ndarray, master_seed: int, ts_value: int) -> np.ndarray:
    """Generate one 64x64x3 patch for the given conditions.

    Stable per-timestamp seeding -> same (master_seed, ts_value) always yields
    the same image. ts_value should be unix-seconds modulo 2**32.
    """
    rng = np.random.default_rng((master_seed * 2654435761 + ts_value) & 0xFFFFFFFF)
    season = month_to_season(month)
    p = SEASON_PARAMS[season]

    # Sun above horizon? Visible band sees almost nothing at night.
    cos_z = max(0.0, float(np.cos(np.radians(zenith_deg))))
    if cos_z < 0.05:
        return np.zeros((H, W, 3), dtype=np.float32)

    # Cloud noise field with season-appropriate spatial scale
    cloud_field = _smooth_noise((H, W), sigma=p["sigma"], rng=rng)

    if cloud_cover > 0.005:
        thresh = float(np.quantile(cloud_field, 1.0 - cloud_cover))
        cloud = np.clip((cloud_field - thresh) / max(1.0 - thresh, 1e-3), 0.0, 1.0)
    else:
        cloud = np.zeros_like(cloud_field)
    cloud = gaussian_filter(cloud, sigma=0.7) * p["cloud_brightness"]

    # Composite: clouds occlude terrain
    cloud_3 = cloud[..., None]
    bg_lit = terrain_bg * cos_z
    cloud_lit = cloud_3 * cos_z
    img = bg_lit * (1.0 - cloud_3) + cloud_lit

    # Atmospheric haze (washes the image out)
    haze = p["haze"] * cos_z
    img = img * (1.0 - haze) + haze * 0.55

    # Sensor noise
    img = img + rng.normal(0, 0.012, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Preview grid: each season x several cloud_cover levels
# ---------------------------------------------------------------------------
def make_preview(terrain_bg: np.ndarray, master_seed: int, out_path: Path):
    import matplotlib.pyplot as plt

    cc_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    seasons = list(SEASON_PARAMS.keys())
    n_rows = len(seasons)
    n_cols = len(cc_levels)

    # Solar-noon zenith approximations for Pune (18.6N) per season
    noon_zenith = {"winter": 42.0, "pre_monsoon": 18.0,
                   "monsoon": 12.0, "post_monsoon": 30.0}
    month_for_season = {"winter": 1, "pre_monsoon": 4,
                        "monsoon": 7, "post_monsoon": 11}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    for r, s in enumerate(seasons):
        for c, cc in enumerate(cc_levels):
            patch = synth_patch(cc, noon_zenith[s], month_for_season[s],
                                terrain_bg, master_seed, ts_value=r * 100 + c)
            ax = axes[r, c]
            ax.imshow(patch)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"cloud={cc:.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(SEASON_PARAMS[s]["name"], fontsize=10)

    plt.suptitle("Synthetic satellite patches over Pune (solar noon)\n"
                 "rows = seasons, columns = cloud_cover fraction", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    parser.add_argument("--out", default="data/satellite_patches.npz")
    parser.add_argument("--preview", default="data/sample_seasons.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("sat_gen")

    log.info("Reading %s", args.csv)
    df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    log.info("Have %d timestamps", len(df))

    log.info("Building Pune ROI terrain background")
    terrain = make_terrain_bg(seed=args.seed)

    log.info("Generating %d patches", len(df))
    out = np.empty((len(df), H, W, 3), dtype=np.uint8)
    months = df.index.month.to_numpy()
    cc_arr = df["cloud_cover"].to_numpy()
    z_arr = df["zenith_deg"].to_numpy()
    ts_int = (df.index.astype("int64").to_numpy() // 10**9) % (2**32)

    for i in range(len(df)):
        patch = synth_patch(
            cloud_cover=float(cc_arr[i]),
            zenith_deg=float(z_arr[i]),
            month=int(months[i]),
            terrain_bg=terrain,
            master_seed=args.seed,
            ts_value=int(ts_int[i]),
        )
        out[i] = (patch * 255.0).astype(np.uint8)
        if (i + 1) % 1500 == 0:
            log.info("  %d / %d", i + 1, len(df))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving to %s", out_path)
    np.savez_compressed(
        out_path,
        images=out,
        timestamps_unix=df.index.astype("int64").to_numpy() // 10**9,
        timestamps_iso=df.index.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
    )

    log.info("Writing preview grid -> %s", args.preview)
    make_preview(terrain, args.seed, Path(args.preview))

    # Verification stats
    mean_brightness = out.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    daytime = z_arr < 85
    if daytime.sum() > 100:
        corr = float(np.corrcoef(mean_brightness[daytime], cc_arr[daytime])[0, 1])
        log.info("Daytime corr(patch_brightness, cloud_cover) = %+.3f "
                 "(should be strongly positive)", corr)

    log.info("File size: %.1f MB", out_path.stat().st_size / 1e6)
    log.info("Per-month mean daytime brightness:")
    df_aux = pd.DataFrame({"brightness": mean_brightness, "z": z_arr},
                          index=df.index)
    monthly = df_aux[df_aux["z"] < 85]["brightness"].resample("ME").mean()
    for ts, v in monthly.items():
        log.info("  %s : %.3f", ts.strftime("%Y-%m"), v)


if __name__ == "__main__":
    main()
