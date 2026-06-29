"""
Extract GOSIF v2 monthly SIF from Monthly.tar and write per-country NetCDFs.

Source:  Li, X., Xiao, J. (2019), Remote Sensing 11, 517.
         Global, OCO-2 based SIF, 0.05-deg monthly, 2000-2024.
         Files inside tar: Monthly/GOSIF_<YYYY>.M<MM>.tif.gz (int16, scale=0.0001)
         Fill value: -9999 (not stored in header — applied manually).

Outputs: data/processed/gosif_<cc>.nc  for each country in COUNTRIES.
         Dimensions: (time, lat, lon)
         Variable:   sif  [W m-2 um-1 sr-1, float32]
         Time:       monthly, first day of each month (numpy datetime64)

Usage:
    python pipeline/extract_gosif.py              # all countries
    python pipeline/extract_gosif.py TH BD        # specific countries
"""

import sys
import gzip
import io
import tarfile
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import xarray as xr
import pandas as pd

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE      = _PIPELINE.parent
TAR_PATH  = BASE / "Monthly.tar"
OUT_DIR   = BASE / "data" / "processed"

SCALE     = 0.0001   # GOSIF int16 → W m-2 um-1 sr-1
FILL_RAW  = -9999    # fill value (not stored in TIF header)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_tar_months(tar: tarfile.TarFile) -> list[tuple[int, int, str]]:
    """Return sorted [(year, month, member_name), ...] for all GOSIF tif.gz entries."""
    pattern = re.compile(r"GOSIF_(\d{4})\.M(\d{2})\.tif\.gz$")
    hits = []
    for m in tar.getmembers():
        match = pattern.search(m.name)
        if match:
            hits.append((int(match.group(1)), int(match.group(2)), m.name))
    return sorted(hits)


def read_clipped(tar: tarfile.TarFile, member_name: str, bbox: tuple) -> tuple[np.ndarray, dict]:
    """
    Read one GOSIF tif.gz from the open tar, clip to bbox, return (array, profile).
    array dtype: float32, fill pixels → NaN.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    raw_gz = tar.extractfile(member_name).read()
    tif_bytes = gzip.decompress(raw_gz)

    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        raw = src.read(1, window=window).astype(np.float32)
        profile = src.profile.copy()
        transform = src.window_transform(window)

    raw[raw == FILL_RAW] = np.nan
    sif = raw * SCALE
    sif[sif < -0.1] = np.nan   # values below -0.1 are not physical
    return sif, transform


def build_coords(transform, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre lat/lon arrays from a rasterio Affine transform."""
    nrows, ncols = shape
    lons = transform.c + (np.arange(ncols) + 0.5) * transform.a
    lats = transform.f + (np.arange(nrows) + 0.5) * transform.e
    return lats, lons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(cc: str) -> None:
    cfg = get_country(cc)
    bbox = cfg["bbox"]
    out_path = OUT_DIR / f"gosif_{cc.lower()}.nc"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {cc} ({cfg['name']}) bbox={bbox}")

    with tarfile.open(TAR_PATH, "r") as tar:
        months = list_tar_months(tar)
        print(f"  Found {len(months)} monthly files in tar ({months[0][:2]} – {months[-1][:2]})")

        slices = []
        times  = []

        for i, (year, month, name) in enumerate(months):
            if (i + 1) % 24 == 0 or i == 0:
                print(f"  Processing {year}.M{month:02d}  ({i+1}/{len(months)})")

            sif, transform = read_clipped(tar, name, bbox)

            if i == 0:
                lats, lons = build_coords(transform, sif.shape)

            slices.append(sif)
            times.append(pd.Timestamp(year=year, month=month, day=1))

    stack = np.stack(slices, axis=0)   # (time, lat, lon)
    time_index = pd.DatetimeIndex(times)

    ds = xr.Dataset(
        {"sif": (["time", "lat", "lon"], stack,
                 {"long_name": "Solar-induced chlorophyll fluorescence (GOSIF v2)",
                  "units": "W m-2 um-1 sr-1",
                  "source": "Li & Xiao (2019), doi:10.3390/rs11050517",
                  "scale_applied": SCALE,
                  "fill_raw": FILL_RAW})},
        coords={
            "time": time_index,
            "lat":  ("lat", lats,  {"units": "degrees_north"}),
            "lon":  ("lon", lons,  {"units": "degrees_east"}),
        },
        attrs={
            "title":   f"GOSIF v2 monthly SIF — {cfg['name']}",
            "country": cc,
            "bbox":    str(bbox),
        },
    )

    encoding = {"sif": {"dtype": "float32", "zlib": True, "complevel": 4,
                        "_FillValue": np.float32(np.nan)}}
    ds.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path.name}  ({size_mb:.1f} MB)")
    print(f"  Shape: {stack.shape}  lat {lats[0]:.3f}–{lats[-1]:.3f}  "
          f"lon {lons[0]:.3f}–{lons[-1]:.3f}")


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    for cc in codes:
        extract(cc)


if __name__ == "__main__":
    main()
