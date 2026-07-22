"""
Extract CSIF all-sky SIF from csif.zip and write per-country monthly NetCDFs.

Source: Zhang et al. (2018), Science Advances.
        Reconstructed from OCO-2 SIF + MODIS reflectance + shortwave radiation.
        All-sky daily SIF (accounts for clouds and diurnal cycle).

Input:  data/raw/csif.zip
        Files inside: OCO2.SIF.all.daily.YYYY.nc (2001-2016)
        Grid: 0.5 deg global (360 lat x 720 lon, pixel centers at ±89.75)
        Dimension: doy (day-of-year, values 1, 5, 9, ..., 365 -- every 4 days, 92 composites)
        Variable: all_daily_sif [mW m-2 nm-1 sr-1 daily average, float32]
        Fill: -999.9

Process:
  1. For each country: clip to bbox using CSIF's 0.5° grid.
  2. Assign each 4-day composite to a calendar month based on its start doy.
  3. Average all composites in the same month -> 12 monthly means per year.
  4. Write csif_<cc>.nc with lats ordered north->south (descending) to match
     what aggregate_sif_by_crop.py's grid_transform function expects.

Resolution note:
  CSIF is 0.5 deg vs. GOSIF's 0.05 deg -- a 10x coarser grid. Small districts
  may contribute fewer or zero pixels (see WARNING output from aggregate_sif_by_crop).
  This is inherent to the dataset and a known trade-off vs. GOSIF.

Coverage: 2001-2016 (16 years; shorter than GOSIF's 2000-2024 span).

Output:  data/processed/csif_<cc>.nc
         dims: (time, lat, lon)
         var:  sif [mW m-2 nm-1 sr-1, daily mean averaged over calendar month, float32]
         time: monthly, first day of each month

Usage:
    python pipeline/extract_csif.py              # all countries
    python pipeline/extract_csif.py TH BD        # specific countries
"""

import re
import sys
import zipfile
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import netCDF4 as nc_lib
import pandas as pd
import xarray as xr

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country

BASE     = _PIPELINE.parent
ZIP_PATH = BASE / "data" / "raw" / "csif.zip"
OUT_DIR  = BASE / "data" / "processed"

FILL_RAW = -999.9


def list_zip_years(zf: zipfile.ZipFile) -> list[tuple[int, str]]:
    """Return sorted [(year, member_name)] for all.daily files only."""
    pattern = re.compile(r"OCO2\.SIF\.all\.daily\.(\d{4})\.nc$")
    hits = []
    for name in zf.namelist():
        m = pattern.match(name)
        if m:
            hits.append((int(m.group(1)), name))
    return sorted(hits)


def doy_to_month(year: int, doy: int) -> int:
    """Calendar month (1-12) for a given day-of-year in a specific year."""
    return (date(year, 1, 1) + timedelta(days=int(doy) - 1)).month


def extract(codes: list[str]) -> None:
    if not ZIP_PATH.exists():
        sys.exit(f"CSIF zip not found: {ZIP_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        years_files = list_zip_years(zf)
        if not years_files:
            sys.exit("No OCO2.SIF.all.daily.YYYY.nc files found in csif.zip")

        print(f"Found {len(years_files)} annual CSIF files "
              f"({years_files[0][0]}–{years_files[-1][0]})")

        # Read grid coordinates from first file
        with zf.open(years_files[0][1]) as f:
            raw = f.read()
        ds0 = nc_lib.Dataset("in-memory", memory=raw)
        lat_all = np.array(ds0.variables["lat"][:])
        lon_all = np.array(ds0.variables["lon"][:])
        ds0.close()

        # Per-country grid masks
        country_meta = {}
        for cc in codes:
            cfg = get_country(cc)
            lon_min, lat_min, lon_max, lat_max = cfg["bbox"]
            lat_mask = (lat_all >= lat_min) & (lat_all <= lat_max)
            lon_mask = (lon_all >= lon_min) & (lon_all <= lon_max)
            country_meta[cc] = {
                "cfg": cfg,
                "lat_mask": lat_mask,
                "lon_mask": lon_mask,
                "lats": lat_all[lat_mask],
                "lons": lon_all[lon_mask],
                "monthly": {},  # (year, month) -> np.ndarray shape (n_lat, n_lon)
            }
            print(f"  {cc} ({cfg['name']}): "
                  f"{lat_mask.sum()} lat x {lon_mask.sum()} lon pixels in bbox {cfg['bbox']}")

        # One pass through the zip: process all years, accumulate per country
        for i, (year, member) in enumerate(years_files):
            print(f"  [{i+1}/{len(years_files)}] Processing {year}...")

            with zf.open(member) as f:
                raw = f.read()
            ds = nc_lib.Dataset("in-memory", memory=raw)
            doy_vals = np.array(ds.variables["doy"][:])
            months_for_doy = np.array([doy_to_month(year, d) for d in doy_vals])

            sif_var = ds.variables["all_daily_sif"]

            for cc, meta in country_meta.items():
                lat_mask = meta["lat_mask"]
                lon_mask = meta["lon_mask"]

                # Clip: (n_doy, n_lat_cc, n_lon_cc)
                sif_clip = np.array(sif_var[:, lat_mask, :])[:, :, lon_mask].astype(np.float32)
                sif_clip[sif_clip <= FILL_RAW] = np.nan

                for month in range(1, 13):
                    idx = months_for_doy == month
                    if not idx.any():
                        # Some months may have no composites in short/leap-year edge cases
                        result = np.full(sif_clip.shape[1:], np.nan, dtype=np.float32)
                    else:
                        with np.errstate(all="ignore"):
                            result = np.nanmean(sif_clip[idx, :, :], axis=0).astype(np.float32)
                        # All-NaN months (e.g. polar night) stay NaN -- fine
                    meta["monthly"][(year, month)] = result

            ds.close()

    # Write one NetCDF per country
    for cc, meta in country_meta.items():
        _write_nc(cc, meta)


def _write_nc(cc: str, meta: dict) -> None:
    cfg = meta["cfg"]
    lats_asc = meta["lats"]   # ascending (south → north) as stored in CSIF
    lons     = meta["lons"]

    keys  = sorted(meta["monthly"].keys())   # [(year, month), ...]
    times = pd.DatetimeIndex([pd.Timestamp(year=y, month=m, day=1) for y, m in keys])
    stack = np.stack([meta["monthly"][k] for k in keys], axis=0)  # (time, lat, lon)

    # Flip lats to descending (north → south) so grid_transform in
    # aggregate_sif_by_crop.py (which assumes lats[0] is northernmost) works correctly.
    lats_desc = lats_asc[::-1]
    stack     = stack[:, ::-1, :]

    ds = xr.Dataset(
        {"sif": (["time", "lat", "lon"], stack,
                 {"long_name": "Solar-induced chlorophyll fluorescence (CSIF all-sky daily)",
                  "units": "mW m-2 nm-1 sr-1",
                  "source": "Zhang et al. (2018), Science Advances, doi:10.1126/sciadv.aat5834",
                  "temporal_aggregation": "monthly mean of 4-day composites",
                  "fill_raw": FILL_RAW})},
        coords={
            "time": times,
            "lat":  ("lat", lats_desc, {"units": "degrees_north",
                                         "note": "descending (N→S) for rasterio compatibility"}),
            "lon":  ("lon", lons,      {"units": "degrees_east"}),
        },
        attrs={
            "title":       f"CSIF all-sky monthly SIF — {cfg['name']}",
            "country":     cc,
            "bbox":        str(cfg["bbox"]),
            "resolution":  "0.5 deg",
            "coverage":    f"{min(y for y,_ in meta['monthly'])}"
                           f"–{max(y for y,_ in meta['monthly'])}",
        },
    )

    out_path = OUT_DIR / f"csif_{cc.lower()}.nc"
    encoding = {"sif": {"dtype": "float32", "zlib": True, "complevel": 4,
                        "_FillValue": np.float32(np.nan)}}
    ds.to_netcdf(out_path, encoding=encoding, engine="netcdf4")
    size_mb = out_path.stat().st_size / 1e6
    n_times = len(times)
    print(f"  Wrote {out_path.name}  ({size_mb:.1f} MB)")
    print(f"  Shape: ({n_times}, {len(lats_desc)}, {len(lons)})  "
          f"lat {lats_desc[-1]:.2f}–{lats_desc[0]:.2f}  "
          f"lon {lons[0]:.2f}–{lons[-1]:.2f}")


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    extract(codes)


if __name__ == "__main__":
    main()
