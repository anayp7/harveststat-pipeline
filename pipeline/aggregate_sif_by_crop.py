"""
Crop-fraction-weighted SIF aggregation to stablebound district units.

Why: GOSIF pixels (0.05 deg) mix the photosynthetic signal of every land
cover in the pixel -- crop of interest, other crops, forest, fallow.
CROPGRIDs gives, per pixel, the physical area (ha) actually planted with a
given crop. Weighting SIF by that area before spatial aggregation recovers
the share of the pixel's signal attributable to the crop being studied.

    SIF_crop(district, month) = sum_px(SIF_px * croparea_px) / sum_px(croparea_px)

restricted to pixels inside the district polygon. This must happen pixel-by-
pixel before aggregating to the district -- never average SIF to a district
mean first and multiply by a single district-level fraction afterward.

Caveats (accepted for this exercise, see conversation):
- CROPGRIDs is a single ~2020 snapshot applied to a 2000-2024 SIF series;
  assumes cropland geography hasn't shifted much over that window.
- croparea (physical extent) is used, not harvarea (harvested area), since
  SIF is an instantaneous/monthly signal and harvarea double-counts
  multi-season fields.

Boundary geometry note: stablebound's per-year geojsons are NOT equivalent
snapshots of the same partition -- they are different granularities of
merging. The base-year file (the lineage's earliest year: TH 1978, BD 1983,
VN 1980) is the coarsest, most-merged grouping -- the one that stays valid
for the *entire* analysis window -- while later-year files show finer
partitions only valid from that later year onward (as more administrative
splits have already resolved). `stats_aggregated.csv` confirms this: BD has
22 stable_ids total (21 base groups + 1 raw late-reporting unit_id), matching
n_stable_groups_at_base, not the 64 features in stable_1985.geojson. So we
use ONLY the base-year boundary file per country and rasterize each district
once; those per-district pixel weights are reused across the whole SIF time
series instead of re-rasterizing per year.

Rows flagged late_reporting in stats_aggregated.csv use a raw unit_id (not
a stable group) and generally have no geometry in the base-year stable file;
they are out of scope for this SIF join and should be handled separately.

Usage:
    python pipeline/aggregate_sif_by_crop.py             # all countries
    python pipeline/aggregate_sif_by_crop.py TH BD       # specific countries

Output:
    data/processed/sif_by_crop_<cc>.csv
    columns: stable_id, crop, year, month, sif_weighted, n_pixels, croparea_ha
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country

BASE    = _PIPELINE.parent
CG_DIR  = BASE / "data" / "raw" / "cropgrids"
SIF_DIR = BASE / "data" / "processed"
OUT_DIR = BASE / "data" / "processed"


# ---------------------------------------------------------------------------
# Boundary merging
# ---------------------------------------------------------------------------

def load_base_boundaries(cc: str) -> gpd.GeoDataFrame:
    """The single base-year stable boundary file -- the canonical partition
    that stats_aggregated.csv's stable_id column actually uses."""
    gdf = gpd.read_file(get_country(cc)["boundary_path"])
    return gdf[["stable_id", "geometry"]]


# ---------------------------------------------------------------------------
# Grid alignment helpers
# ---------------------------------------------------------------------------

def grid_transform(lats: np.ndarray, lons: np.ndarray):
    """Affine transform for a regular lat/lon grid given pixel-center coords."""
    res = abs(lons[1] - lons[0])
    west = lons[0] - res / 2
    north = lats[0] + res / 2  # lats assumed descending (north -> south)
    return from_origin(west, north, res, res)


def rasterize_district(geom, transform, shape) -> np.ndarray:
    """Boolean mask of pixels whose center falls inside `geom`."""
    mask = features.rasterize(
        [(geom, 1)], out_shape=shape, transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    return mask.astype(bool)


def slice_cropgrid_to_bbox(cg_da: xr.DataArray, lat_target: np.ndarray, lon_target: np.ndarray) -> np.ndarray:
    """Pull the CROPGRIDs croparea values onto the exact SIF grid window via nearest-index alignment."""
    return cg_da.sel(lat=lat_target, lon=lon_target, method="nearest").values


# ---------------------------------------------------------------------------
# Main per-country routine
# ---------------------------------------------------------------------------

def process_country(cc: str) -> None:
    print(f"\n=== {cc}")
    cfg = get_country(cc)
    crop_map = cfg["crop_map"]

    sif_ds = xr.open_dataset(SIF_DIR / f"csif_{cc.lower()}.nc")
    lats = sif_ds["lat"].values
    lons = sif_ds["lon"].values
    transform = grid_transform(lats, lons)
    shape = (len(lats), len(lons))
    print(f"  SIF grid: {shape}, {len(sif_ds.time)} months")

    boundaries = load_base_boundaries(cc)
    print(f"  Boundaries: {len(boundaries)} base-year ({cfg['base_year']}) stable_id districts")

    # Pre-rasterize each district once (reused across all crops & months)
    district_masks = {}
    for _, row in boundaries.iterrows():
        district_masks[row["stable_id"]] = rasterize_district(row["geometry"], transform, shape)
    n_empty = sum(not m.any() for m in district_masks.values())
    if n_empty:
        print(f"  WARNING: {n_empty} districts have zero pixels in the SIF grid "
              f"(too small relative to 0.5deg resolution)")

    sif_values = sif_ds["sif"].values   # (time, lat, lon)
    time_index = pd.to_datetime(sif_ds["time"].values)
    sif_ds.close()

    all_rows = []

    for fews_crop, cg_crop in crop_map.items():
        cg_path = CG_DIR / f"CROPGRIDSv1.08_{cg_crop}.nc"
        if not cg_path.exists():
            print(f"  SKIP {fews_crop} -> {cg_crop}: file not found ({cg_path.name})")
            continue

        cg_ds = xr.open_dataset(cg_path)
        croparea_full = cg_ds["croparea"]
        croparea = slice_cropgrid_to_bbox(croparea_full, lats, lons)
        croparea = np.nan_to_num(croparea, nan=0.0)
        croparea[croparea < 0] = 0.0  # ocean sentinel (-1) and any other negatives
        cg_ds.close()

        print(f"  {fews_crop} -> {cg_crop}: total crop area in bbox = {croparea.sum():,.0f} ha")

        for stable_id, mask in district_masks.items():
            weight = croparea * mask
            total_weight = weight.sum()
            if total_weight <= 0:
                continue  # no mapped crop area in this district; nothing to report

            # weighted mean SIF per month: sum(SIF*w) / sum(w), vectorized over time
            flat_w = weight.ravel()
            nz = flat_w > 0
            w_nz = flat_w[nz]
            sif_flat = sif_values.reshape(sif_values.shape[0], -1)[:, nz]
            valid = np.isfinite(sif_flat)
            # treat NaN SIF pixels as zero-weight for that month only
            w_matrix = np.where(valid, w_nz[None, :], 0.0)
            num = np.nansum(sif_flat * w_matrix, axis=1)
            den = w_matrix.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sif_weighted = np.where(den > 0, num / den, np.nan)

            n_pixels = int(nz.sum())
            for t, val in zip(time_index, sif_weighted):
                all_rows.append((stable_id, fews_crop, t.year, t.month,
                                  val, n_pixels, total_weight))

    out_df = pd.DataFrame(all_rows, columns=[
        "stable_id", "crop", "year", "month", "sif_weighted", "n_pixels", "croparea_ha"
    ])
    out_path = OUT_DIR / f"sif_by_crop_{cc.lower()}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name}: {len(out_df)} rows, "
          f"{out_df['stable_id'].nunique()} districts, {out_df['crop'].nunique()} crops")


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    for cc in codes:
        process_country(cc)


if __name__ == "__main__":
    main()
