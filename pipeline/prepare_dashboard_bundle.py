"""
Precompute a small, committable bundle so the dashboard runs from a clean
`git clone` -- with no stablebound checkout present.

The dashboard otherwise reaches into stablebound_starter/ for three things,
all of which are gitignored (~40 MB, and stats_aggregated.csv carries the
proprietary FEWS yield rows we deliberately do not publish):

    boundary geojson      -> every choropleth
    stats_aggregated.csv  -> season-level yield for the click-through drilldown
    name_history.csv      -> admin-name -> stable_id map for QA exclusions

This script distills each into a compact derived form under data/dashboard/:

    boundaries_<cc>.geojson   simplified geometry (~95% smaller, ~1 km
                              tolerance -- ample for province choropleths)
    season_yield_<cc>.csv     season-level yield per (stable_id, crop,
                              season, year), already collapsed
    exclusions_<cc>.csv       the QA exclusion sets, resolved to stable_ids
                              (blank year = whole series excluded)

Only derived values ship -- never stats_aggregated.csv itself. Re-run this
whenever the upstream stablebound outputs or QA CSVs change.

Usage:
    python pipeline/prepare_dashboard_bundle.py          # all countries
    python pipeline/prepare_dashboard_bundle.py BD       # one country
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country
from sif_yield_season import (
    SEASON_MONTH_OFFSETS, _load_season_yield, _build_exclusions,
)

BASE       = _PIPELINE.parent
BUNDLE_DIR = BASE / "data" / "dashboard"

# ~0.01 deg ~= 1 km. Province polygons stay visually identical at the zoom
# levels the dashboard uses; preserve_topology keeps borders from cracking.
SIMPLIFY_TOLERANCE = 0.01


def build_boundaries(cc: str) -> None:
    cfg = get_country(cc)
    src = cfg["boundary_path"]
    if not src.exists():
        print(f"  boundaries: SKIP (missing {src.name})")
        return

    gdf = gpd.read_file(src)[["stable_id", "geometry"]]
    before = src.stat().st_size / 1e6
    gdf["geometry"] = gdf.geometry.simplify(
        SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    out = BUNDLE_DIR / f"boundaries_{cc.lower()}.geojson"
    gdf.to_file(out, driver="GeoJSON")
    after = out.stat().st_size / 1e6
    print(f"  boundaries: {len(gdf)} districts, "
          f"{before:.1f} MB -> {after:.2f} MB ({100 * (1 - after / before):.0f}% smaller)")


def build_season_yield(cc: str) -> None:
    """Season-level yield for every (crop, season) the dashboard can drill into."""
    cfg = get_country(cc)
    if not cfg["stats_aggregated_path"].exists():
        print(f"  season yield: SKIP (no stats_aggregated.csv)")
        return

    keys = [(crop, season) for (country, crop, season) in SEASON_MONTH_OFFSETS
            if country == cc]
    parts = []
    for crop, season in keys:
        y = _load_season_yield(cfg, crop, season)
        if y.empty:
            continue
        y = y.copy()
        y["crop"], y["season"] = crop, season
        parts.append(y[["stable_id", "crop", "season", "year", "yield_mt_ha"]])

    out = BUNDLE_DIR / f"season_yield_{cc.lower()}.csv"
    if not parts:
        pd.DataFrame(columns=["stable_id", "crop", "season", "year", "yield_mt_ha"]) \
            .to_csv(out, index=False)
        print("  season yield: no rows")
        return

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"  season yield: {len(df)} rows across {df['season'].nunique()} season(s), "
          f"{out.stat().st_size / 1e6:.2f} MB")


def build_exclusions(cc: str) -> None:
    """Flatten the QA exclusion sets so the dashboard needn't re-derive them
    (which would need name_history.csv to map admin names to stable_ids)."""
    try:
        points, series = _build_exclusions(cc)
    except FileNotFoundError as e:
        print(f"  exclusions: SKIP ({e})")
        return

    rows = [{"stable_id": s, "crop": c, "year": y} for (s, c, y) in sorted(points)]
    rows += [{"stable_id": s, "crop": c, "year": ""} for (s, c) in sorted(series)]

    out = BUNDLE_DIR / f"exclusions_{cc.lower()}.csv"
    pd.DataFrame(rows, columns=["stable_id", "crop", "year"]).to_csv(out, index=False)
    print(f"  exclusions: {len(points)} point(s), {len(series)} whole series")


def process_country(cc: str) -> None:
    print(f"\n=== {cc}")
    build_boundaries(cc)
    build_season_yield(cc)
    build_exclusions(cc)


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for cc in codes:
        process_country(cc)

    total = sum(p.stat().st_size for p in BUNDLE_DIR.glob("*")) / 1e6
    print(f"\nBundle written to {BUNDLE_DIR.relative_to(BASE)} — {total:.2f} MB total")
    print("Commit this directory so collaborators can run the dashboard "
          "without a stablebound checkout.")


if __name__ == "__main__":
    main()
