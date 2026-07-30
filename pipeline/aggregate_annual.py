"""
Collapse multi-season stablebound stats + monthly SIF onto one annual grain
per (stable_id, crop, year).

Season-collapse rule (see conversation; verified empirically per crop):
  If an all-encompassing season label ("All Year" / "Annual" / "Calendar
  Year") is present in a (stable_id, year) group, use ONLY that row --
  it already IS the annual total, and summing it with the sub-season rows
  double-counts (confirmed for TH soybeans: "All Year" co-occurs with
  "Dry"+"Wet" in the same district-year 863 times).
  Otherwise, sum all season rows present -- absent an all-encompassing
  label, multiple season rows represent genuinely disjoint cropping
  cycles within the same calendar year (e.g. BD rice's Aman/Aus/Boro,
  VN rice's regional spring/summer/autumn/winter splits), not duplicate
  reporting of the same harvest.

Yield is recomputed as an intensive ratio (production / area) on the
collapsed annual totals -- never averaged from reported per-season yields,
per the stablebound README. Area preference: harvested area if reported,
else planted area (BD and VN rice paddy report planted area only).

SIF is collapsed using crop-calendar growing season windows, checked in two
tiers:
  1. Sacks et al. (data/raw/All_data_with_climate.csv) -- a peer-reviewed,
     published crop calendar. Covers rice (all three countries) and wheat
     (Bangladesh only).
  2. A researched-but-unverified fallback (data/raw/researched_crop_calendars.csv),
     compiled via LLM-assisted literature research (FAO/GIEWS, USDA FAS,
     CIMMYT, national ministries) for the crops absent from Sacks: TH
     soybeans; BD maize/soybean/groundnut; VN maize/cassava/soybean/
     sugarcane/groundnut. Citations in that file have not been independently
     verified against source documents -- treat with more caution than the
     Sacks-derived windows. Rows are keyed directly on the FEWS crop slug
     (not the canonical CROPGRIDs name) and unioned the same way as Sacks.
  3. Crops present in neither tier fall back to a flat 12-month average.

The `sif_season_window` column in the output records which tier produced the
window, so downstream consumers (e.g. the dashboard) can distinguish
published-source windows from researched/unverified ones.

late_reporting rows are dropped (no stable-group geometry to join against).

Usage:
    python pipeline/aggregate_annual.py             # all countries
    python pipeline/aggregate_annual.py TH BD       # specific countries

Output:
    data/processed/annual_yield_<cc>.csv
        stable_id, crop, year, production_mt, area_ha, area_type, yield_mt_ha
    data/processed/annual_sif_<cc>.csv
        stable_id, crop, year, sif_annual_mean, n_months, sif_season_window
    data/processed/annual_joined_<cc>.csv
        the two above merged on (stable_id, crop, year)
"""

import sys
from pathlib import Path

import pandas as pd

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country

BASE     = _PIPELINE.parent
PROC_DIR = BASE / "data" / "processed"
CALENDAR_PATH = BASE / "data" / "raw" / "All_data_with_climate.csv"
RESEARCHED_CALENDAR_PATH = BASE / "data" / "raw" / "researched_crop_calendars.csv"

ALL_ENCOMPASSING = {"all year", "annual", "calendar year"}

# Maps canonical CROPGRIDs crop names to the crop calendar's Crop field.
# Crops absent from this dict fall back to the 12-month mean.
_CANONICAL_TO_CALENDAR = {
    "rice":      "Rice",
    "wheat":     "Wheat",
    "maize":     "Maize",
    "sugarcane": "Sugarcane",
}


def _load_growing_season_months(
    country_name: str, canonical_crop: str, calendar_df: pd.DataFrame
) -> set[int] | None:
    """
    Return the set of calendar months (1-12) active during the growing season
    for a given country + canonical crop, or None if not in the calendar.

    Multi-season crops: takes the union of active months across all season rows
    for that country-crop combination, so the returned set covers every month
    during which *any* season of that crop might be in the field.
    """
    cal_crop = _CANONICAL_TO_CALENDAR.get(canonical_crop)
    if cal_crop is None:
        return None

    rows = calendar_df[
        calendar_df["Location"].str.contains(country_name, case=False, na=False)
        & (calendar_df["Crop"] == cal_crop)
    ]
    if rows.empty:
        return None

    months: set[int] = set()
    for _, row in rows.iterrows():
        try:
            plant_m   = int(str(row["Plant.start.date"]).split("/")[0])
            harvest_m = int(str(row["Harvest.end.date"]).split("/")[0])
        except (ValueError, IndexError):
            continue

        if harvest_m >= plant_m:
            months.update(range(plant_m, harvest_m + 1))
        else:
            # Season wraps across the year boundary (e.g. plant Nov, harvest Feb)
            months.update(range(plant_m, 13))
            months.update(range(1, harvest_m + 1))

    return months if months else None


def _load_researched_season_months(
    country_name: str, fews_crop: str, researched_df: pd.DataFrame
) -> set[int] | None:
    """
    Fallback tier for crops absent from the Sacks calendar: an LLM-researched,
    unverified calendar (data/raw/researched_crop_calendars.csv). Keyed
    directly on the FEWS crop slug rather than the canonical CROPGRIDs name,
    since that file was compiled against this pipeline's own crop_map keys.
    Same wraparound-union logic as _load_growing_season_months.
    """
    if researched_df.empty:
        return None

    rows = researched_df[
        (researched_df["Country"] == country_name) & (researched_df["Crop"] == fews_crop)
    ]
    if rows.empty:
        return None

    months: set[int] = set()
    for _, row in rows.iterrows():
        plant_m   = int(row["Plant_start_month"])
        harvest_m = int(row["Harvest_end_month"])
        wraps     = str(row["Year_boundary_wrap"]).strip().lower() == "yes"

        if not wraps:
            months.update(range(plant_m, harvest_m + 1))
        else:
            months.update(range(plant_m, 13))
            months.update(range(1, harvest_m + 1))

    return months if months else None


# ---------------------------------------------------------------------------
# Season collapse
# ---------------------------------------------------------------------------

def collapse_seasons(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    df: rows of one variable (already filtered to one crop+measure), with
        columns stable_id, year, season, value.
    Returns one row per (stable_id, year) with `value_col` collapsed per
    the all-encompassing-label rule.
    """
    def _collapse(g: pd.DataFrame) -> float:
        is_annual = g["season"].str.lower().isin(ALL_ENCOMPASSING)
        if is_annual.any():
            return g.loc[is_annual, "value"].iloc[0]
        return g["value"].sum()

    out = (
        df.groupby(["stable_id", "year"])
        .apply(_collapse, include_groups=False)
        .reset_index(name=value_col)
    )
    return out


def load_measure(stats: pd.DataFrame, crop: str, measure: str) -> pd.DataFrame:
    var = f"{crop}_{measure}"
    sub = stats[(stats["variable"] == var) & (~stats["late_reporting"])]
    if sub.empty:
        return pd.DataFrame(columns=["stable_id", "year", measure])
    return collapse_seasons(sub[["stable_id", "year", "season", "value"]], measure)


# ---------------------------------------------------------------------------
# Per-country processing
# ---------------------------------------------------------------------------

def process_country(cc: str) -> None:
    print(f"\n=== {cc}")
    cfg = get_country(cc)
    stats = pd.read_csv(cfg["stats_aggregated_path"])
    sif_path = PROC_DIR / f"sif_by_crop_{cc.lower()}.csv"
    sif = pd.read_csv(sif_path) if sif_path.exists() else None

    crops = sorted(sif["crop"].unique()) if sif is not None else []
    if not crops:
        print("  No SIF crop file found, skipping.")
        return

    calendar_df    = pd.read_csv(CALENDAR_PATH) if CALENDAR_PATH.exists() else pd.DataFrame()
    researched_df  = pd.read_csv(RESEARCHED_CALENDAR_PATH) if RESEARCHED_CALENDAR_PATH.exists() else pd.DataFrame()
    crop_map    = cfg["crop_map"]  # fews_crop -> canonical_crop

    yield_rows = []
    for crop in crops:
        prod = load_measure(stats, crop, "production_mt")
        harv = load_measure(stats, crop, "area_harvested_ha")
        plnt = load_measure(stats, crop, "area_planted_ha")

        if prod.empty:
            print(f"  {crop}: no production_mt rows, skipping")
            continue

        merged = prod.merge(harv, on=["stable_id", "year"], how="left")
        merged = merged.merge(plnt, on=["stable_id", "year"], how="left")

        merged["area_ha"] = merged["area_harvested_ha"]
        merged["area_type"] = "harvested"
        use_planted = merged["area_ha"].isna() & merged["area_planted_ha"].notna()
        merged.loc[use_planted, "area_ha"] = merged.loc[use_planted, "area_planted_ha"]
        merged.loc[use_planted, "area_type"] = "planted"

        safe_area = merged["area_ha"].replace(0, pd.NA)
        merged["yield_mt_ha"] = merged["production_mt"] / safe_area
        merged["crop"] = crop

        n_no_area = merged["area_ha"].isna().sum()
        print(f"  {crop}: {len(merged)} district-years "
              f"({merged['area_type'].value_counts().to_dict()}, "
              f"{n_no_area} missing area -> yield NaN)")

        yield_rows.append(merged[["stable_id", "crop", "year",
                                   "production_mt", "area_ha", "area_type", "yield_mt_ha"]])

    yield_df = pd.concat(yield_rows, ignore_index=True) if yield_rows else pd.DataFrame()
    yield_out = PROC_DIR / f"annual_yield_{cc.lower()}.csv"
    yield_df.to_csv(yield_out, index=False)
    print(f"  Wrote {yield_out.name}: {len(yield_df)} rows")

    sif_annual_parts = []
    for fews_crop in sorted(sif["crop"].unique()):
        canonical  = crop_map.get(fews_crop, fews_crop)
        gs_months  = _load_growing_season_months(cfg["name"], canonical, calendar_df)

        if gs_months:
            source_tag = "Sacks et al."
        else:
            gs_months  = _load_researched_season_months(cfg["name"], fews_crop, researched_df)
            source_tag = "researched, unverified"

        crop_sif = sif[sif["crop"] == fews_crop].copy()
        if gs_months:
            crop_sif = crop_sif[crop_sif["month"].isin(gs_months)]
            season_label = f"months {sorted(gs_months)} ({source_tag})"
        else:
            season_label = "all 12 months (no calendar entry)"

        agg = (
            crop_sif.groupby(["stable_id", "crop", "year"])["sif_weighted"]
            .agg(sif_annual_mean="mean", n_months="count")
            .reset_index()
        )
        agg["sif_season_window"] = season_label
        sif_annual_parts.append(agg)
        print(f"  SIF window for {fews_crop} ({canonical}): {season_label}")

    sif_annual = (
        pd.concat(sif_annual_parts, ignore_index=True)
        if sif_annual_parts else pd.DataFrame(
            columns=["stable_id", "crop", "year", "sif_annual_mean", "n_months", "sif_season_window"]
        )
    )
    sif_out = PROC_DIR / f"annual_sif_{cc.lower()}.csv"
    sif_annual.to_csv(sif_out, index=False)
    print(f"  Wrote {sif_out.name}: {len(sif_annual)} rows")

    joined = yield_df.merge(sif_annual, on=["stable_id", "crop", "year"], how="inner")
    joined_out = PROC_DIR / f"annual_joined_{cc.lower()}.csv"
    joined.to_csv(joined_out, index=False)
    print(f"  Wrote {joined_out.name}: {len(joined)} rows "
          f"({joined['yield_mt_ha'].notna().sum()} with valid yield)")


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    for cc in codes:
        process_country(cc)


if __name__ == "__main__":
    main()
