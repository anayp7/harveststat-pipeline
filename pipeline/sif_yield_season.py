"""
Season-specific SIF-yield correlation using CSIF and crop calendars.

For each (stable_id, crop, season) with >= MIN_YEARS overlapping years:
  1. Pull season-level yield directly from stats_aggregated.csv (not annual totals).
  2. Average CSIF over the crop's growing-season months from the crop calendar.
     Seasons that span a year boundary (Boro, Rabi, VN Winter/Spring) are
     handled with per-month year offsets relative to the yield year — see
     SEASON_MONTH_OFFSETS below for the assumed convention.
  3. Linearly detrend both series, compute signed Pearson r + p-value.
  4. Two scenarios: full (all years) and cleaned (same QA exclusions as
     sif_yield_variance.py: |z|>3 anomalies, hard repeat-run flags, mismatch
     >10%, absolute low-CV series).

Season → growing-window mapping (crop calendar audit, see pipeline discussion):

  TH rice  Wet   : May–Dec (planting year; Jan–Feb harvest tail dropped)
  TH rice  Dry   : Jan–Jun
  BD rice  Aus   : Mar–Aug
  BD rice  Aman  : May–Dec
  BD rice  Boro  : Nov–May  (Nov–Dec = prior year, Jan–May = yield year)
  BD wheat Rabi  : Nov–Apr  (Nov–Dec = prior year, Jan–Apr = yield year)
  BD wheat Annual: same as Rabi
  VN rice  Northern Summer/Autumn : Jun–Dec
  VN rice  Northern Winter/Spring : Nov–Jul  (Nov–Dec = prior year, Jan–Jul = yield year)
  VN rice  Southern Summer/Autumn : Jun–Dec
  VN rice  Southern Winter/Spring : Nov–May  (Nov–Dec = prior year, Jan–May = yield year)

Year-offset convention: yield_year is the calendar year in which the season
is primarily harvested (harvest-year attribution). For seasons that begin in
the preceding calendar year (Boro, Rabi, Winter/Spring), those earlier months
get year_offset = -1 relative to yield_year.

Output:
    data/processed/sif_yield_season_corr_<cc>.csv
        stable_id, crop, season, scenario, n_years, r, r_squared, p_value
    figures/<cc>/sif_yield_season/<crop>_<season>_<scenario>.png
        choropleth of signed r; p<0.05 districts outlined

Usage:
    python pipeline/sif_yield_season.py BD
    python pipeline/sif_yield_season.py            # all countries
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes, get_country

BASE     = _PIPELINE.parent
PROC_DIR = BASE / "data" / "processed"
FIG_DIR  = BASE / "figures"

MIN_YEARS            = 5
ANOMALY_REMOVE_Z     = 3.0
MISMATCH_THRESHOLD   = 10.0

# ---------------------------------------------------------------------------
# Season → growing-window definition
# Each entry: list of (month, year_offset) where year_offset is 0 (same
# calendar year as yield_year) or -1 (prior calendar year).
# Key: (cc, fews_crop, season_label_in_stats_aggregated)
# ---------------------------------------------------------------------------

SEASON_MONTH_OFFSETS = {
    # ---- Thailand ----
    ("TH", "rice", "Wet"):  [(m, 0) for m in [5,6,7,8,9,10,11,12]],
    ("TH", "rice", "Dry"):  [(m, 0) for m in [1,2,3,4,5,6]],

    # ---- Bangladesh rice ----
    ("BD", "rice_paddy", "Aus"):  [(m, 0) for m in [3,4,5,6,7,8]],
    ("BD", "rice_paddy", "Aman"): [(m, 0) for m in [5,6,7,8,9,10,11,12]],
    ("BD", "rice_paddy", "Boro"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4,5]],

    # ---- Bangladesh wheat ----
    ("BD", "wheat_grain", "Rabi"):   [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4]],
    ("BD", "wheat_grain", "Annual"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4]],

    # ---- Vietnam rice (northern provinces, lat > 16.5°N) ----
    ("VN", "rice_paddy", "Northern Summer Rice"): [(m, 0) for m in [6,7,8,9,10,11,12]],
    ("VN", "rice_paddy", "Northern Autumn Rice"): [(m, 0) for m in [6,7,8,9,10,11,12]],
    ("VN", "rice_paddy", "Northern Winter Rice"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4,5,6,7]],
    ("VN", "rice_paddy", "Northern Spring Rice"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4,5,6,7]],

    # ---- Vietnam rice (southern provinces, lat <= 16.5°N) ----
    ("VN", "rice_paddy", "Southern Summer Rice"): [(m, 0) for m in [6,7,8,9,10,11,12]],
    ("VN", "rice_paddy", "Southern Autumn Rice"): [(m, 0) for m in [6,7,8,9,10,11,12]],
    ("VN", "rice_paddy", "Southern Winter Rice"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4,5]],
    ("VN", "rice_paddy", "Southern Spring Rice"): [(11,-1),(12,-1)] + [(m, 0) for m in [1,2,3,4,5]],
}

# Dual label: canonical season name for display (calendar / stablebound)
SEASON_DISPLAY = {
    "Wet":  "Wet / Main",   "Dry":  "Dry / S2",
    "Aman": "Aman / Main",  "Aus":  "Aus / S3",  "Boro": "Boro / S2",
    "Rabi": "Rabi (wheat)", "Annual": "Annual (wheat)",
    "Northern Summer Rice": "N. Summer Rice",
    "Northern Autumn Rice": "N. Autumn Rice",
    "Northern Winter Rice": "N. Winter / Dong Xuan",
    "Northern Spring Rice": "N. Spring / Dong Xuan",
    "Southern Summer Rice": "S. Summer Rice",
    "Southern Autumn Rice": "S. Autumn Rice",
    "Southern Winter Rice": "S. Winter / Dong Xuan",
    "Southern Spring Rice": "S. Spring / Dong Xuan",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_slug(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-").lower()


def _load_season_yield(cfg: dict, crop: str, season: str) -> pd.DataFrame:
    """
    Pull season-level yield from stats_aggregated: production / area per
    (stable_id, year) for the given crop + season label.
    Skips late_reporting rows. Uses harvested area, falls back to planted area.
    Returns DataFrame with columns [stable_id, year, yield_mt_ha].
    """
    stats = pd.read_csv(cfg["stats_aggregated_path"])
    stats = stats[~stats["late_reporting"]]

    def _get(measure: str) -> pd.DataFrame:
        sub = stats[(stats["variable"] == f"{crop}_{measure}") & (stats["season"] == season)]
        if sub.empty:
            return pd.DataFrame(columns=["stable_id", "year", "value"])
        return sub[["stable_id", "year", "value"]].copy()

    prod = _get("production_mt").rename(columns={"value": "production_mt"})
    harv = _get("area_harvested_ha").rename(columns={"value": "area_harvested_ha"})
    plnt = _get("area_planted_ha").rename(columns={"value": "area_planted_ha"})

    if prod.empty:
        return pd.DataFrame(columns=["stable_id", "year", "yield_mt_ha"])

    merged = prod.merge(harv, on=["stable_id", "year"], how="left")
    merged = merged.merge(plnt, on=["stable_id", "year"], how="left")

    merged["area_ha"] = merged.get("area_harvested_ha")
    no_harv = merged["area_ha"].isna() & merged.get("area_planted_ha", pd.Series(dtype=float)).notna()
    merged.loc[no_harv, "area_ha"] = merged.loc[no_harv, "area_planted_ha"]

    safe_area = merged["area_ha"].replace(0, pd.NA)
    merged["yield_mt_ha"] = merged["production_mt"] / safe_area
    return merged[["stable_id", "year", "yield_mt_ha"]].dropna(subset=["yield_mt_ha"])


def _build_sif_lookup(sif_df: pd.DataFrame) -> dict:
    """Pre-index sif_by_crop as {(stable_id, crop, year, month): sif_value}."""
    lookup = {}
    for row in sif_df.itertuples(index=False):
        lookup[(row.stable_id, row.crop, row.year, row.month)] = row.sif_weighted
    return lookup


def _season_sif_mean(
    sif_lookup: dict, stable_id, fews_crop: str,
    yield_year: int, month_offsets: list[tuple[int, int]]
) -> float:
    """Mean CSIF over the season months for a given district-year."""
    vals = []
    for month, offset in month_offsets:
        v = sif_lookup.get((stable_id, fews_crop, yield_year + offset, month))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _detrend(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(years, values, 1)
    return values - np.polyval(coeffs, years)


# ---------------------------------------------------------------------------
# QA exclusion set (same logic as sif_yield_variance.py)
# ---------------------------------------------------------------------------

def _build_exclusions(cc: str) -> tuple[set, set]:
    excluded_points  = set()
    excluded_series  = set()

    events = pd.read_csv(PROC_DIR / f"qaqc_year_anomaly_{cc.lower()}.csv")
    for _, r in events[events["zscore"].abs() > ANOMALY_REMOVE_Z].iterrows():
        excluded_points.add((r["stable_id"], r["crop"], int(r["year"])))

    rp = PROC_DIR / f"qaqc_repeat_runs_{cc.lower()}.csv"
    if rp.exists():
        repeats = pd.read_csv(rp)
        for _, r in repeats[repeats["flag_repeat"] == True].iterrows():
            for yr in range(int(r["start_year"]), int(r["end_year"]) + 1):
                excluded_points.add((r["stable_id"], r["crop"], yr))

    mp = PROC_DIR / f"qaqc_reported_vs_calc_{cc.lower()}.csv"
    if mp.exists():
        mismatches = pd.read_csv(mp)
        flagged = mismatches[mismatches["pct_diff"] > MISMATCH_THRESHOLD]
        if not flagged.empty:
            cfg = get_country(cc)
            nh  = pd.read_csv(cfg["name_history_path"])
            n2s = nh.groupby("name")["stable_id"].apply(set).to_dict()
            aliases = cfg["admin_name_aliases"]
            for _, r in flagged.iterrows():
                name = aliases.get(r["admin_1"], r["admin_1"])
                for sid in n2s.get(name, set()):
                    excluded_points.add((sid, r["crop"], int(r["year"])))

    cv = pd.read_csv(PROC_DIR / f"qaqc_low_cv_{cc.lower()}.csv")
    for _, r in cv[cv["flag_low_cv"] == True].iterrows():
        excluded_series.add((r["stable_id"], r["crop"]))

    return excluded_points, excluded_series


# ---------------------------------------------------------------------------
# Main correlation computation
# ---------------------------------------------------------------------------

def compute_season_correlations(cc: str) -> pd.DataFrame:
    cfg      = get_country(cc)
    crop_map = cfg["crop_map"]

    sif_path = PROC_DIR / f"sif_by_crop_{cc.lower()}.csv"
    if not sif_path.exists():
        print(f"  No sif_by_crop file for {cc}, skipping.")
        return pd.DataFrame()
    sif_df = pd.read_csv(sif_path)
    sif_lookup = _build_sif_lookup(sif_df)

    excluded_points, excluded_series = _build_exclusions(cc)

    # Find which (fews_crop, season) keys exist for this country
    keys = [(fews_crop, season)
            for (country, fews_crop, season) in SEASON_MONTH_OFFSETS
            if country == cc]

    rows = []
    for fews_crop, season in keys:
        month_offsets = SEASON_MONTH_OFFSETS[(cc, fews_crop, season)]
        yld = _load_season_yield(cfg, fews_crop, season)
        if yld.empty:
            print(f"  {fews_crop}/{season}: no yield data, skipping")
            continue

        n_districts = yld["stable_id"].nunique()
        print(f"  {fews_crop} / {season}: {n_districts} districts, "
              f"{len(month_offsets)} SIF months, "
              f"yield years {yld['year'].min()}-{yld['year'].max()}")

        for stable_id, g in yld.groupby("stable_id"):
            g = g.sort_values("year").reset_index(drop=True)

            # Compute SIF mean for each year in the yield series
            g["sif_mean"] = g["year"].apply(
                lambda y: _season_sif_mean(sif_lookup, stable_id, fews_crop, y, month_offsets)
            )

            for scenario in ("full", "cleaned"):
                gs = g.dropna(subset=["yield_mt_ha", "sif_mean"]).copy()

                if scenario == "cleaned":
                    if (stable_id, fews_crop) in excluded_series:
                        continue
                    gs = gs[~gs.apply(
                        lambda r: (stable_id, fews_crop, int(r["year"])) in excluded_points, axis=1
                    )]

                gs = gs.dropna(subset=["yield_mt_ha", "sif_mean"])
                if len(gs) < MIN_YEARS:
                    continue

                years  = gs["year"].values.astype(float)
                y_anom = _detrend(years, gs["yield_mt_ha"].values.astype(float))
                s_anom = _detrend(years, gs["sif_mean"].values.astype(float))

                if y_anom.std() == 0 or s_anom.std() == 0:
                    continue

                r, p = stats.pearsonr(y_anom, s_anom)
                rows.append((stable_id, fews_crop, season, scenario,
                             len(gs), r, r**2, p))

    return pd.DataFrame(rows, columns=[
        "stable_id", "crop", "season", "scenario",
        "n_years", "r", "r_squared", "p_value",
    ])


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def plot_season_maps(cc: str, corr: pd.DataFrame) -> None:
    cfg = get_country(cc)
    boundaries = gpd.read_file(cfg["boundary_path"])
    out_dir = FIG_DIR / cc.lower() / "sif_yield_season"
    out_dir.mkdir(parents=True, exist_ok=True)

    for (crop, season, scenario), g in corr.groupby(["crop", "season", "scenario"]):
        geo = boundaries.merge(g, on="stable_id", how="left")
        if geo["r"].notna().sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        vmax = max(abs(geo["r"].dropna().min()), abs(geo["r"].dropna().max()), 0.3)
        geo.plot(column="r", ax=ax, cmap="coolwarm", legend=True,
                 vmin=-vmax, vmax=vmax,
                 missing_kwds={"color": "lightgrey", "label": "no data / ineligible"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "Pearson r (detrended yield vs. CSIF)", "shrink": 0.6})

        sig = geo[geo["p_value"] < 0.05]
        if not sig.empty:
            sig.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.8)

        n_sig   = int((geo["p_value"] < 0.05).sum())
        n_total = int(geo["r"].notna().sum())
        label   = SEASON_DISPLAY.get(season, season)
        ax.set_title(f"{cc} — {crop} — {label} — yield~CSIF ({scenario})\n"
                     f"Sig (p<0.05, black outline): {n_sig}/{n_total} districts")
        ax.axis("off")

        fname = f"sif_yield_season_{cc.lower()}_{_safe_slug(crop)}_{_safe_slug(season)}_{scenario}.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_path.relative_to(FIG_DIR)}")


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(cc: str, corr: pd.DataFrame) -> None:
    print(f"\n  Summary by crop / season / scenario:")
    for (crop, season, scenario), g in corr.groupby(["crop", "season", "scenario"]):
        n_sig = (g["p_value"] < 0.05).sum()
        label = SEASON_DISPLAY.get(season, season)
        print(f"    {crop:20s} [{label:28s}] [{scenario:8s}]  "
              f"n={len(g):3d}  mean r={g['r'].mean():+.3f}  "
              f"median r={g['r'].median():+.3f}  sig={n_sig}/{len(g)}")


# ---------------------------------------------------------------------------
# Per-country entry point
# ---------------------------------------------------------------------------

def process_country(cc: str) -> None:
    print(f"\n=== {cc}: season-specific SIF-yield correlation")
    corr = compute_season_correlations(cc)
    if corr.empty:
        print("  No results.")
        return

    out_path = PROC_DIR / f"sif_yield_season_corr_{cc.lower()}.csv"
    corr.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path.name}: {len(corr)} rows")

    print_summary(cc, corr)
    plot_season_maps(cc, corr)


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    for cc in codes:
        process_country(cc)


if __name__ == "__main__":
    main()
