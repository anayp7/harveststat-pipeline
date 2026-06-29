"""
SIF-yield variance analysis: for each (stable_id, crop) series, how well
does crop-weighted SIF (Step 4 of the pipeline) track crop yield?

Methodology (decided over several rounds of discussion, see PROGRESS_LOG.md):

1. DETREND both yield and SIF before correlating. Both series likely carry
   long-run secular trends (yield from technology/inputs, SIF potentially
   from land-use change or the sensor product's own multi-decade
   calibration) -- correlating raw values risks a high R^2 driven by two
   series that simply both climb over decades, not by real year-to-year
   co-variability. Each series is detrended with a simple linear fit
   against year; everything downstream uses the residual anomalies. This
   mirrors the original India pipeline's use of detrended yield anomalies
   against climate drivers.

2. Report signed Pearson r (not just R^2), plus the p-value and the
   number of years used. R^2 = r^2 throws away direction; a district
   where SIF and yield move oppositely is itself worth seeing, not hiding.

3. Two scenarios, run side by side rather than picking one:
     - "full": every reported year, no QA exclusions.
     - "cleaned": QA-flagged points/series removed first. See exactly
       which flags are used and why below.

   QA exclusions applied for "cleaned" (decided explicitly, not all flags
   are used the same way -- see PROGRESS_LOG.md for the full reasoning):
     - Single-year anomaly events at |z| > 3 (NOT the check's own |z| > 2
       flagging threshold). z=2 is tuned for "worth a look," not "almost
       certainly an error" -- real extreme-but-genuine shocks often land
       in the z=2-3 range too, and excluding those would bias the analysis
       against finding genuine SIF-yield signal in exactly the years
       where it should be strongest. Checked empirically against known
       cases before settling on 3: at |z|>4 almost nothing survives
       (Vietnam: 0 of 154 events) and it stops catching the clearly-
       impossible Vietnam cassava values (z~3.1-3.5); |z|>3 keeps most of
       those while still leaving Bangladesh's milder 2021 maize event
       (z~-2.0 to -2.2) in the "full" scenario only.
     - Exact-repeat runs, hard-flagged (length>=4): every year in the run
       excluded, not just one point -- these aren't independent
       observations.
     - Reported-vs-calculated mismatch (>10%, same threshold as the flag
       itself): joined via name_history.csv (admin name -> stable_id,
       same approach and same non-year-aware-name caveat as
       district_drilldown.py -- acceptable given how few rows this is).
     - Absolute low-CV flag: entire series dropped, not just a point.
     - Relative CV outlier (z-score vs. peers): deliberately NOT excluded
       -- could be genuine, more-informative volatility rather than bad
       data (see PROGRESS_LOG.md).

Output:
    data/processed/sif_yield_corr_<cc>.csv
        stable_id, crop, scenario, n_years, r, r_squared, p_value
    figures/<cc>/sif_yield_corr_map_<cc>_<crop>_<scenario>.png
        choropleth of signed r, p<0.05 districts outlined

Usage:
    python pipeline/sif_yield_variance.py BD
    python pipeline/sif_yield_variance.py            # all three countries
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

MIN_YEARS = 5             # same eligibility floor used throughout qaqc_yield_checks.py
ANOMALY_REMOVE_Z = 3.0    # stricter than the check's own |z|>2 flagging threshold
MISMATCH_THRESHOLD_PCT = 10.0


def _path(cc: str, name: str) -> Path:
    return PROC_DIR / f"{name}_{cc.lower()}.csv"


# ---------------------------------------------------------------------------
# Build the "cleaned" exclusion set
# ---------------------------------------------------------------------------

def build_excluded_points(cc: str) -> tuple[set, set]:
    """
    Returns (excluded_points, excluded_series):
      excluded_points: set of (stable_id, crop, year) to drop
      excluded_series: set of (stable_id, crop) to drop entirely
    """
    excluded_points = set()
    excluded_series = set()

    # --- single-year anomalies at the stricter |z| > 3 ---
    events = pd.read_csv(_path(cc, "qaqc_year_anomaly"))
    strict = events[events["zscore"].abs() > ANOMALY_REMOVE_Z]
    for _, r in strict.iterrows():
        excluded_points.add((r["stable_id"], r["crop"], int(r["year"])))

    # --- exact-repeat runs, hard-flagged ---
    repeats_path = _path(cc, "qaqc_repeat_runs")
    if repeats_path.exists():
        repeats = pd.read_csv(repeats_path)
        flagged_runs = repeats[repeats["flag_repeat"] == True]
        for _, r in flagged_runs.iterrows():
            for yr in range(int(r["start_year"]), int(r["end_year"]) + 1):
                excluded_points.add((r["stable_id"], r["crop"], yr))

    # --- reported vs. calculated mismatch (>10%), joined via admin name ---
    mismatch_path = _path(cc, "qaqc_reported_vs_calc")
    if mismatch_path.exists():
        mismatches = pd.read_csv(mismatch_path)
        flagged_mismatch = mismatches[mismatches["pct_diff"] > MISMATCH_THRESHOLD_PCT]
        if not flagged_mismatch.empty:
            cfg = get_country(cc)
            name_hist = pd.read_csv(cfg["name_history_path"])
            name_to_stable = name_hist.groupby("name")["stable_id"].apply(set).to_dict()
            aliases = cfg["admin_name_aliases"]
            for _, r in flagged_mismatch.iterrows():
                lineage_name = aliases.get(r["admin_1"], r["admin_1"])
                stable_ids = name_to_stable.get(lineage_name, set())
                for sid in stable_ids:
                    excluded_points.add((sid, r["crop"], int(r["year"])))

    # --- absolute low-CV flag: drop the whole series ---
    cv = pd.read_csv(_path(cc, "qaqc_low_cv"))
    low_cv_series = cv[cv["flag_low_cv"] == True]
    for _, r in low_cv_series.iterrows():
        excluded_series.add((r["stable_id"], r["crop"]))

    return excluded_points, excluded_series


# ---------------------------------------------------------------------------
# Detrend + correlate, per (stable_id, crop), per scenario
# ---------------------------------------------------------------------------

def detrend(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(years, values, 1)
    trend = np.polyval(coeffs, years)
    return values - trend


def compute_correlations(cc: str) -> pd.DataFrame:
    joined = pd.read_csv(_path(cc, "annual_joined")).dropna(subset=["yield_mt_ha", "sif_annual_mean"])
    excluded_points, excluded_series = build_excluded_points(cc)

    print(f"  QA exclusions for 'cleaned' scenario: {len(excluded_points)} point(s), "
          f"{len(excluded_series)} whole series")

    rows = []
    for (stable_id, crop), g in joined.groupby(["stable_id", "crop"]):
        g = g.sort_values("year")
        for scenario in ("full", "cleaned"):
            gs = g
            if scenario == "cleaned":
                if (stable_id, crop) in excluded_series:
                    continue
                mask = ~gs.apply(lambda r: (stable_id, crop, int(r["year"])) in excluded_points, axis=1)
                gs = gs[mask]

            if len(gs) < MIN_YEARS:
                continue

            years = gs["year"].values.astype(float)
            yield_anom = detrend(years, gs["yield_mt_ha"].values)
            sif_anom = detrend(years, gs["sif_annual_mean"].values)

            if np.std(yield_anom) == 0 or np.std(sif_anom) == 0:
                continue

            r, p = stats.pearsonr(yield_anom, sif_anom)
            rows.append((stable_id, crop, scenario, len(gs), r, r ** 2, p))

    return pd.DataFrame(rows, columns=[
        "stable_id", "crop", "scenario", "n_years", "r", "r_squared", "p_value",
    ])


# ---------------------------------------------------------------------------
# Country/crop-level summary
# ---------------------------------------------------------------------------

def print_summary(cc: str, corr: pd.DataFrame) -> None:
    print(f"\n  Summary by crop and scenario:")
    for (crop, scenario), g in corr.groupby(["crop", "scenario"]):
        n_sig = (g["p_value"] < 0.05).sum()
        print(f"    {crop:25s} [{scenario:8s}] n_districts={len(g):3d}  "
              f"mean r={g['r'].mean():+.3f}  median r={g['r'].median():+.3f}  "
              f"mean R2={g['r_squared'].mean():.3f}  median R2={g['r_squared'].median():.3f}  "
              f"mean p={g['p_value'].mean():.3f}  sig(p<0.05)={n_sig}/{len(g)}")


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def plot_corr_maps(cc: str, corr: pd.DataFrame) -> None:
    boundaries = gpd.read_file(get_country(cc)["boundary_path"])
    out_dir = FIG_DIR / cc.lower() / "sif_yield"
    out_dir.mkdir(parents=True, exist_ok=True)

    for (crop, scenario), g in corr.groupby(["crop", "scenario"]):
        geo = boundaries.merge(g, on="stable_id", how="left")
        if geo["r"].notna().sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        vmax = max(abs(geo["r"].min()), abs(geo["r"].max()), 0.5)
        geo.plot(column="r", ax=ax, cmap="coolwarm", legend=True,
                 vmin=-vmax, vmax=vmax,
                 missing_kwds={"color": "lightgrey", "label": "no data / ineligible"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "Pearson r (detrended yield vs. SIF)", "shrink": 0.6})

        sig_geo = geo[geo["p_value"] < 0.05]
        if not sig_geo.empty:
            sig_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.8)

        n_sig = int((geo["p_value"] < 0.05).sum())
        n_total = int(geo["r"].notna().sum())
        ax.set_title(f"{cc} — {crop} — yield~SIF correlation ({scenario})\n"
                     f"Sig (p<0.05, black outline): {n_sig}/{n_total} districts")
        ax.axis("off")

        out_path = out_dir / f"sif_yield_corr_map_{cc.lower()}_{crop}_{scenario}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_path.relative_to(FIG_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_country(cc: str) -> None:
    print(f"\n=== {cc}: SIF-yield variance analysis")
    corr = compute_correlations(cc)
    if corr.empty:
        print("  No eligible series found.")
        return

    out_path = PROC_DIR / f"sif_yield_corr_{cc.lower()}.csv"
    corr.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name}: {len(corr)} rows")

    print_summary(cc, corr)
    plot_corr_maps(cc, corr)


def main() -> None:
    codes = [sys.argv[1].upper()] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country code(s): {invalid}. Choose from {all_codes()}")
    for cc in codes:
        process_country(cc)


if __name__ == "__main__":
    main()
