"""
District drill-down: given a district flagged by any of the qaqc_yield_checks
checks, pull together everything a researcher would want to look at next.

The choropleth maps are a first-glance triage tool -- they tell you WHICH
district to look at. This is the second-glance tool: once you've picked one,
it assembles the detail needed to judge whether a flag is a real data
problem or just normal variation, without re-running every check by hand.

For a given (country, stable_id), prints:
  1. Admin names this district has carried (from stablebound's name_history.csv
     -- needed because the reported-vs-calculated check operates on raw admin
     names, not stable_id, so this is the join key back to that check).
  2. Per crop grown in this district:
     - low-CV / CV-relative-outlier flag status
     - each anomalous year (from qaqc_year_anomaly), shown together with its
       immediate neighboring years' yield, so it's visible at a glance
       whether it's an isolated spike or part of a longer move
     - any exact-repeat runs
  3. Cross-crop check: for every anomalous year found for ANY crop in this
     district, show what every OTHER crop in the same district was doing
     in that same year. If several crops show unusual behavior in the same
     year, that points to a shared cause (a real shock, or a reporting/
     methodology issue that year) rather than a crop-specific data problem --
     this is exactly the lens that surfaced the Bangladesh 2021 maize finding
     in PROGRESS_LOG.md, generalized into a reusable per-district tool.
  4. Reported-vs-calculated mismatch rows (raw FEWS data) for any admin name
     ever associated with this district.

This is a read-only diagnostic tool -- it does not change any flag, just
assembles already-computed results plus a bit of raw-data context. No
figures; output is print statements (for ad hoc use) plus a dict of
DataFrames (for programmatic use / notebooks).

Usage:
    python pipeline/district_drilldown.py BD BD.ADM2.00012
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

NEIGHBOR_WINDOW = 2   # show this many years before/after an anomalous year


def _path(cc: str, name: str) -> Path:
    return PROC_DIR / f"{name}_{cc.lower()}.csv"


def drilldown(cc: str, stable_id: str, verbose: bool = True) -> dict:
    out = {}
    cfg = get_country(cc)

    # --- 1. admin names this district has carried ---
    name_hist = pd.read_csv(cfg["name_history_path"])
    names = sorted(name_hist[name_hist["stable_id"] == stable_id]["name"].unique())
    out["admin_names"] = names
    if verbose:
        print(f"\n{'='*70}\nDISTRICT DRILLDOWN: {cc} / {stable_id}\n{'='*70}")
        print(f"\nAdmin name(s) ever associated with this district: {names}")

    # --- load the per-country tables this district could appear in ---
    yld = pd.read_csv(_path(cc, "annual_yield"))
    yld_district = yld[yld["stable_id"] == stable_id].sort_values(["crop", "year"])
    out["yield_series"] = yld_district

    cv = pd.read_csv(_path(cc, "qaqc_low_cv"))
    cv_district = cv[cv["stable_id"] == stable_id]
    out["cv_summary"] = cv_district

    events = pd.read_csv(_path(cc, "qaqc_year_anomaly"))
    events_district = events[events["stable_id"] == stable_id]
    out["anomaly_events"] = events_district

    repeats_path = _path(cc, "qaqc_repeat_runs")
    repeats = pd.read_csv(repeats_path) if repeats_path.exists() else pd.DataFrame()
    repeats_district = repeats[repeats["stable_id"] == stable_id] if not repeats.empty else repeats
    out["repeat_runs"] = repeats_district

    corr_path = _path(cc, "sif_yield_corr")
    corr = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()
    corr_district = corr[corr["stable_id"] == stable_id] if not corr.empty else corr
    out["sif_yield_corr"] = corr_district

    crops_here = sorted(yld_district["crop"].unique())
    if verbose:
        print(f"Crops grown in this district: {crops_here}")

    # --- 2. per-crop detail ---
    if verbose:
        print(f"\n--- Per-crop detail ---")
    for crop in crops_here:
        crop_series = yld_district[yld_district["crop"] == crop].set_index("year")["yield_mt_ha"]
        cv_row = cv_district[cv_district["crop"] == crop]

        if verbose:
            print(f"\n  [{crop}]")
            if not cv_row.empty:
                r = cv_row.iloc[0]
                flags = []
                if r.get("flag_low_cv"):
                    flags.append("LOW-CV (suspiciously flat)")
                if r.get("flag_cv_outlier_high"):
                    flags.append(f"CV relative outlier HIGH (z={r['cv_zscore']:.2f})")
                if r.get("flag_cv_outlier_low"):
                    flags.append(f"CV relative outlier LOW (z={r['cv_zscore']:.2f})")
                flag_str = ", ".join(flags) if flags else "no CV flags"
                print(f"    CV = {r['cv']:.3f} over {int(r['n_years'])} years "
                      f"(crop mean CV = {r['crop_mean_cv']:.3f} if eligible) -- {flag_str}")

            crop_corr = corr_district[corr_district["crop"] == crop] if not corr_district.empty else pd.DataFrame()
            if not crop_corr.empty:
                for _, cr in crop_corr.iterrows():
                    sig = "significant" if cr["p_value"] < 0.05 else "not significant"
                    print(f"    SIF~yield [{cr['scenario']:7s}]: r={cr['r']:+.3f}  R2={cr['r_squared']:.3f}  "
                          f"p={cr['p_value']:.3f} ({sig})  n_years={int(cr['n_years'])}")
            else:
                print(f"    SIF~yield: no eligible series (likely too few years overlapping SIF coverage, "
                      f"2000 onward)")

        crop_events = events_district[events_district["crop"] == crop]
        for _, ev in crop_events.iterrows():
            yr = int(ev["year"])
            window_years = range(yr - NEIGHBOR_WINDOW, yr + NEIGHBOR_WINDOW + 1)
            window = crop_series.reindex(window_years)
            if verbose:
                print(f"    ANOMALOUS YEAR {yr}: yield={ev['yield_mt_ha']:.3f} "
                      f"(series mean={ev['series_mean']:.3f}, z={ev['zscore']:.2f})")
                print(f"      neighboring years: " +
                      ", ".join(f"{y}={'NA' if pd.isna(v) else round(v,3)}" for y, v in window.items()))

        crop_repeats = repeats_district[repeats_district["crop"] == crop] if not repeats_district.empty else pd.DataFrame()
        for _, rr in crop_repeats.iterrows():
            if verbose:
                tag = "ALSO repeats in production+area" if rr["production_and_area_also_identical"] else \
                      "yield repeats but production/area do NOT (ratio coincidence)"
                print(f"    EXACT-REPEAT RUN {int(rr['start_year'])}-{int(rr['end_year'])} "
                      f"({int(rr['run_length'])} yrs) at yield={rr['yield_mt_ha']:.3f} -- {tag}")

    # --- 3. cross-crop check: what were other crops doing in the same flagged year? ---
    flagged_years = sorted(events_district["year"].unique())
    cross_crop_rows = []
    if flagged_years and verbose:
        print(f"\n--- Cross-crop check: other crops' yield in this district's flagged years ---")
    for yr in flagged_years:
        if verbose:
            print(f"\n  Year {yr} (flagged for: "
                  f"{', '.join(events_district[events_district['year']==yr]['crop'])})")
        for crop in crops_here:
            crop_series = yld_district[yld_district["crop"] == crop].set_index("year")["yield_mt_ha"]
            if yr not in crop_series.index:
                continue
            val = crop_series.loc[yr]
            cv_row = cv_district[cv_district["crop"] == crop]
            also_flagged = ((events_district["crop"] == crop) & (events_district["year"] == yr)).any()
            mean = cv_row.iloc[0]["mean_yield"] if not cv_row.empty else float("nan")
            note = "  <-- ALSO ANOMALOUS" if also_flagged else ""
            cross_crop_rows.append((yr, crop, val, mean, also_flagged))
            if verbose:
                print(f"    {crop:25s} yield={val:.3f}  (district-crop mean={mean:.3f}){note}")
    out["cross_crop_same_year"] = pd.DataFrame(
        cross_crop_rows, columns=["year", "crop", "yield_mt_ha", "mean_yield", "also_anomalous"]
    )

    # --- 4. reported vs. calculated mismatches for any admin name this district has carried ---
    mismatch_path = _path(cc, "qaqc_reported_vs_calc")
    mismatches = pd.read_csv(mismatch_path) if mismatch_path.exists() else pd.DataFrame()
    if not mismatches.empty:
        # expand lineage-spelling names with any raw alias that maps to one of them,
        # so renamed districts (e.g. "Chattogram" -> "Chittagong") still match
        aliases = cfg["admin_name_aliases"]
        raw_aliases = {raw for raw, lineage in aliases.items() if lineage in names}
        match_names = set(names) | raw_aliases
        district_mismatches = mismatches[mismatches["admin_1"].isin(match_names)]
    else:
        district_mismatches = mismatches
    out["reported_vs_calc_mismatches"] = district_mismatches

    if verbose:
        print(f"\n--- Reported-vs-calculated mismatches for this district's admin name(s) ---")
        if district_mismatches.empty:
            print("    none")
        else:
            print(district_mismatches[["admin_1", "year", "crop", "reported", "calculated", "pct_diff"]]
                  .to_string(index=False))

    if verbose:
        print(f"\n{'='*70}\n")

    return out


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("Usage: python pipeline/district_drilldown.py <CC> <stable_id>")
    cc, stable_id = sys.argv[1].upper(), sys.argv[2]
    if cc not in all_codes():
        sys.exit(f"Unknown country code {cc}. Choose from {all_codes()}")
    drilldown(cc, stable_id)


if __name__ == "__main__":
    main()
