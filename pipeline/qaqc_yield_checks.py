"""
Yield-only QA/QC checks, run before any SIF/RS data enters the analysis.

Two checks, deliberately kept independent of each other:

1. Reported-vs-calculated yield mismatch (row-level, raw FEWS data).
   The source CSVs carry both a "Yield: MT/ha (reported)" column (as
   published by the original statistics agency) and a "Yield: MT/ha
   (calculated)" column (presumably production/area, computed by FEWS).
   Large disagreement between the two flags a row before it is ever
   touched by our aggregation -- a unit error, an area/production
   mismatch, or a non-standard reporting basis. This check works on the
   raw CSV directly, NOT on stable-boundary-aggregated data, since the
   row is the level at which "what went wrong" is interpretable.

2. Low coefficient-of-variation ("suspiciously flat yield"), district
   level. Direct generalization of the India fig_yield_cv.py check:
   districts whose yield barely changes year to year tend to indicate
   copy-forward/stale data entry rather than a genuinely stable harvest.
   Requires a minimum number of reported years to be meaningful -- a CV
   computed on a 3-year sparse series is not trustworthy.

A third check (implausible yield ceiling per crop) was considered and
deliberately deferred -- a defensible ceiling needs crop-specific
agronomic research, otherwise the cutoff is arbitrary. See PROGRESS_LOG.md.

4. Single-year anomaly ("one bad year"), within a single district-crop
   series. For each (stable_id, crop) series, z-score every year against
   THAT series' own mean and std (not against other districts, and not
   against local neighbors), flag |z| > 2.0, and count the number of
   anomalous years per series. The map plots that count directly per
   district-crop. This is deliberately the plain, standard definition of
   a yield outlier year -- it will also catch a year that's part of a
   genuine multi-year trend shift, not just an isolated spike that
   reverts, which is an accepted tradeoff for keeping the definition
   simple and standard. (An earlier version of this check used a local
   neighbor-ratio and a second z-score layer across districts; replaced
   after clarifying the intent -- see PROGRESS_LOG.md.)

5. Exact-repeat run detection, district-crop-series level. Flags runs of
   consecutive, truly-adjacent years (no gap) reporting an EXACTLY
   identical yield value -- a stronger fingerprint of copy-forward/stale
   data entry than the low-CV check, which only catches low *overall*
   variance, not literal repetition. Also checks whether production_mt
   and area_ha are themselves exactly repeated for the same run (a
   stronger signal than yield alone coincidentally matching across
   different production/area pairs).

Usage:
    python pipeline/qaqc_yield_checks.py BD          # one country
    python pipeline/qaqc_yield_checks.py             # all three

Output:
    data/processed/qaqc_reported_vs_calc_<cc>.csv   -- flagged raw rows
    data/processed/qaqc_low_cv_<cc>.csv             -- full district x crop CV table
    data/processed/qaqc_year_anomaly_<cc>.csv       -- flagged single-year anomalies
    data/processed/qaqc_repeat_runs_<cc>.csv        -- exact-repeat runs, length >= 3
    figures/<cc>/qaqc_cv_map_<cc>_<crop>.png             -- absolute CV choropleth
    figures/<cc>/qaqc_cv_zscore_map_<cc>_<crop>.png      -- relative CV outlier choropleth
    figures/<cc>/qaqc_anomaly_map_<cc>_<crop>.png        -- single-year anomaly count choropleth
    figures/<cc>/qaqc_repeat_map_<cc>_<crop>.png         -- max repeat-run-length choropleth
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
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

MISMATCH_THRESHOLD_PCT = 10.0   # flag if |reported - calculated| / calculated > this
LOW_CV_THRESHOLD       = 0.05   # same threshold used in the India fig_yield_cv.py check
MIN_YEARS_FOR_CV       = 5      # require at least this many reported years
MIN_DISTRICTS_FOR_ZSCORE = 5    # below this, a crop's mean/std CV is too noisy to z-score against
Z_OUTLIER_THRESHOLD    = 2.0    # |z| beyond this flags a year/district as anomalous
REPEAT_RUN_MIN_REPORT  = 3      # report runs of exact-repeat years at this length or longer
REPEAT_RUN_FLAG        = 4      # hard-flag runs at this length or longer


def crop_slug(label: str) -> str:
    s = str(label).strip().lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


# ---------------------------------------------------------------------------
# Check 1: reported vs. calculated yield mismatch
# ---------------------------------------------------------------------------

def check_reported_vs_calculated(cc: str) -> pd.DataFrame:
    print(f"\n=== {cc}: reported vs. calculated yield mismatch")
    df = pd.read_csv(get_country(cc)["raw_csv_path"], low_memory=False)

    reported   = pd.to_numeric(df["Yield: MT/ha (reported)"], errors="coerce")
    calculated = pd.to_numeric(df["Yield: MT/ha (calculated)"], errors="coerce")

    both = reported.notna() & calculated.notna() & (calculated > 0)
    pct_diff = ((reported - calculated).abs() / calculated) * 100

    crop = df["Source crop"].fillna(df.get("crop")).map(crop_slug)

    out = pd.DataFrame({
        "admin_1":    df["Admin 1"],
        "year":       df["Year"],
        "season":     df.get("Season"),
        "crop":       crop,
        "reported":   reported,
        "calculated": calculated,
        "pct_diff":   pct_diff,
    })
    out = out[both].copy()

    flagged = out[out["pct_diff"] > MISMATCH_THRESHOLD_PCT].sort_values(
        "pct_diff", ascending=False
    )

    print(f"  {both.sum()} rows with both yield fields reported")
    print(f"  {len(flagged)} rows exceed {MISMATCH_THRESHOLD_PCT:.0f}% disagreement "
          f"({len(flagged)/both.sum()*100:.1f}% of comparable rows)")
    if len(flagged):
        print("  by crop (top 5):")
        print("    " + flagged["crop"].value_counts().head(5).to_string().replace("\n", "\n    "))

    out_path = PROC_DIR / f"qaqc_reported_vs_calc_{cc.lower()}.csv"
    flagged.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name}: {len(flagged)} flagged rows")
    return flagged


# ---------------------------------------------------------------------------
# Check 3: low coefficient of variation
# ---------------------------------------------------------------------------

def check_low_cv(cc: str) -> pd.DataFrame:
    print(f"\n=== {cc}: low-CV (suspiciously flat yield) check")
    yld = pd.read_csv(PROC_DIR / f"annual_yield_{cc.lower()}.csv")
    yld = yld.dropna(subset=["yield_mt_ha"])

    def _stats(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mean = g["yield_mt_ha"].mean()
        std = g["yield_mt_ha"].std()
        cv = std / mean if mean and mean > 0 else np.nan
        year_min, year_max = g["year"].min(), g["year"].max()
        span = year_max - year_min + 1
        completeness = n / span if span > 0 else np.nan
        return pd.Series({
            "n_years": n, "year_min": year_min, "year_max": year_max,
            "completeness": completeness, "mean_yield": mean, "cv": cv,
        })

    summary = yld.groupby(["stable_id", "crop"]).apply(_stats, include_groups=False).reset_index()
    summary["eligible"] = summary["n_years"] >= MIN_YEARS_FOR_CV
    summary["flag_low_cv"] = summary["eligible"] & (summary["cv"] < LOW_CV_THRESHOLD)

    n_eligible = summary["eligible"].sum()
    n_flagged = summary["flag_low_cv"].sum()
    print(f"  {len(summary)} (district, crop) series total")
    print(f"  {n_eligible} have >= {MIN_YEARS_FOR_CV} reported years (eligible for CV flagging)")
    print(f"  {n_flagged} flagged as low-CV (< {LOW_CV_THRESHOLD:.0%}) among eligible series")
    if n_flagged:
        print("  by crop:")
        print("    " + summary[summary["flag_low_cv"]]["crop"].value_counts().to_string().replace("\n", "\n    "))

    # --- relative outlier check: how does each district's CV compare to its
    # peers growing the same crop? Catches districts that stand out from the
    # pack even when no district crosses the absolute LOW_CV_THRESHOLD floor.
    summary["crop_n_eligible"] = summary.groupby("crop")["eligible"].transform("sum")
    summary["crop_mean_cv"] = np.nan
    summary["crop_std_cv"] = np.nan
    summary["cv_zscore"] = np.nan

    print(f"\n  Relative CV outlier check (z-score vs. same-crop peers, "
          f"|z|>{Z_OUTLIER_THRESHOLD}; requires >= {MIN_DISTRICTS_FOR_ZSCORE} eligible districts/crop):")
    for crop, idx in summary.groupby("crop").groups.items():
        elig_mask = summary.loc[idx, "eligible"]
        n_elig_crop = elig_mask.sum()
        if n_elig_crop < MIN_DISTRICTS_FOR_ZSCORE:
            print(f"    {crop}: only {n_elig_crop} eligible districts, skipping (too few for a stable mean/std)")
            continue
        elig_idx = summary.loc[idx][elig_mask].index
        mean_cv = summary.loc[elig_idx, "cv"].mean()
        std_cv = summary.loc[elig_idx, "cv"].std()
        summary.loc[elig_idx, "crop_mean_cv"] = mean_cv
        summary.loc[elig_idx, "crop_std_cv"] = std_cv
        if std_cv and std_cv > 0:
            summary.loc[elig_idx, "cv_zscore"] = (summary.loc[elig_idx, "cv"] - mean_cv) / std_cv
        n_out = (summary.loc[elig_idx, "cv_zscore"].abs() > Z_OUTLIER_THRESHOLD).sum()
        print(f"    {crop}: {n_elig_crop} eligible districts, mean CV={mean_cv:.3f}, "
              f"std={std_cv:.3f}, {n_out} relative outlier(s)")

    summary["flag_cv_outlier_high"] = summary["cv_zscore"] > Z_OUTLIER_THRESHOLD
    summary["flag_cv_outlier_low"] = summary["cv_zscore"] < -Z_OUTLIER_THRESHOLD

    out_path = PROC_DIR / f"qaqc_low_cv_{cc.lower()}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path.name}: {len(summary)} rows")
    return summary


# ---------------------------------------------------------------------------
# Check 4: single-year anomaly ("one bad year")
# ---------------------------------------------------------------------------

def check_year_anomaly(cc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within-series anomaly detection: for each (stable_id, crop) series,
    z-score every year against THAT series' own mean and std (not against
    other districts, and not against local neighbors). Flag |z| > 2.0.
    District-crop output is then just the COUNT of anomalous years -- no
    further normalization across districts. This is the standard,
    simplest definition of a yield outlier year, and deliberately
    different from the earlier neighbor-ratio version: it will also catch
    a year that's part of a genuine multi-year trend shift, not just an
    isolated spike that reverts -- that's an accepted tradeoff for the
    simpler, more standard definition.
    """
    print(f"\n=== {cc}: single-year anomaly check (within-series z-score)")
    yld = pd.read_csv(PROC_DIR / f"annual_yield_{cc.lower()}.csv").dropna(subset=["yield_mt_ha"])

    event_rows = []
    district_rows = []   # one row per (stable_id, crop): n_years, n_anomalous_years

    for (stable_id, crop), g in yld.groupby(["stable_id", "crop"]):
        g = g.sort_values("year").reset_index(drop=True)
        n_years = len(g)
        if n_years < MIN_YEARS_FOR_CV:
            district_rows.append((stable_id, crop, n_years, np.nan, 0))
            continue

        mean = g["yield_mt_ha"].mean()
        std = g["yield_mt_ha"].std()
        if not std or std == 0:
            district_rows.append((stable_id, crop, n_years, mean, 0))
            continue

        zscores = (g["yield_mt_ha"] - mean) / std
        flagged = zscores.abs() > Z_OUTLIER_THRESHOLD
        for i in g.index[flagged]:
            event_rows.append((stable_id, crop, int(g.loc[i, "year"]), g.loc[i, "yield_mt_ha"],
                                mean, std, zscores.loc[i]))
        district_rows.append((stable_id, crop, n_years, mean, int(flagged.sum())))

    events = pd.DataFrame(event_rows, columns=[
        "stable_id", "crop", "year", "yield_mt_ha", "series_mean", "series_std", "zscore",
    ]).sort_values("zscore", key=lambda s: s.abs(), ascending=False)

    print(f"  {len(events)} anomalous-year events flagged (|z| > {Z_OUTLIER_THRESHOLD}, "
          f"z computed against each district-crop series' own mean/std)")
    if len(events):
        print("  by crop:")
        print("    " + events["crop"].value_counts().to_string().replace("\n", "\n    "))

    events_path = PROC_DIR / f"qaqc_year_anomaly_{cc.lower()}.csv"
    events.to_csv(events_path, index=False)
    print(f"  Wrote {events_path.name}: {len(events)} rows")

    district = pd.DataFrame(district_rows, columns=[
        "stable_id", "crop", "n_years", "series_mean", "n_anomalous_years",
    ])
    district["eligible"] = district["n_years"] >= MIN_YEARS_FOR_CV

    n_eligible = district["eligible"].sum()
    n_with_anomaly = (district["n_anomalous_years"] > 0).sum()
    print(f"  {n_eligible} (district, crop) series have >= {MIN_YEARS_FOR_CV} years (eligible)")
    print(f"  {n_with_anomaly} have at least 1 anomalous year")

    district_path = PROC_DIR / f"qaqc_year_anomaly_district_{cc.lower()}.csv"
    district.to_csv(district_path, index=False)
    print(f"  Wrote {district_path.name}: {len(district)} rows")

    return events, district


# ---------------------------------------------------------------------------
# Check 5: exact-repeat run detection
# ---------------------------------------------------------------------------

def check_repeat_runs(cc: str) -> pd.DataFrame:
    print(f"\n=== {cc}: exact-repeat run detection")
    yld = pd.read_csv(PROC_DIR / f"annual_yield_{cc.lower()}.csv").dropna(subset=["yield_mt_ha"])

    runs = []
    for (stable_id, crop), g in yld.groupby(["stable_id", "crop"]):
        g = g.sort_values("year").reset_index(drop=True)
        years = g["year"].values
        vals = g["yield_mt_ha"].values
        prod = g["production_mt"].values
        area = g["area_ha"].values

        run_start = 0
        for i in range(1, len(g) + 1):
            same_as_prev = (
                i < len(g)
                and years[i] - years[i - 1] == 1
                and vals[i] == vals[i - 1]
            )
            if not same_as_prev:
                run_len = i - run_start
                if run_len >= REPEAT_RUN_MIN_REPORT:
                    also_prod_area = bool(
                        np.all(prod[run_start:i] == prod[run_start])
                        and np.all(area[run_start:i] == area[run_start])
                    )
                    runs.append((stable_id, crop, int(years[run_start]), int(years[i - 1]),
                                 run_len, vals[run_start], also_prod_area))
                run_start = i

    out = pd.DataFrame(runs, columns=[
        "stable_id", "crop", "start_year", "end_year", "run_length",
        "yield_mt_ha", "production_and_area_also_identical",
    ]).sort_values("run_length", ascending=False)
    out["flag_repeat"] = out["run_length"] >= REPEAT_RUN_FLAG

    n_flagged = out["flag_repeat"].sum()
    print(f"  {len(out)} runs of length >= {REPEAT_RUN_MIN_REPORT} found; "
          f"{n_flagged} flagged at length >= {REPEAT_RUN_FLAG}")
    if n_flagged:
        print("  by crop (flagged only):")
        print("    " + out[out["flag_repeat"]]["crop"].value_counts().to_string().replace("\n", "\n    "))

    out_path = PROC_DIR / f"qaqc_repeat_runs_{cc.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name}: {len(out)} rows")
    return out


# ---------------------------------------------------------------------------
# Choropleth: CV per district, per crop
# ---------------------------------------------------------------------------

def plot_cv_maps(cc: str, summary: pd.DataFrame) -> None:
    boundaries = gpd.read_file(get_country(cc)["boundary_path"])
    out_dir = FIG_DIR / cc.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = sorted(summary["crop"].unique())
    for crop in crops:
        sub = summary[summary["crop"] == crop]
        geo = boundaries.merge(sub, on="stable_id", how="left")
        if geo["cv"].notna().sum() == 0:
            continue

        # --- absolute CV map ---
        fig, ax = plt.subplots(figsize=(8, 8))
        geo.plot(column="cv", ax=ax, cmap="viridis_r", legend=True,
                 missing_kwds={"color": "lightgrey", "label": "no data"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "Yield CV (std / mean)", "shrink": 0.6})

        flagged_geo = geo[geo["flag_low_cv"] == True]
        if not flagged_geo.empty:
            flagged_geo.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.8)

        n_flag = int(geo["flag_low_cv"].sum())
        n_eligible = int(geo["eligible"].sum())
        ax.set_title(f"{cc} — {crop} — yield CV by district\n"
                     f"Flagged low-CV (red outline, CV<{LOW_CV_THRESHOLD:.0%}): "
                     f"{n_flag}/{n_eligible} eligible districts")
        ax.axis("off")

        out_path = out_dir / f"qaqc_cv_map_{cc.lower()}_{crop}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {cc.lower()}/{out_path.name}")

        # --- relative (z-score) CV outlier map ---
        if geo["cv_zscore"].notna().sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        vmax = max(abs(geo["cv_zscore"].min()), abs(geo["cv_zscore"].max()), Z_OUTLIER_THRESHOLD)
        geo.plot(column="cv_zscore", ax=ax, cmap="coolwarm", legend=True,
                 vmin=-vmax, vmax=vmax,
                 missing_kwds={"color": "lightgrey", "label": "no data / ineligible"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "CV z-score (vs. same-crop districts)", "shrink": 0.6})

        outlier_geo = geo[(geo["flag_cv_outlier_high"] == True) | (geo["flag_cv_outlier_low"] == True)]
        if not outlier_geo.empty:
            outlier_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.0)

        n_out = int(((geo["flag_cv_outlier_high"] == True) | (geo["flag_cv_outlier_low"] == True)).sum())
        mean_cv = geo["crop_mean_cv"].dropna().iloc[0] if geo["crop_mean_cv"].notna().any() else float("nan")
        ax.set_title(f"{cc} — {crop} — yield CV relative to peers\n"
                     f"Crop mean CV={mean_cv:.3f} | outliers (black outline, |z|>{Z_OUTLIER_THRESHOLD}): "
                     f"{n_out}/{n_eligible} eligible districts")
        ax.axis("off")

        out_path = out_dir / f"qaqc_cv_zscore_map_{cc.lower()}_{crop}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {cc.lower()}/{out_path.name}")


# ---------------------------------------------------------------------------
# Choropleth: district-level anomaly rate + relative outlier flag, per crop
# ---------------------------------------------------------------------------

def plot_anomaly_maps(cc: str, district: pd.DataFrame) -> None:
    if district.empty or district["n_anomalous_years"].sum() == 0:
        return
    boundaries = gpd.read_file(get_country(cc)["boundary_path"])
    out_dir = FIG_DIR / cc.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    for crop in sorted(district["crop"].unique()):
        sub = district[(district["crop"] == crop) & (district["eligible"] == True)]
        geo = boundaries.merge(sub, on="stable_id", how="left")
        if geo["n_anomalous_years"].fillna(0).sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        geo.plot(column="n_anomalous_years", ax=ax, cmap="Reds", legend=True,
                 missing_kwds={"color": "lightgrey", "label": "no data / ineligible"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "# anomalous years (|z|>2 vs. own series mean)", "shrink": 0.6})

        n_with_any = int((geo["n_anomalous_years"].fillna(0) > 0).sum())
        n_eligible = int((geo["eligible"] == True).sum())
        ax.set_title(f"{cc} — {crop} — anomalous years by district\n"
                     f"(year's yield deviates >{Z_OUTLIER_THRESHOLD} std from that district's own series mean)\n"
                     f"{n_with_any}/{n_eligible} eligible districts have >=1 anomalous year")
        ax.axis("off")

        out_path = out_dir / f"qaqc_anomaly_map_{cc.lower()}_{crop}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {cc.lower()}/{out_path.name}")


# ---------------------------------------------------------------------------
# Choropleth: max exact-repeat run length per district, per crop
# ---------------------------------------------------------------------------

def plot_repeat_maps(cc: str, runs: pd.DataFrame, all_series: pd.DataFrame) -> None:
    if runs.empty:
        return
    boundaries = gpd.read_file(get_country(cc)["boundary_path"])
    out_dir = FIG_DIR / cc.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_runs = runs.groupby(["stable_id", "crop"])["run_length"].max().reset_index()
    eligible_pairs = all_series[["stable_id", "crop"]].drop_duplicates()

    for crop in sorted(eligible_pairs["crop"].unique()):
        crop_pairs = eligible_pairs[eligible_pairs["crop"] == crop][["stable_id"]]
        crop_runs = max_runs[max_runs["crop"] == crop]
        merged = crop_pairs.merge(crop_runs, on="stable_id", how="left")
        merged["run_length"] = merged["run_length"].fillna(0)
        if merged["run_length"].sum() == 0:
            continue

        geo = boundaries.merge(merged, on="stable_id", how="left")
        fig, ax = plt.subplots(figsize=(8, 8))
        geo.plot(column="run_length", ax=ax, cmap="Oranges", legend=True,
                 missing_kwds={"color": "lightgrey", "label": "no data"},
                 edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "Max exact-repeat run (years)", "shrink": 0.6})

        flagged_geo = geo[geo["run_length"] >= REPEAT_RUN_FLAG]
        if not flagged_geo.empty:
            flagged_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.0)

        n_flag = int((geo["run_length"] >= REPEAT_RUN_FLAG).sum())
        ax.set_title(f"{cc} — {crop} — longest exact-repeat run by district\n"
                     f"Flagged (black outline, run>={REPEAT_RUN_FLAG} yrs): {n_flag} districts")
        ax.axis("off")

        out_path = out_dir / f"qaqc_repeat_map_{cc.lower()}_{crop}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {cc.lower()}/{out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_country(cc: str) -> None:
    check_reported_vs_calculated(cc)
    summary = check_low_cv(cc)
    plot_cv_maps(cc, summary)

    all_series = pd.read_csv(PROC_DIR / f"annual_yield_{cc.lower()}.csv").dropna(subset=["yield_mt_ha"])
    events, district_anomaly = check_year_anomaly(cc)
    plot_anomaly_maps(cc, district_anomaly)

    runs = check_repeat_runs(cc)
    plot_repeat_maps(cc, runs, all_series)


def main() -> None:
    codes = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country codes: {invalid}. Choose from {all_codes()}")
    for cc in codes:
        process_country(cc)


if __name__ == "__main__":
    main()
