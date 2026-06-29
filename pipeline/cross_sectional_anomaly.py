"""
Cross-sectional anomaly check: for each calendar year, what fraction of
districts had an anomalous yield that year? Fills a blind spot named in
PROGRESS_LOG.md -- the per-district anomaly check (qaqc_yield_checks.py)
and the district drilldown tool can each show a shared-cause pattern once
you already suspect one, but neither *finds* "most districts moved the
same anomalous way this year" automatically. This check does that
directly, at the country level, across every year in the data.

Built on top of the already-computed single-year anomaly results
(qaqc_year_anomaly_<cc>.csv, qaqc_year_anomaly_district_<cc>.csv) -- same
|z|>2 threshold and same eligibility rule (>=5 reported years), not a
new/different anomaly definition. This script only re-aggregates those
results across the district dimension, year by year.

Two views are produced, since "fraction of districts" is ambiguous once
a district grows multiple crops:

  - POOLED (district-crop level): every (stable_id, crop) pair is its own
    unit. Unambiguous and exact when filtered to a single crop -- "what
    fraction of districts growing rice had an anomalous rice year in
    1988." When multiple crops are included, a 5-crop district
    contributes 5 slots, so it can pull the rate up or down depending on
    how many crops it grows -- this view answers "how widespread is this
    among crop-years," not "how many distinct places were affected."

  - DISTRICT (any-crop-per-district level): a district counts ONCE if
    AT LEAST ONE of the selected crops was anomalous that year, regardless
    of how many. This matches the plain-language framing "what fraction
    of all districts had an anomalous year" when looking across all
    crops together.

Both are split into high/low/total fractions. The denominator each year
is restricted to district-crop pairs that (a) are eligible overall
(>=5 reported years, same rule as the per-district check) AND (b) actually
reported a value in that specific year -- a district that simply didn't
report a crop that year is excluded from both numerator and denominator,
not silently counted as "not anomalous."

No automatic flagging/thresholding on the fraction itself is applied --
deliberately, to avoid inventing an arbitrary cutoff the way the deferred
yield-ceiling check would have. The summary print ranks the top years by
fraction so a researcher can judge for themselves what's notable; the
full table and the plot are there for further exploration.

Usage:
    python pipeline/cross_sectional_anomaly.py BD                     # all crops
    python pipeline/cross_sectional_anomaly.py BD --crop rice_paddy   # one crop
    python pipeline/cross_sectional_anomaly.py BD --crop rice_paddy,wheat_grain
    python pipeline/cross_sectional_anomaly.py            # all three countries, all crops

Output:
    data/processed/qaqc_cross_sectional_pooled_<cc>[_<croptag>].csv
    data/processed/qaqc_cross_sectional_district_<cc>[_<croptag>].csv
    figures/<cc>/qaqc_cross_sectional_<cc>[_<croptag>].png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PIPELINE = Path(__file__).resolve().parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from pipeline_config import all_codes

BASE     = _PIPELINE.parent
PROC_DIR = BASE / "data" / "processed"
FIG_DIR  = BASE / "figures"

MIN_YEARS_FOR_CV = 5   # same eligibility rule as qaqc_yield_checks.py


def _path(cc: str, name: str) -> Path:
    return PROC_DIR / f"{name}_{cc.lower()}.csv"


def cross_sectional_anomaly(cc: str, crops: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    yld = pd.read_csv(_path(cc, "annual_yield")).dropna(subset=["yield_mt_ha"])
    events = pd.read_csv(_path(cc, "qaqc_year_anomaly"))
    district_summary = pd.read_csv(_path(cc, "qaqc_year_anomaly_district"))

    if crops:
        yld = yld[yld["crop"].isin(crops)]
        events = events[events["crop"].isin(crops)]
        district_summary = district_summary[district_summary["crop"].isin(crops)]

    eligible_pairs = district_summary[district_summary["eligible"] == True][["stable_id", "crop"]]

    # presence: which (stable_id, crop, year) actually reported a value,
    # restricted to district-crop pairs that are eligible overall
    presence = yld.merge(eligible_pairs, on=["stable_id", "crop"])[["stable_id", "crop", "year"]].drop_duplicates()

    events_elig = events.merge(eligible_pairs, on=["stable_id", "crop"])
    events_elig = events_elig.copy()
    events_elig["direction"] = np.where(events_elig["zscore"] > 0, "high", "low")

    # --- pooled (district-crop) level ---
    pooled_rows = []
    for year, grp in presence.groupby("year"):
        n_elig = len(grp)
        yr_events = events_elig[events_elig["year"] == year]
        n_high = int((yr_events["direction"] == "high").sum())
        n_low = int((yr_events["direction"] == "low").sum())
        pooled_rows.append((
            year, n_elig, n_high, n_low, n_high + n_low,
            n_high / n_elig if n_elig else np.nan,
            n_low / n_elig if n_elig else np.nan,
            (n_high + n_low) / n_elig if n_elig else np.nan,
        ))
    pooled = pd.DataFrame(pooled_rows, columns=[
        "year", "n_eligible_pairs", "n_high", "n_low", "n_total",
        "fraction_high", "fraction_low", "fraction_total",
    ]).sort_values("year").reset_index(drop=True)

    # --- district level (any selected crop counts once per district) ---
    district_rows = []
    for year, grp in presence.groupby("year"):
        districts_present = set(grp["stable_id"].unique())
        n_elig_d = len(districts_present)
        yr_events = events_elig[events_elig["year"] == year]
        high_d = set(yr_events.loc[yr_events["direction"] == "high", "stable_id"]) & districts_present
        low_d = set(yr_events.loc[yr_events["direction"] == "low", "stable_id"]) & districts_present
        any_d = high_d | low_d
        district_rows.append((
            year, n_elig_d, len(high_d), len(low_d), len(any_d),
            len(high_d) / n_elig_d if n_elig_d else np.nan,
            len(low_d) / n_elig_d if n_elig_d else np.nan,
            len(any_d) / n_elig_d if n_elig_d else np.nan,
        ))
    district = pd.DataFrame(district_rows, columns=[
        "year", "n_eligible_districts", "n_districts_high", "n_districts_low", "n_districts_any",
        "fraction_districts_high", "fraction_districts_low", "fraction_districts_any",
    ]).sort_values("year").reset_index(drop=True)

    return pooled, district


def plot_cross_sectional(cc: str, district: pd.DataFrame, crop_tag: str) -> Path:
    out_dir = FIG_DIR / cc.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(district["year"], district["fraction_districts_high"], color="firebrick",
            marker="o", markersize=3, label="anomalously HIGH")
    ax.plot(district["year"], district["fraction_districts_low"], color="steelblue",
            marker="o", markersize=3, label="anomalously LOW")
    ax.plot(district["year"], district["fraction_districts_any"], color="black",
            linewidth=1, linestyle="--", alpha=0.6, label="any direction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of eligible districts")
    title_crop = "all crops" if crop_tag == "all" else crop_tag
    ax.set_title(f"{cc} — fraction of districts with an anomalous year ({title_crop})")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_dir / f"qaqc_cross_sectional_{cc.lower()}_{crop_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(cc: str, crops: list[str] | None) -> None:
    crop_tag = "all" if not crops else "_".join(crops)
    label = "all crops" if not crops else ", ".join(crops)
    print(f"\n=== {cc}: cross-sectional anomaly check ({label})")

    pooled, district = cross_sectional_anomaly(cc, crops)

    if district.empty:
        print("  No eligible district-crop series found for this crop selection.")
        return

    pooled_path = PROC_DIR / f"qaqc_cross_sectional_pooled_{cc.lower()}_{crop_tag}.csv"
    district_path = PROC_DIR / f"qaqc_cross_sectional_district_{cc.lower()}_{crop_tag}.csv"
    pooled.to_csv(pooled_path, index=False)
    district.to_csv(district_path, index=False)
    print(f"  Wrote {pooled_path.name}: {len(pooled)} years")
    print(f"  Wrote {district_path.name}: {len(district)} years")

    top = district.dropna(subset=["fraction_districts_any"]).sort_values(
        "fraction_districts_any", ascending=False
    ).head(5)
    print(f"  Top years by fraction of districts with ANY anomaly:")
    for _, r in top.iterrows():
        print(f"    {int(r['year'])}: {r['fraction_districts_any']:.1%} of {int(r['n_eligible_districts'])} "
              f"districts ({int(r['n_districts_high'])} high, {int(r['n_districts_low'])} low)")

    out_path = plot_cross_sectional(cc, district, crop_tag)
    print(f"  Wrote {cc.lower()}/{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("country", nargs="?", default=None,
                         help="TH, BD, or VN. Omit to run all three.")
    parser.add_argument("--crop", default=None,
                         help="Comma-separated crop slug(s) to filter to. Omit for all crops.")
    args = parser.parse_args()

    codes = [args.country.upper()] if args.country else all_codes()
    invalid = [c for c in codes if c not in all_codes()]
    if invalid:
        sys.exit(f"Unknown country code(s): {invalid}. Choose from {all_codes()}")

    crops = [c.strip() for c in args.crop.split(",")] if args.crop else None

    for cc in codes:
        run(cc, crops)


if __name__ == "__main__":
    main()
