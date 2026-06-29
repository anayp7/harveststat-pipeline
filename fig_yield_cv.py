"""
Produce figure: yield CV (std/mean) per district by crop.
Flags districts with suspiciously low year-to-year variability.
Saves to figures/fig_yield_cv.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = r"C:\Users\anayp\Documents\HarvestStat\India Code Claude"
PROC      = os.path.join(BASE, "data", "processed")
FIG       = os.path.join(BASE, "figures")
YLD_FILE  = os.path.join(PROC, "icrisat_yield_area_irrig_4crops_long_2000_2017.csv")
GPKG_FILE = os.path.join(PROC, "districts_joined.gpkg")

CROPS = ["maize", "rice", "sorghum_kharif", "sorghum_rabi"]
CROP_TITLES = {
    "maize":          "Maize",
    "rice":           "Rice",
    "sorghum_kharif": "Sorghum Kharif",
    "sorghum_rabi":   "Sorghum Rabi",
}
LOW_CV_THRESHOLD = 0.05   # flag districts with CV below this

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
yld = pd.read_csv(YLD_FILE)
districts_gdf = gpd.read_file(GPKG_FILE)
state_boundaries_gdf = districts_gdf.dissolve(by="state_norm").reset_index()

# ── Compute CV per district-crop ──────────────────────────────────────────────
print("Computing CV...")
cv_df = (
    yld.groupby(["Dist Code", "crop"])["yield_kg_per_ha"]
    .agg(yield_std="std", yield_mean="mean", n_years="count")
    .reset_index()
)
cv_df["yield_cv"]   = cv_df["yield_std"] / cv_df["yield_mean"]
cv_df["low_cv_flag"] = cv_df["yield_cv"] < LOW_CV_THRESHOLD

print("Flagged districts (CV < {:.0%}):".format(LOW_CV_THRESHOLD))
for crop in CROPS:
    n = cv_df[(cv_df["crop"] == crop) & cv_df["low_cv_flag"]]["Dist Code"].nunique()
    print(f"  {crop}: {n}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 8))

for i, crop in enumerate(CROPS):
    ax = axes[i]
    data     = cv_df[cv_df["crop"] == crop].copy()
    geo_data = districts_gdf.merge(data, on="Dist Code", how="inner")

    vals = geo_data["yield_cv"].dropna()
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=vals.quantile(0.95))

    geo_data.plot(
        column="yield_cv",
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        edgecolor="white", linewidth=0.15,
        legend=(i == 0),
        legend_kwds={"label": "CV  (std / mean yield)", "shrink": 0.55,
                     "orientation": "horizontal", "pad": 0.02},
        missing_kwds={"color": "#cccccc"},
    )
    flagged_geo = geo_data[geo_data["low_cv_flag"]]
    if not flagged_geo.empty:
        flagged_geo.plot(ax=ax, color="blue", edgecolor="blue", linewidth=0.5, alpha=0.7)

    state_boundaries_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.4, alpha=0.5)
    ax.set_title(CROP_TITLES[crop], fontsize=16, fontweight="bold")
    ax.set_axis_off()

    med_cv = vals.median()
    n_flag = geo_data["low_cv_flag"].sum()
    ax.text(0.04, 0.04,
            f"Median CV: {med_cv:.2f}\nFlagged (blue): {n_flag}",
            transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

plt.suptitle(
    "Yield Variability by District  (CV = std / mean across years)\n"
    f"Blue = suspiciously constant  (CV < {LOW_CV_THRESHOLD:.0%})",
    fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out = os.path.join(FIG, "fig_yield_cv.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
