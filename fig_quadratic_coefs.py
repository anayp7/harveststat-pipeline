"""
Four-quadrant scatter plots for the quadratic sensitivity models from produce_figures.py.
  X-axis : linear term coefficient   (slope at the district mean)
  Y-axis : quadratic term coefficient (curvature)

  Quadrant labels
  ---------------
  Q1 (Lin+, Quad+)  – accelerating gain     : positive & steepening above mean
  Q2 (Lin-, Quad+)  – U-shaped / unusual    : hurt at mean but recovers at extremes
  Q3 (Lin-, Quad-)  – concave negative      : optimum below mean, penalised above
  Q4 (Lin+, Quad-)  – inverted-U / optimum  : optimum above mean, penalised at extremes

Saves:
  figures/fig_quadratic_coefs_SM.png
  figures/fig_quadratic_coefs_Temp.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

BASE       = r"C:\Users\anayp\Documents\HarvestStat\India Code Claude"
CACHE_DIR  = os.path.join(BASE, "model_cache")
FIG        = os.path.join(BASE, "figures")

CROPS = ["maize", "rice", "sorghum_kharif", "sorghum_rabi"]
CROP_TITLES = {
    "maize":          "Maize",
    "rice":           "Rice",
    "sorghum_kharif": "Sorghum Kharif",
    "sorghum_rabi":   "Sorghum Rabi",
}

QUADRANT_LABELS = {
    "Q1": ("Lin+, Quad+",  "accelerating\ngain",   0.97, 0.97, "right", "top"),
    "Q2": ("Lin-, Quad+",  "U-shaped /\nunusual",  0.03, 0.97, "left",  "top"),
    "Q3": ("Lin-, Quad-",  "concave\nnegative",    0.03, 0.03, "left",  "bottom"),
    "Q4": ("Lin+, Quad-",  "inverted-U /\noptimum",0.97, 0.03, "right", "bottom"),
}

# ── Load cached results ───────────────────────────────────────────────────────
print("Loading model cache...")
quad_sm = pd.read_parquet(os.path.join(CACHE_DIR, "quad_sm.parquet"))
quad_t  = pd.read_parquet(os.path.join(CACHE_DIR, "quad_t.parquet"))
print(f"  SM:   {len(quad_sm)} district-crop rows")
print(f"  Temp: {len(quad_t)} district-crop rows")


# ── Shared plot function ──────────────────────────────────────────────────────
def make_quad_plot(df, lin_col, quad_col, xlabel, ylabel, title, fname,
                   lin_unit="", quad_unit="",
                   sig_lin_col=None, sig_quad_col=None):

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)

    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        sub = df[df["crop"] == crop].dropna(subset=[lin_col, quad_col])

        x = sub[lin_col].values
        y = sub[quad_col].values

        # Build significance mask: significant if either linear or quadratic term p<0.05
        sig_mask = np.zeros(len(x), dtype=bool)
        if sig_lin_col and sig_lin_col in sub.columns:
            sig_mask |= sub[sig_lin_col].fillna(False).values
        if sig_quad_col and sig_quad_col in sub.columns:
            sig_mask |= sub[sig_quad_col].fillna(False).values

        # Clip axes to 2–98th percentile to handle outliers
        xlo, xhi = np.percentile(x, 2), np.percentile(x, 98)
        ylo, yhi = np.percentile(y, 2), np.percentile(y, 98)
        xpad = (xhi - xlo) * 0.12
        ypad = (yhi - ylo) * 0.12
        xlim = (xlo - xpad, xhi + xpad)
        ylim = (ylo - ypad, yhi + ypad)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Zero-axis cross-hairs
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)

        # Scatter: non-significant grey, significant orange-red and larger
        ax.scatter(x[~sig_mask], y[~sig_mask], s=18, alpha=0.45,
                   color="#888888", edgecolors="none", zorder=2)
        if sig_mask.any():
            ax.scatter(x[sig_mask], y[sig_mask], s=55, alpha=0.85,
                       color="#d7191c", edgecolors="none", zorder=3)

        # Count only points within the displayed axis range
        vis = ((x >= xlim[0]) & (x <= xlim[1]) &
               (y >= ylim[0]) & (y <= ylim[1]))
        xv, yv = x[vis], y[vis]
        n_q1 = int(((xv > 0) & (yv > 0)).sum())
        n_q2 = int(((xv < 0) & (yv > 0)).sum())
        n_q3 = int(((xv < 0) & (yv < 0)).sum())
        n_q4 = int(((xv > 0) & (yv < 0)).sum())
        n_hidden = len(x) - vis.sum()

        kw = dict(fontsize=9, transform=ax.transAxes,
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.text(0.97, 0.97, f"n={n_q1}", ha="right", va="top",    color="black", **kw)
        ax.text(0.03, 0.97, f"n={n_q2}", ha="left",  va="top",    color="black", **kw)
        ax.text(0.03, 0.03, f"n={n_q3}", ha="left",  va="bottom", color="black", **kw)
        ax.text(0.97, 0.03, f"n={n_q4}", ha="right", va="bottom", color="black", **kw)

        # Quadrant interpretation labels (just inside axes, near crosshairs)
        mid_kw = dict(fontsize=8, style="italic", color="#555555", transform=ax.transAxes)
        ax.text(0.73, 0.55, "accel. gain",       ha="right", va="bottom", **mid_kw)
        ax.text(0.27, 0.55, "U-shaped",          ha="left",  va="bottom", **mid_kw)
        ax.text(0.27, 0.45, "concave neg.",       ha="left",  va="top",   **mid_kw)
        ax.text(0.73, 0.45, "inverted-U",         ha="right", va="top",   **mid_kw)

        # Hidden point note
        if n_hidden > 0:
            ax.text(0.5, 0.01, f"{n_hidden} outlier(s) outside axis range",
                    ha="center", va="bottom", fontsize=7.5, color="#888888",
                    transform=ax.transAxes)

        # Legend for significant points
        if sig_mask.any():
            n_sig = int(sig_mask.sum())
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                       markersize=5, label="not sig."),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7191c",
                       markersize=7, label=f"sig. (p<0.05): {n_sig}"),
            ]
            ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
                      framealpha=0.8, edgecolor="none")

        ax.set_title(CROP_TITLES[crop], fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel if i == 0 else "", fontsize=10)
        ax.tick_params(labelsize=8)

        # Axis formatting: use scientific notation if values are large
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2g}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2g}"))

    plt.tight_layout()
    out = os.path.join(FIG, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 1: Soil Moisture ───────────────────────────────────────────────────
make_quad_plot(
    df        = quad_sm,
    lin_col   = "Coef_SM_Lin",
    quad_col  = "Coef_SM_Quad",
    xlabel    = "Linear term  (kg/ha per unit SM, at district mean)",
    ylabel    = "Quadratic term  (kg/ha per unit SM²)",
    title     = ("Quadratic Soil-Moisture Model — Linear vs Quadratic Coefficients\n"
                 "Each point = one district  |  centered SM  |  year-detrended yield"),
    fname     = "fig_quadratic_coefs_SM.png",
    sig_lin_col  = "Sig_SM_Lin",
    sig_quad_col = "Sig_SM_Quad",
)

# ── Figure 2: Temperature ─────────────────────────────────────────────────────
make_quad_plot(
    df        = quad_t,
    lin_col   = "Coef_Temp_Lin",
    quad_col  = "Coef_Temp_Quad",
    xlabel    = "Linear term  (kg/ha per °C, at district mean)",
    ylabel    = "Quadratic term  (kg/ha per °C²)",
    title     = ("Quadratic Temperature Model — Linear vs Quadratic Coefficients\n"
                 "Each point = one district  |  centered Tmax  |  year-detrended yield"),
    fname     = "fig_quadratic_coefs_Temp.png",
    sig_lin_col  = "Sig_Temp_Lin",
    sig_quad_col = "Sig_Temp_Quad",
)
