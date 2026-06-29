"""
Standalone figure-production script.
Fits simple sensitivity models from Notebook 06 and saves plots to figures/.

Model results are cached to model_cache/ so NC files are only loaded once.
Delete the cache folder to force a full re-fit.
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from scipy import stats
from rasterio import features
from rasterio.transform import from_origin

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = r"C:\Users\anayp\Documents\HarvestStat\India Code Claude"
RAW       = os.path.join(BASE, "data", "raw")
PROC      = os.path.join(BASE, "data", "processed")
FIG       = os.path.join(BASE, "figures")
CACHE_DIR = os.path.join(BASE, "model_cache")
SM_FILE   = os.path.join(RAW, "sm_data_00_23.nc")
TX_FILE   = os.path.join(RAW, "tmax_data_00_23.nc")
YLD_FILE  = os.path.join(PROC, "icrisat_yield_area_irrig_4crops_long_2000_2017.csv")
GPKG_FILE = os.path.join(PROC, "districts_joined.gpkg")

YCOL     = "yield_kg_per_ha"
YEAR_MIN, YEAR_MAX = 2000, 2017
MIN_YRS  = 15

CROP_CALENDARS = {
    "maize":          {"planting_start": 3,  "planting_end": 7,  "harvest_start": 9,  "harvest_end": 12},
    "rice":           {"planting_start": 5,  "planting_end": 7,  "harvest_start": 9,  "harvest_end": 12},
    "sorghum_kharif": {"planting_start": 6,  "planting_end": 8,  "harvest_start": 11, "harvest_end": 12},
    "sorghum_rabi":   {"planting_start": 10, "planting_end": 11, "harvest_start": 2,  "harvest_end": 4},
}

CROP_TITLES = {"maize": "Maize", "rice": "Rice",
               "sorghum_kharif": "Sorghum Kharif",
               "sorghum_rabi":   "Sorghum Rabi"}
CROPS = list(CROP_TITLES.keys())

# ── Cache I/O ─────────────────────────────────────────────────────────────────
CACHE_FILES = {
    "lin_sm":      os.path.join(CACHE_DIR, "lin_sm.parquet"),
    "quad_sm":     os.path.join(CACHE_DIR, "quad_sm.parquet"),
    "lin_t":       os.path.join(CACHE_DIR, "lin_t.parquet"),
    "quad_t":      os.path.join(CACHE_DIR, "quad_t.parquet"),
    "climatology": os.path.join(CACHE_DIR, "climatology.parquet"),
}

def cache_exists():
    if not all(os.path.exists(p) for p in CACHE_FILES.values()):
        return False
    # Also require p-value columns (added in a later version)
    try:
        return "P_SM" in pd.read_parquet(CACHE_FILES["lin_sm"]).columns
    except Exception:
        return False

def save_cache(lin_sm, quad_sm, lin_t, quad_t, climatology):
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.concat(lin_sm.values()).to_parquet(CACHE_FILES["lin_sm"],  index=False)
    pd.concat(quad_sm.values()).to_parquet(CACHE_FILES["quad_sm"], index=False)
    pd.concat(lin_t.values()).to_parquet(CACHE_FILES["lin_t"],   index=False)
    pd.concat(quad_t.values()).to_parquet(CACHE_FILES["quad_t"],  index=False)
    pd.concat(climatology.values()).to_parquet(CACHE_FILES["climatology"], index=False)
    print(f"  Model results cached to {CACHE_DIR}/")

def load_cache():
    lin_sm     = {c: df for c, df in pd.read_parquet(CACHE_FILES["lin_sm"]).groupby("crop")}
    quad_sm    = {c: df for c, df in pd.read_parquet(CACHE_FILES["quad_sm"]).groupby("crop")}
    lin_t      = {c: df for c, df in pd.read_parquet(CACHE_FILES["lin_t"]).groupby("crop")}
    quad_t     = {c: df for c, df in pd.read_parquet(CACHE_FILES["quad_t"]).groupby("crop")}
    climatology = {c: df for c, df in pd.read_parquet(CACHE_FILES["climatology"]).groupby("crop")}
    print("  Loaded model results from cache.")
    return lin_sm, quad_sm, lin_t, quad_t, climatology

# ── Month helpers ─────────────────────────────────────────────────────────────
def get_month_range(start, end):
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 13)) + list(range(1, end + 1))

def get_middle_month(start, end):
    months = get_month_range(start, end)
    return months[len(months) // 2]

# ── Climate panel builder ─────────────────────────────────────────────────────
def _to_monthly_time(da, ds):
    dims = [d.lower() for d in da.dims]
    if "time" in dims:
        t = pd.to_datetime(da["time"].values if "time" in da.coords else ds["time"].values)
        t = pd.to_datetime([f"{d.year:04d}-{d.month:02d}-01" for d in t])
        return da.assign_coords(time=("time", t)).transpose("time", ...)
    y_dim = next(d for d in da.dims if d.lower() == "year")
    m_dim = next(d for d in da.dims if d.lower() == "month")
    years  = np.array(da[y_dim].values if y_dim in da.coords else ds[y_dim].values, int)
    months = np.arange(1, da.sizes[m_dim] + 1, dtype=int)
    time   = np.array([np.datetime64(f"{Y:04d}-{M:02d}-01") for Y in years for M in months])
    return (da.transpose(y_dim, m_dim, ...)
              .stack(time=(y_dim, m_dim))
              .assign_coords(time=("time", time))
              .transpose("time", ...))

def _load_districts():
    gdf = gpd.read_file(GPKG_FILE).dropna(subset=["Dist Code"])
    gdf = gdf.to_crs(4326).dissolve(by="Dist Code", as_index=False, aggfunc="first")
    gdf["Dist Code"] = gdf["Dist Code"].astype(int)
    return gdf

def _build_seasonal_grid(kind, months):
    if kind == "sm":
        ds = xr.open_dataset(SM_FILE);  da = ds["sm"]
    else:
        ds = xr.open_dataset(TX_FILE);  da = ds["tmax"]

    da_m = _to_monthly_time(da, ds).sel(
        time=slice(f"{YEAR_MIN-1}-01-01", f"{YEAR_MAX+1}-12-31"))

    if kind == "tmax":
        units = (da_m.attrs.get("units") or "").lower()
        if units in {"k", "kelvin"}:
            da_m = da_m - 273.15

    season_start = months[0]
    is_crossing  = any(m < season_start for m in months)
    da_sub = da_m.where(da_m["time.month"].isin(months), drop=True)

    if is_crossing:
        da_sub = da_sub.assign_coords(
            season_year=xr.where(
                da_sub["time.month"] < season_start,
                da_sub["time.year"] - 1,
                da_sub["time.year"]))
        return da_sub.groupby("season_year").mean("time").rename({"season_year": "year_dim"})
    else:
        return da_sub.groupby("time.year").mean("time").rename({"year": "year_dim"})

def _rasterize_to_panel(seasonal_grid, gdf, col_name):
    lats = seasonal_grid.lat.values
    lons = seasonal_grid.lon.values
    dx   = float(np.abs(np.diff(lons)).mean())
    dy   = float(np.abs(np.diff(lats)).mean())
    transform = from_origin(lons.min() - dx/2, lats.max() + dy/2, dx, dy)
    shapes = [(geom, int(code)) for geom, code in zip(gdf.geometry, gdf["Dist Code"])]
    mask   = features.rasterize(shapes=shapes,
                                out_shape=(len(lats), len(lons)),
                                transform=transform, fill=0, dtype="int32",
                                all_touched=True)
    if lats[0] < lats[-1]:
        mask = np.flipud(mask)
    rows = []
    for yr in [y for y in seasonal_grid.year_dim.values if YEAR_MIN <= y <= YEAR_MAX]:
        sl   = seasonal_grid.sel(year_dim=yr)
        vals = [sl.where(mask == c).mean().item() if sl.where(mask == c).notnull().any()
                else np.nan for c in gdf["Dist Code"]]
        rows.append(pd.DataFrame({"Dist Code": gdf["Dist Code"].values, "year": yr, "value": vals}))
    return pd.concat(rows, ignore_index=True).rename(columns={"value": col_name})

def get_climate_panel(months):
    print(f"  Loading climate for months {months}...")
    gdf  = _load_districts()
    sm_p = _rasterize_to_panel(_build_seasonal_grid("sm",   months), gdf, "sm_season_mean")
    tx_p = _rasterize_to_panel(_build_seasonal_grid("tmax", months), gdf, "tmax_season_mean")
    return pd.merge(sm_p, tx_p, on=["Dist Code", "year"], how="outer")

# ── P-value helper ────────────────────────────────────────────────────────────
def _coef_pvalues(X, y, m):
    """Two-sided t-test p-values for each predictor column in X."""
    n, p = X.shape
    df_res = n - p - 1          # subtract 1 for the fitted intercept
    if df_res <= 0:
        return np.full(p, np.nan)
    resid = y - m.predict(X)
    mse   = (resid ** 2).sum() / df_res
    Xb    = np.hstack([np.ones((n, 1)), X])   # add intercept column
    try:
        cov = mse * np.linalg.inv(Xb.T @ Xb)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    se    = np.sqrt(np.maximum(np.diag(cov), 0))
    coefs = np.concatenate([[m.intercept_], m.coef_])
    t_stat = np.where(se > 0, coefs / se, np.nan)
    pvals  = 2 * stats.t.sf(np.abs(t_stat), df=df_res)
    return pvals[1:]   # drop intercept; return one p-value per predictor column

# ── Model fitting ─────────────────────────────────────────────────────────────
def fit_models_for_crop(crop, yld):
    cal    = CROP_CALENDARS[crop]
    months = get_month_range(
        get_middle_month(cal["planting_start"], cal["planting_end"]),
        get_middle_month(cal["harvest_start"],  cal["harvest_end"]))

    clim  = get_climate_panel(months)
    yld_c = yld[yld["crop"] == crop].copy()

    # Drop districts where the crop is not grown (mean yield == 0)
    grown_districts = yld_c.groupby("Dist Code")[YCOL].mean()
    yld_c = yld_c[yld_c["Dist Code"].isin(grown_districts[grown_districts > 0].index)]

    # Interpolate missing irrigation data within each district, then compute pct
    yld_c["irrig_area_1000ha"] = (yld_c.groupby("Dist Code")["irrig_area_1000ha"]
                                        .transform(lambda x: x.interpolate(
                                            method="linear", limit_direction="both")))
    yld_c["irrigated_pct"] = (yld_c["irrig_area_1000ha"] /
                               yld_c["total_area_1000ha"].replace(0, np.nan)
                              ).fillna(0).clip(0, 1)

    df = pd.merge(clim, yld_c, on=["Dist Code", "year"]).dropna(
             subset=["sm_season_mean", "tmax_season_mean", YCOL])

    # District climatology: mean and std over all years per district
    clim_mean = (df.groupby("Dist Code")[["sm_season_mean", "tmax_season_mean", "irrigated_pct"]]
                   .mean().reset_index()
                   .rename(columns={"sm_season_mean":   "mean_sm",
                                    "tmax_season_mean": "mean_tmax",
                                    "irrigated_pct":    "mean_irrig_pct"}))
    clim_std  = (df.groupby("Dist Code")[["sm_season_mean", "tmax_season_mean"]]
                   .std().reset_index()
                   .rename(columns={"sm_season_mean":   "std_sm",
                                    "tmax_season_mean": "std_tmax"}))
    climatology = clim_mean.merge(clim_std, on="Dist Code", how="left")
    climatology["crop"] = crop

    lin_sm, quad_sm, lin_t, quad_t = [], [], [], []

    for code, g in df.groupby("Dist Code"):
        if len(g) < MIN_YRS:
            continue
        y = g[YCOL].values
        t = (g["year"].values - g["year"].mean()).reshape(-1, 1)

        # Linear SM
        sm = g["sm_season_mean"].values.reshape(-1, 1)
        X_lsm = np.hstack([t, sm])
        m  = LinearRegression().fit(X_lsm, y)
        pv = _coef_pvalues(X_lsm, y, m)
        lin_sm.append({"Dist Code": code, "crop": crop,
                        "Coef_SM": m.coef_[1], "P_SM": pv[1]})

        # Quadratic SM
        sm_c = sm - sm.mean();  sm_sq = sm_c ** 2
        X_qsm = np.hstack([t, sm_c, sm_sq])
        m = LinearRegression().fit(X_qsm, y)
        pv = _coef_pvalues(X_qsm, y, m)
        quad_sm.append({"Dist Code": code, "crop": crop,
                         "Coef_SM_Lin": m.coef_[1], "Coef_SM_Quad": m.coef_[2],
                         "P_SM_Lin": pv[1], "P_SM_Quad": pv[2]})

        # Linear Tmax
        tx = g["tmax_season_mean"].values.reshape(-1, 1)
        X_lt = np.hstack([t, tx])
        m  = LinearRegression().fit(X_lt, y)
        pv = _coef_pvalues(X_lt, y, m)
        lin_t.append({"Dist Code": code, "crop": crop,
                       "Coef_Temp": m.coef_[1], "P_Temp": pv[1]})

        # Quadratic Tmax
        tx_c = tx - tx.mean();  tx_sq = tx_c ** 2
        X_qt = np.hstack([t, tx_c, tx_sq])
        m = LinearRegression().fit(X_qt, y)
        pv = _coef_pvalues(X_qt, y, m)
        quad_t.append({"Dist Code": code, "crop": crop,
                        "Coef_Temp_Lin": m.coef_[1], "Coef_Temp_Quad": m.coef_[2],
                        "P_Temp_Lin": pv[1], "P_Temp_Quad": pv[2]})

    ls = pd.DataFrame(lin_sm)
    ls["Sig_SM"] = ls["P_SM"] < 0.05
    qs = pd.DataFrame(quad_sm)
    qs["Sig_SM_Lin"]  = qs["P_SM_Lin"]  < 0.05
    qs["Sig_SM_Quad"] = qs["P_SM_Quad"] < 0.05
    lt = pd.DataFrame(lin_t)
    lt["Sig_Temp"] = lt["P_Temp"] < 0.05
    qt = pd.DataFrame(quad_t)
    qt["Sig_Temp_Lin"]  = qt["P_Temp_Lin"]  < 0.05
    qt["Sig_Temp_Quad"] = qt["P_Temp_Quad"] < 0.05
    return ls, qs, lt, qt, climatology

# ── Cropping-intensity filter ─────────────────────────────────────────────────
def get_cropping_fraction(crop_name, yld, districts_gdf):
    crop_data = yld[(yld["crop"] == crop_name) & (yld["yield_kg_per_ha"] > 0)].copy()
    avg_area  = crop_data.groupby("Dist Code")["total_area_1000ha"].mean().reset_index()
    avg_area["harvested_ha"] = avg_area["total_area_1000ha"] * 1000

    dist_proj  = districts_gdf.to_crs(epsg=6933)
    dist_areas = dist_proj[["Dist Code", "geometry"]].copy()
    dist_areas["district_total_ha"] = dist_proj.geometry.area / 10000

    merged = avg_area.merge(dist_areas, on="Dist Code", how="inner")
    merged["cropping_fraction"] = (merged["harvested_ha"] / merged["district_total_ha"]).clip(upper=1.0)
    return merged[["Dist Code", "cropping_fraction"]]

# ── Shared scatter builder ────────────────────────────────────────────────────
def _scatter_panels(axes, results_by_crop, clim_by_crop, coef_col, clim_col,
                    xlabel, ylabel, yld, districts_gdf, top50=True, sig_col=None):
    """Fills a flat array of 4 axes (one per crop) with scatter panels."""
    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        df  = results_by_crop[crop]
        cli = clim_by_crop[crop]

        if df.empty or coef_col not in df.columns or clim_col not in cli.columns:
            ax.set_axis_off()
            continue

        src_cols = ["Dist Code", coef_col]
        if sig_col and sig_col in df.columns:
            src_cols.append(sig_col)
        merged = df[src_cols].merge(
                     cli[["Dist Code", clim_col]], on="Dist Code", how="inner")

        if top50:
            cf = get_cropping_fraction(crop, yld, districts_gdf)
            threshold = cf["cropping_fraction"].quantile(0.5)
            merged = merged.merge(cf, on="Dist Code", how="inner")
            merged = merged[merged["cropping_fraction"] >= threshold].copy()

        if merged.empty:
            ax.set_axis_off()
            continue

        x = merged[clim_col].values
        y = merged[coef_col].values

        if sig_col and sig_col in merged.columns:
            sig_mask = merged[sig_col].fillna(False).values
        else:
            sig_mask = np.zeros(len(x), dtype=bool)

        # Non-significant districts: small, muted grey
        ax.scatter(x[~sig_mask], y[~sig_mask], alpha=0.45, s=18,
                   color="#aaaaaa", edgecolors="none", zorder=2)
        # Significant districts: larger, red
        if sig_mask.any():
            ax.scatter(x[sig_mask], y[sig_mask], alpha=0.85, s=60,
                       color="#d7191c", edgecolors="none", zorder=3)

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)

        if len(x) > 2:
            p  = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 200)
            ax.plot(xr, np.polyval(p, xr), color="#333333", linewidth=1.8,
                    label=f"all slope={p[0]:.2f}")

        if sig_mask.sum() > 2:
            p_sig = np.polyfit(x[sig_mask], y[sig_mask], 1)
            xr_sig = np.linspace(x[sig_mask].min(), x[sig_mask].max(), 200)
            ax.plot(xr_sig, np.polyval(p_sig, xr_sig), color="#d7191c",
                    linewidth=1.8, linestyle="--", label=f"sig slope={p_sig[0]:.2f}")

        pos = int((y > 0).sum());  neg = int((y < 0).sum())
        n_sig = int(sig_mask.sum())
        ann = f"n={len(y)}  +:{pos}  −:{neg}"
        if sig_col:
            ann += f"\nsig: {n_sig}"
        ax.text(0.03, 0.97, ann,
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels, fontsize=8, loc="upper right")

        ax.set_title(CROP_TITLES[crop], fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)

def _save_fig(fig, fname):
    out = os.path.join(FIG, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Plot: coefficient maps ────────────────────────────────────────────────────
def make_coef_map(results_by_crop, coef_col, cmap, title, label, fname,
                  sequential=False, sig_col=None):
    """
    4-panel map coloring districts by a numeric column.
    sequential=False (default): diverging norm centred at 0, annotates +/- counts.
    sequential=True: linear norm over [vmin, vmax], no +/- annotation.
    """
    gdf_all   = gpd.read_file(GPKG_FILE)
    gdf_all["Dist Code"] = gdf_all["Dist Code"].astype(int)
    state_gdf = gdf_all.dissolve(by="state_shp").reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        df  = results_by_crop[crop]
        if df.empty or coef_col not in df.columns:
            ax.set_axis_off();  ax.set_title(CROP_TITLES[crop], fontsize=14, fontweight="bold")
            continue

        merge_cols = ["Dist Code", coef_col] + ([sig_col] if sig_col and sig_col in df.columns else [])
        geo  = gdf_all.merge(df[merge_cols], on="Dist Code", how="inner")
        vals = geo[coef_col].dropna()

        if sequential:
            norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
        else:
            lim  = max(np.percentile(np.abs(vals), 95), 1e-6)
            norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

        geo.plot(column=coef_col, ax=ax, cmap=cmap, norm=norm,
                 edgecolor="white", linewidth=0.15,
                 legend=(i == 0),
                 legend_kwds={"label": label, "shrink": 0.55,
                              "orientation": "horizontal", "pad": 0.02})
        state_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.4, alpha=0.5)

        if sequential:
            med = vals.median()
            ax.text(0.04, 0.04, f"Median: {med:.2f}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
        else:
            pos = int((geo[coef_col] > 0).sum());  neg = int((geo[coef_col] < 0).sum())
            if sig_col and sig_col in geo.columns:
                sig = int(geo[sig_col].fillna(False).sum())
                ann = f"Positive: {pos}\nNegative: {neg}\nSig (p<0.05): {sig}/{len(geo)}"
            else:
                ann = f"Positive: {pos}\nNegative: {neg}"
            ax.text(0.04, 0.04, ann,
                    transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

        # Bold colored borders for statistically significant districts
        if sig_col and sig_col in geo.columns:
            sig_geo = geo[geo[sig_col].fillna(False)]
            if not sig_geo.empty:
                _sig_color = "#CC0000" if cmap in ["BrBG"] else "#00008B"
                sig_geo.plot(ax=ax, color="none", edgecolor=_sig_color,
                             linewidth=1.8, zorder=4)
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor="none", edgecolor=_sig_color,
                                         linewidth=1.8, label="p < 0.05")],
                          loc="upper right", fontsize=9, framealpha=0.85)

        ax.set_title(CROP_TITLES[crop], fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Plot: sensitivity vs climatology scatter ──────────────────────────────────
def make_sensitivity_scatter(results_by_crop, clim_by_crop, coef_col, clim_col,
                             xlabel, ylabel, title, fname, yld, districts_gdf,
                             sig_col=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _scatter_panels(axes.flatten(), results_by_crop, clim_by_crop,
                    coef_col, clim_col, xlabel, ylabel, yld, districts_gdf,
                    top50=True, sig_col=sig_col)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Plot: sensitivity vs irrigation scatter ───────────────────────────────────
def make_irrigation_scatter(results_by_crop, clim_by_crop, coef_col,
                            ylabel, title, fname, yld, districts_gdf,
                            sig_col=None):
    """
    Sensitivity coefficient (y) vs district mean irrigation % (x).
    Tests whether infrastructure buffers against climate shocks.
    Top-50% cropping intensity filter applied.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _scatter_panels(axes.flatten(), results_by_crop, clim_by_crop,
                    coef_col, "mean_irrig_pct",
                    "District mean irrigation fraction (0–1)",
                    ylabel, yld, districts_gdf, top50=True, sig_col=sig_col)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Plot: binned scatter with IQR band ───────────────────────────────────────
def make_binned_scatter(results_by_crop, clim_by_crop, coef_col, clim_col,
                        xlabel, ylabel, title, fname, yld, districts_gdf, n_bins=10,
                        sig_col=None):
    """
    For each crop: faint raw dots in background, overlaid with binned median
    and Q25–Q75 shaded band. Bins are quantile-based (equal district count).
    Top-50% cropping intensity filter applied.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        df  = results_by_crop[crop]
        cli = clim_by_crop[crop]

        if df.empty or coef_col not in df.columns or clim_col not in cli.columns:
            ax.set_axis_off();  continue

        src_cols = ["Dist Code", coef_col]
        if sig_col and sig_col in df.columns:
            src_cols.append(sig_col)
        merged = df[src_cols].merge(
                     cli[["Dist Code", clim_col]], on="Dist Code", how="inner")
        cf        = get_cropping_fraction(crop, yld, districts_gdf)
        threshold = cf["cropping_fraction"].quantile(0.5)
        merged    = merged.merge(cf, on="Dist Code", how="inner")
        merged    = merged[merged["cropping_fraction"] >= threshold].copy()

        if len(merged) < n_bins * 2:
            ax.set_axis_off();  continue

        x = merged[clim_col].values
        y = merged[coef_col].values

        if sig_col and sig_col in merged.columns:
            sig_mask = merged[sig_col].fillna(False).values
        else:
            sig_mask = np.zeros(len(x), dtype=bool)

        # Raw dots (faint background): non-sig grey, sig red and larger
        ax.scatter(x[~sig_mask], y[~sig_mask], alpha=0.18, s=12,
                   color="#aaaaaa", edgecolors="none", zorder=1)
        if sig_mask.any():
            ax.scatter(x[sig_mask], y[sig_mask], alpha=0.55, s=40,
                       color="#d7191c", edgecolors="none", zorder=2)

        # Quantile bins
        merged["bin"] = pd.qcut(merged[clim_col], q=n_bins, duplicates="drop")
        binned = (merged.groupby("bin", observed=True)[coef_col]
                        .agg(median="median", q25=lambda s: s.quantile(0.25),
                             q75=lambda s: s.quantile(0.75), n="count")
                        .reset_index())
        bin_centers = binned["bin"].apply(lambda b: b.mid).values.astype(float)

        ax.fill_between(bin_centers, binned["q25"], binned["q75"],
                        alpha=0.25, color="#d7191c", zorder=2)
        ax.plot(bin_centers, binned["median"], color="#d7191c",
                linewidth=2.2, marker="o", markersize=5, zorder=3,
                label="Median (Q25–Q75 band)")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)

        # Annotate district counts per bin along the bottom
        y_min = ax.get_ylim()[0]
        for cx, n_d in zip(bin_centers, binned["n"]):
            ax.text(cx, y_min, str(int(n_d)), ha="center", va="bottom",
                    fontsize=6.5, color="gray")

        ax.legend(fontsize=9)
        ax.set_title(CROP_TITLES[crop], fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.text(0.03, 0.97, f"n={len(merged)}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Plot: 2D sensitivity heatmap (climate × irrigation) ──────────────────────
def make_sensitivity_heatmap(results_by_crop, clim_by_crop, coef_col, clim_col,
                              clim_label, coef_label, title, fname,
                              yld, districts_gdf, n_bins=5, top50=True):
    """
    2D heatmap: rows = mean climate bins, cols = irrigation fraction bins.
    Cell colour = mean sensitivity coefficient. Cells annotated with n.
    top50=True applies the cropping intensity filter; False uses all districts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    all_means = []
    crop_data = {}
    for crop in CROPS:
        df  = results_by_crop[crop]
        cli = clim_by_crop[crop]
        if df.empty or coef_col not in df.columns:
            crop_data[crop] = None;  continue

        merged = df[["Dist Code", coef_col]].merge(
                     cli[["Dist Code", clim_col, "mean_irrig_pct"]],
                     on="Dist Code", how="inner")

        if top50:
            cf        = get_cropping_fraction(crop, yld, districts_gdf)
            threshold = cf["cropping_fraction"].quantile(0.5)
            merged    = merged.merge(cf, on="Dist Code", how="inner")
            merged    = merged[merged["cropping_fraction"] >= threshold].copy()

        if len(merged) < n_bins * 2:
            crop_data[crop] = None;  continue

        merged["clim_bin"]  = pd.qcut(merged[clim_col],        q=n_bins, duplicates="drop")
        merged["irrig_bin"] = pd.qcut(merged["mean_irrig_pct"], q=n_bins, duplicates="drop")

        pivot_mean = merged.pivot_table(values=coef_col, index="clim_bin",
                                        columns="irrig_bin", aggfunc="mean")
        pivot_n    = merged.pivot_table(values=coef_col, index="clim_bin",
                                        columns="irrig_bin", aggfunc="count")
        crop_data[crop] = (pivot_mean, pivot_n)
        all_means.extend(pivot_mean.values.flatten().tolist())

    all_means = [v for v in all_means if not np.isnan(v)]
    if not all_means:
        print("  No data for heatmap.");  return
    clim_limit = max(abs(np.percentile(all_means, 5)),
                     abs(np.percentile(all_means, 95)))
    norm = mcolors.TwoSlopeNorm(vmin=-clim_limit, vcenter=0, vmax=clim_limit)
    cmap = plt.cm.RdYlGn

    for i, crop in enumerate(CROPS):
        ax = axes[i]
        if crop_data[crop] is None:
            ax.set_axis_off();  continue

        pivot_mean, pivot_n = crop_data[crop]

        # Build a numeric array aligned to sorted bin labels
        clim_labels  = [f"{b.left:.2f}–{b.right:.2f}" for b in pivot_mean.index]
        irrig_labels = [f"{b.left:.0%}–{b.right:.0%}" for b in pivot_mean.columns]

        mat = pivot_mean.values           # shape (n_clim_bins, n_irrig_bins)
        n_mat = pivot_n.values

        im = ax.imshow(mat, aspect="auto", origin="lower",
                       cmap=cmap, norm=norm)

        # Cell annotations
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                n = n_mat[r, c]
                if np.isnan(v):
                    continue
                txt_color = "white" if abs(norm(v) - 0.5) > 0.3 else "black"
                ax.text(c, r, f"{v:.0f}\n(n={int(n) if not np.isnan(n) else 0})",
                        ha="center", va="center", fontsize=7, color=txt_color,
                        fontweight="bold")

        ax.set_xticks(range(len(irrig_labels)))
        ax.set_xticklabels(irrig_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(clim_labels)))
        ax.set_yticklabels(clim_labels, fontsize=8)
        ax.set_xlabel("Mean irrigation fraction", fontsize=10)
        ax.set_ylabel(clim_label, fontsize=10)
        ax.set_title(CROP_TITLES[crop], fontsize=13, fontweight="bold")

    # Shared colourbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=coef_label)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    out = os.path.join(FIG, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Plot: vulnerability map ───────────────────────────────────────────────────
def make_vulnerability_map(results_by_crop, clim_by_crop, coef_col, std_col,
                           title, label, fname):
    """
    Maps |coefficient| × std(climate_variable) per district.
    Represents the expected yield swing from a typical annual climate shock.
    All districts included (no intensity filter).
    """
    gdf_all   = gpd.read_file(GPKG_FILE)
    gdf_all["Dist Code"] = gdf_all["Dist Code"].astype(int)
    state_gdf = gdf_all.dissolve(by="state_shp").reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        df  = results_by_crop[crop]
        cli = clim_by_crop[crop]

        if df.empty or coef_col not in df.columns or std_col not in cli.columns:
            ax.set_axis_off();  ax.set_title(CROP_TITLES[crop], fontsize=14, fontweight="bold")
            continue

        merged = df[["Dist Code", coef_col]].merge(
                     cli[["Dist Code", std_col]], on="Dist Code", how="inner")
        merged["vulnerability"] = merged[coef_col].abs() * merged[std_col]

        geo  = gdf_all.merge(merged[["Dist Code", "vulnerability"]], on="Dist Code", how="inner")
        vals = geo["vulnerability"].dropna()

        norm = mcolors.Normalize(vmin=0, vmax=np.percentile(vals, 95))

        geo.plot(column="vulnerability", ax=ax, cmap="YlOrRd", norm=norm,
                 edgecolor="white", linewidth=0.15,
                 legend=(i == 0),
                 legend_kwds={"label": label, "shrink": 0.55,
                              "orientation": "horizontal", "pad": 0.02})
        state_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.4, alpha=0.5)

        top10 = np.percentile(vals, 90)
        n_exposed = int((vals >= top10).sum())
        ax.text(0.04, 0.04, f"Top 10%: ≥{top10:.0f} kg/ha",
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
        ax.set_title(CROP_TITLES[crop], fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Plot: quadratic response curves ──────────────────────────────────────────
def make_response_curves(results_by_crop, clim_by_crop, coef_lin, coef_quad,
                         clim_col, xlabel, title, fname, yld, districts_gdf):
    """
    For each crop, draws:
      - Faint individual parabolas for every top-50% district
        (each centered on that district's own mean climatology)
      - A bold representative curve using median coefficients,
        centered on the median climatology
      - A vertical line at the predicted optimum (peak of parabola)
    Y-axis = detrended yield anomaly (kg/ha), X-axis = raw climate units.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, crop in enumerate(CROPS):
        ax  = axes[i]
        df  = results_by_crop[crop]
        cli = clim_by_crop[crop]

        if df.empty or coef_lin not in df.columns:
            ax.set_axis_off();  continue

        # Merge coefficients + climatology
        merged = (df[["Dist Code", coef_lin, coef_quad]]
                    .merge(cli[["Dist Code", clim_col]], on="Dist Code", how="inner"))

        # Top-50% filter
        cf        = get_cropping_fraction(crop, yld, districts_gdf)
        threshold = cf["cropping_fraction"].quantile(0.5)
        merged    = merged.merge(cf, on="Dist Code", how="inner")
        merged    = merged[merged["cropping_fraction"] >= threshold].copy()

        if merged.empty:
            ax.set_axis_off();  continue

        b1 = merged[coef_lin].values
        b2 = merged[coef_quad].values
        mx = merged[clim_col].values   # each district's mean X

        # X range: 5th–95th percentile of district mean climatology
        x_lo = np.percentile(mx, 5)
        x_hi = np.percentile(mx, 95)
        x_grid = np.linspace(x_lo, x_hi, 300)

        # ── Individual district curves (faint) ────────────────────────────────
        for j in range(len(b1)):
            xc = x_grid - mx[j]           # centred for this district
            y  = b1[j] * xc + b2[j] * xc**2
            ax.plot(x_grid, y, color="steelblue", alpha=0.08, linewidth=0.8)

        # ── Representative curve (median coefficients) ────────────────────────
        med_b1 = np.median(b1)
        med_b2 = np.median(b2)
        med_mx = np.median(mx)
        xc_rep = x_grid - med_mx
        y_rep  = med_b1 * xc_rep + med_b2 * xc_rep**2
        ax.plot(x_grid, y_rep, color="#d7191c", linewidth=2.5,
                label=f"Median: β₁={med_b1:.1f}, β₂={med_b2:.1f}")

        # ── Optimum marker (only meaningful when β₂ < 0) ─────────────────────
        if med_b2 < 0:
            x_opt = med_mx - med_b1 / (2 * med_b2)
            y_opt = med_b1 * (x_opt - med_mx) + med_b2 * (x_opt - med_mx)**2
            if x_lo <= x_opt <= x_hi:
                ax.axvline(x_opt, color="#d7191c", linestyle="--",
                           linewidth=1.2, alpha=0.7)
                ax.annotate(f"  optimum\n  {x_opt:.2f}",
                            xy=(x_opt, y_opt), fontsize=8.5, color="#d7191c",
                            va="top")
        elif med_b2 > 0:
            x_opt = med_mx - med_b1 / (2 * med_b2)
            if x_lo <= x_opt <= x_hi:
                ax.axvline(x_opt, color="#756bb1", linestyle="--",
                           linewidth=1.2, alpha=0.7)
                ax.annotate(f"  minimum\n  {x_opt:.2f}",
                            xy=(x_opt, 0), fontsize=8.5, color="#756bb1",
                            va="bottom")

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8.5, loc="lower center")
        ax.set_title(CROP_TITLES[crop], fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Yield anomaly from trend (kg/ha)", fontsize=10)
        ax.tick_params(labelsize=9)

        n = len(merged)
        ax.text(0.03, 0.97, f"n={n} districts", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, fname)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading yield data and shapefile...")
    yld           = pd.read_csv(YLD_FILE)
    districts_gdf = gpd.read_file(GPKG_FILE)
    districts_gdf["Dist Code"] = districts_gdf["Dist Code"].astype(int)

    # ── Fit or load from cache ────────────────────────────────────────────────
    if cache_exists():
        print("\nCache found — skipping model fitting.")
        all_lin_sm, all_quad_sm, all_lin_t, all_quad_t, all_climatology = load_cache()
    else:
        print("\nNo cache found — fitting models (this takes a few minutes)...")
        all_lin_sm, all_quad_sm, all_lin_t, all_quad_t, all_climatology = {}, {}, {}, {}, {}
        for crop in CROPS:
            print(f"\n  Fitting {crop}...")
            ls, qs, lt, qt, cli = fit_models_for_crop(crop, yld)
            all_lin_sm[crop]      = ls
            all_quad_sm[crop]     = qs
            all_lin_t[crop]       = lt
            all_quad_t[crop]      = qt
            all_climatology[crop] = cli
        save_cache(all_lin_sm, all_quad_sm, all_lin_t, all_quad_t, all_climatology)

    # ── Significance summary ──────────────────────────────────────────────────
    print("\n== Significance summary (p < 0.05) =================================")
    SIG_SPECS = [
        ("Linear SM      (Coef_SM)",       all_lin_sm,  "Sig_SM"),
        ("Quadratic SM   (linear term)",   all_quad_sm, "Sig_SM_Lin"),
        ("Quadratic SM   (quad term)",     all_quad_sm, "Sig_SM_Quad"),
        ("Linear Temp    (Coef_Temp)",     all_lin_t,   "Sig_Temp"),
        ("Quadratic Temp (linear term)",   all_quad_t,  "Sig_Temp_Lin"),
        ("Quadratic Temp (quad term)",     all_quad_t,  "Sig_Temp_Quad"),
    ]
    for model_name, data, col in SIG_SPECS:
        print(f"  {model_name}:")
        for crop in CROPS:
            df  = data[crop]
            sig = int(df[col].sum()) if col in df.columns else 0
            tot = len(df)
            pct = 100 * sig / tot if tot > 0 else 0
            print(f"    {CROP_TITLES[crop]:<20}: {sig:3d} / {tot:3d}  ({pct:.1f}%)")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # 1. Linear SM coefficient map
    make_coef_map(all_lin_sm, "Coef_SM", "BrBG",
                  "Linear SM Model: Soil Moisture Coefficient\n"
                  "(Green = positive response, Brown = negative/waterlogging)",
                  "kg/ha per unit SM", "fig1_linear_SM_coef.png",
                  sig_col="Sig_SM")

    # 2a. Quadratic SM — linear term
    make_coef_map(all_quad_sm, "Coef_SM_Lin", "BrBG",
                  "Quadratic SM Model: Linear SM Term\n"
                  "(Slope at mean soil moisture)",
                  "kg/ha per unit SM (linear)", "fig2a_quadratic_SM_linear_term.png",
                  sig_col="Sig_SM_Lin")

    # 2b. Quadratic SM — quadratic term
    make_coef_map(all_quad_sm, "Coef_SM_Quad", "RdYlGn",
                  "Quadratic SM Model: Quadratic SM Term\n"
                  "(Red = diminishing returns / waterlogging)",
                  "kg/ha per unit SM²", "fig2b_quadratic_SM_quadratic_term.png",
                  sig_col="Sig_SM_Quad")

    # 3. Linear Temp coefficient map
    make_coef_map(all_lin_t, "Coef_Temp", "RdYlGn",
                  "Linear Temp Model: Temperature Coefficient\n"
                  "(Green = heat helps, Red = heat hurts)",
                  "kg/ha per °C", "fig3_linear_Temp_coef.png",
                  sig_col="Sig_Temp")

    # 4a. Quadratic Temp — linear term
    make_coef_map(all_quad_t, "Coef_Temp_Lin", "RdYlGn",
                  "Quadratic Temp Model: Linear Temperature Term\n"
                  "(Slope at mean temperature)",
                  "kg/ha per °C (linear)", "fig4a_quadratic_Temp_linear_term.png",
                  sig_col="Sig_Temp_Lin")

    # 4b. Quadratic Temp — quadratic term
    make_coef_map(all_quad_t, "Coef_Temp_Quad", "RdYlGn",
                  "Quadratic Temp Model: Quadratic Temperature Term\n"
                  "(Red = heat cliff / accelerating damage)",
                  "kg/ha per °C²", "fig4b_quadratic_Temp_quadratic_term.png",
                  sig_col="Sig_Temp_Quad")

    # 5. Linear SM sensitivity vs mean SM
    make_sensitivity_scatter(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM", clim_col="mean_sm",
        xlabel="District mean seasonal soil moisture",
        ylabel="SM sensitivity (kg/ha per unit SM)",
        title="Linear SM Model: Sensitivity vs. District Mean Soil Moisture\n(Top 50% cropping intensity districts)",
        fname="fig5_linear_SM_sensitivity_vs_climatology.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_SM")

    # 6. Linear Temp sensitivity vs mean Tmax
    make_sensitivity_scatter(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp", clim_col="mean_tmax",
        xlabel="District mean seasonal max temperature (°C)",
        ylabel="Temperature sensitivity (kg/ha per °C)",
        title="Linear Temp Model: Sensitivity vs. District Mean Temperature\n(Top 50% cropping intensity districts)",
        fname="fig6_linear_Temp_sensitivity_vs_climatology.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_Temp")

    # 7. Linear SM sensitivity vs irrigation — does irrigation buffer moisture shocks?
    make_irrigation_scatter(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM",
        ylabel="SM sensitivity (kg/ha per unit SM)",
        title="Does Irrigation Buffer Soil Moisture Shocks?\n"
              "Linear SM Model — Sensitivity vs. District Mean Irrigation\n"
              "(Top 50% cropping intensity districts)",
        fname="fig7_SM_sensitivity_vs_irrigation.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_SM")

    # 8. Linear Temp sensitivity vs irrigation — does irrigation buffer heat shocks?
    make_irrigation_scatter(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp",
        ylabel="Temperature sensitivity (kg/ha per °C)",
        title="Does Irrigation Buffer Temperature Shocks?\n"
              "Linear Temp Model — Sensitivity vs. District Mean Irrigation\n"
              "(Top 50% cropping intensity districts)",
        fname="fig8_Temp_sensitivity_vs_irrigation.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_Temp")

    # 9. Quadratic SM — response curves (Goldilocks)
    make_response_curves(
        all_quad_sm, all_climatology,
        coef_lin="Coef_SM_Lin", coef_quad="Coef_SM_Quad",
        clim_col="mean_sm",
        xlabel="Seasonal soil moisture",
        title="Soil Moisture Response Curves (Quadratic Model)\n"
              "Red = median district; faint lines = individual top-50% districts",
        fname="fig9_quadratic_SM_response_curves.png",
        yld=yld, districts_gdf=districts_gdf)

    # 10. Quadratic Temp — response curves (Heat Cliff)
    make_response_curves(
        all_quad_t, all_climatology,
        coef_lin="Coef_Temp_Lin", coef_quad="Coef_Temp_Quad",
        clim_col="mean_tmax",
        xlabel="Seasonal max temperature (°C)",
        title="Temperature Response Curves (Quadratic Model)\n"
              "Red = median district; faint lines = individual top-50% districts",
        fname="fig10_quadratic_Temp_response_curves.png",
        yld=yld, districts_gdf=districts_gdf)

    # 12–15. Binned scatter versions of figs 5–8
    make_binned_scatter(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM", clim_col="mean_sm",
        xlabel="District mean seasonal soil moisture",
        ylabel="SM sensitivity (kg/ha per unit SM)",
        title="Linear SM: Sensitivity vs. Mean Soil Moisture (Binned)\n(Top 50% districts — median ± IQR per decile)",
        fname="fig12_binned_SM_sensitivity_vs_climatology.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_SM")

    make_binned_scatter(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp", clim_col="mean_tmax",
        xlabel="District mean seasonal max temperature (°C)",
        ylabel="Temperature sensitivity (kg/ha per °C)",
        title="Linear Temp: Sensitivity vs. Mean Temperature (Binned)\n(Top 50% districts — median ± IQR per decile)",
        fname="fig13_binned_Temp_sensitivity_vs_climatology.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_Temp")

    make_binned_scatter(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM", clim_col="mean_irrig_pct",
        xlabel="District mean irrigation fraction (0–1)",
        ylabel="SM sensitivity (kg/ha per unit SM)",
        title="Does Irrigation Buffer Soil Moisture Shocks? (Binned)\n(Top 50% districts — median ± IQR per decile)",
        fname="fig14_binned_SM_sensitivity_vs_irrigation.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_SM")

    make_binned_scatter(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp", clim_col="mean_irrig_pct",
        xlabel="District mean irrigation fraction (0–1)",
        ylabel="Temperature sensitivity (kg/ha per °C)",
        title="Does Irrigation Buffer Temperature Shocks? (Binned)\n(Top 50% districts — median ± IQR per decile)",
        fname="fig15_binned_Temp_sensitivity_vs_irrigation.png",
        yld=yld, districts_gdf=districts_gdf, sig_col="Sig_Temp")

    # 16–17. 2D sensitivity heatmaps (climate × irrigation, 5×5)
    make_sensitivity_heatmap(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM", clim_col="mean_sm",
        clim_label="Mean seasonal soil moisture (bin)",
        coef_label="Mean SM sensitivity (kg/ha per unit SM)",
        title="SM Sensitivity across Soil Moisture × Irrigation (4×4 grid)\n(All districts — cell colour = mean Coef_SM)",
        fname="fig16_heatmap_SM_sensitivity.png",
        yld=yld, districts_gdf=districts_gdf, n_bins=4, top50=False)

    make_sensitivity_heatmap(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp", clim_col="mean_tmax",
        clim_label="Mean seasonal max temperature (bin)",
        coef_label="Mean Temp sensitivity (kg/ha per °C)",
        title="Temp Sensitivity across Temperature × Irrigation (4×4 grid)\n(All districts — cell colour = mean Coef_Temp)",
        fname="fig17_heatmap_Temp_sensitivity.png",
        yld=yld, districts_gdf=districts_gdf, n_bins=4, top50=False)

    # 18–19. Vulnerability maps: |coef| × std(climate variable)
    make_vulnerability_map(
        all_lin_sm, all_climatology,
        coef_col="Coef_SM", std_col="std_sm",
        title="Soil Moisture Vulnerability: |Coef_SM| × σ(SM)\n"
              "Expected yield swing from a typical annual moisture shock (kg/ha)",
        label="|Coef_SM| × σ(SM)  [kg/ha]",
        fname="fig18_vulnerability_SM.png")

    make_vulnerability_map(
        all_lin_t, all_climatology,
        coef_col="Coef_Temp", std_col="std_tmax",
        title="Temperature Vulnerability: |Coef_Temp| × σ(Tmax)\n"
              "Expected yield swing from a typical annual temperature shock (kg/ha)",
        label="|Coef_Temp| × σ(Tmax)  [kg/ha]",
        fname="fig19_vulnerability_Temp.png")

    # 11. Irrigation fraction map
    make_coef_map(
        {crop: all_climatology[crop].rename(columns={"mean_irrig_pct": "irrig_pct"})
         for crop in CROPS},
        coef_col="irrig_pct",
        cmap="YlGnBu",
        title="Mean Irrigation Fraction by District (2000–2017)\n(fraction of harvested area that is irrigated)",
        label="Irrigation fraction (0–1)",
        fname="fig11_irrigation_fraction_map.png",
        sequential=True)

    print("\nAll figures saved to figures/")
