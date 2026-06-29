# FILE: 05_Symbolic_Discovery.ipynb
# NEW CELL: Grand Model Comparison (District-Level Refitting)

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("--- Final Model Showdown: Hierarchy vs. Parsimony ---")
print("    Metric: Mean Partial R² (Variance Explained EXCLUDING Trend)")
print("    Method: District-Level Refitting (Local Coefficients)")

# --- 0. Setup ---
if 'yld' not in locals():
    yld = pd.read_csv("data/processed/icrisat_yield_area_irrig_4crops_long_2000_2017.csv")

if 'districts_gdf' not in locals():
    districts_gdf = gpd.read_file("data/processed/districts_joined.gpkg")

# Helper for intensity
def get_cropping_fraction(crop_name):
    crop_data = yld[yld['crop'] == crop_name].copy()
    avg_area = crop_data.groupby('Dist Code')['total_area_1000ha'].mean().reset_index()
    avg_area['harvested_ha'] = avg_area['total_area_1000ha'] * 1000

    dist_proj = districts_gdf.to_crs(epsg=6933)
    dist_areas = dist_proj[['Dist Code', 'geometry']].copy()
    dist_areas['district_total_ha'] = dist_proj.geometry.area / 10000

    merged = avg_area.merge(dist_areas, on='Dist Code', how='inner')
    merged['cropping_fraction'] = merged['harvested_ha'] / merged['district_total_ha']
    merged['cropping_fraction'] = merged['cropping_fraction'].clip(upper=1.0)
    return merged[['Dist Code', 'cropping_fraction']]

# --- 1. Load Hierarchical Results ---
# (Adjust paths if necessary)
FILE_LIN   = "results/FINAL_HIERARCHICAL_metrics.parquet"
FILE_QUAD  = "results/FINAL_HIERARCHICAL_QUADRATIC_metrics.parquet"
FILE_INTER = "results/FINAL_HIERARCHICAL_INTERACTION_metrics.parquet"

try:
    df_lin   = pd.read_parquet(FILE_LIN)
    df_quad  = pd.read_parquet(FILE_QUAD)
    df_inter = pd.read_parquet(FILE_INTER)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")

# Filter Shift 0
df_lin = df_lin[df_lin['year_shift'] == 0].copy()
df_quad = df_quad[df_quad['year_shift'] == 0].copy()
df_inter = df_inter[df_inter['year_shift'] == 0].copy()

# Calculate Partials
df_lin['Partial_Linear'] = df_lin['R2_4_Full'] - df_lin['R2_1_Trend']
df_quad['Partial_Quad']  = df_quad['R2_4_Full'] - df_quad['R2_1_Trend']
df_inter['Partial_Inter'] = df_inter['R2_4_Full'] - df_inter['R2_1_Trend']

# --- 2. Calculate SR (Simple) Model Metrics ---
sr_results = []
crops = ['maize', 'rice', 'sorghum_kharif', 'sorghum_rabi']

print("Fitting 'Simple' SR Models per district...")

for crop in crops:
    # Prepare Data
    calendar = CROP_CALENDARS[crop]
    months = get_month_range(get_middle_month(calendar['planting_start'], calendar['planting_end']),
                             get_middle_month(calendar['harvest_start'], calendar['harvest_end']))

    clim = get_climate_panel_for_months_NO_WEIGHTING(months)

    yld_s = yld[yld["crop"] == crop].copy()
    yld_s['year'] -= 0

    # Irrig
    yld_s['irrig_area_1000ha'] = yld_s.groupby('Dist Code')['irrig_area_1000ha'].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    yld_s['total_area_temp'] = yld_s['total_area_1000ha'].replace(0, np.nan)
    yld_s['irrigated_pct'] = (yld_s['irrig_area_1000ha'] / yld_s['total_area_temp']).fillna(0).clip(0,1)

    df = pd.merge(clim, yld_s, on=["Dist Code", "year"]).dropna()

    for code, g in df.groupby('Dist Code'):
        if len(g) < 15: continue

        y = g[YCOL].values
        t = (g['year'] - g['year'].mean()).values.reshape(-1, 1)

        # Baseline Trend
        r2_trend = LinearRegression().fit(t, y).score(t, y)

        # --- DEFINING THE SIMPLE MODELS (Based on PySR) ---
        if crop == 'maize':
            # Model: Quadratic SM (Trend + SM + SM^2)
            sm = g['sm_season_mean'].values
            sm_sq = (sm - sm.mean())**2
            X = np.column_stack([t, sm, sm_sq])

        elif crop == 'rice':
            # Model: Soil Moisture Only (Trend + SM)
            # (PySR detrended analysis favored SM over Irrig for anomalies)
            sm = g['sm_season_mean'].values
            X = np.column_stack([t, sm])

        elif crop == 'sorghum_kharif':
            # Model: Interaction (Trend + SM + T + SM*T)
            # We include main effects + interaction to allow the district to tune the sensitivity
            sm = g['sm_season_mean'].values
            temp = g['tmax_season_mean'].values
            inter = (sm - sm.mean()) * (temp - temp.mean())
            X = np.column_stack([t, sm, temp, inter])

        elif crop == 'sorghum_rabi':
            # Model: Additive Water (Trend + SM + Irrig)
            sm = g['sm_season_mean'].values
            irrig = g['irrigated_pct'].values
            X = np.column_stack([t, sm, irrig])

        r2_full = LinearRegression().fit(X, y).score(X, y)
        sr_results.append({'Dist Code': code, 'crop': crop, 'Partial_SR': r2_full - r2_trend})

df_sr = pd.DataFrame(sr_results)

# --- 3. Merge ---
# Inner join to ensure exact sample match
merged = pd.merge(df_lin[['Dist Code', 'crop', 'Partial_Linear']],
                  df_quad[['Dist Code', 'crop', 'Partial_Quad']],
                  on=['Dist Code', 'crop'], how='inner')
merged = pd.merge(merged, df_inter[['Dist Code', 'crop', 'Partial_Inter']], on=['Dist Code', 'crop'], how='inner')
merged = pd.merge(merged, df_sr[['Dist Code', 'crop', 'Partial_SR']], on=['Dist Code', 'crop'], how='inner')

# --- 4. Filtering (Top 50%) ---
cf_list = []
for crop in crops:
    cf = get_cropping_fraction(crop)
    cf['crop'] = crop
    cf_list.append(cf)
cf_all = pd.concat(cf_list)

final_df = pd.merge(merged, cf_all, on=['Dist Code', 'crop'], how='left')
final_df['cropping_fraction'] = final_df['cropping_fraction'].fillna(0)

top_districts = final_df.groupby('crop').apply(lambda x: x[x['cropping_fraction'] >= x['cropping_fraction'].quantile(0.5)]).reset_index(drop=True)

# --- 5. Display ---
def make_pretty(df):
    return df.style.background_gradient(cmap='Greens', axis=1).format("{:.3f}")

print("\n=== COMPARISON: Mean Partial R² (Top 50% Districts) ===")
print("    Col 1-3: Hierarchical Models (Kitchen Sink)")
print("    Col 4:   Symbolic Models (Parsimonious)")

table = top_districts.groupby('crop')[['Partial_Linear', 'Partial_Quad', 'Partial_Inter', 'Partial_SR']].mean()
table.columns = ['1. Linear', '2. +Quadratic', '3. +Interaction', '4. SR (Simple)']
display(make_pretty(table))