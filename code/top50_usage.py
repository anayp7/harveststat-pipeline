# FILE: 04_Diagnostic_Plots.ipynb
# RE-RUN CELL: Winning Shift Maps (Updated Legend)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("--- Generating Winning Shift Maps (Final Legend Logic) ---")

# (Data prep remains the same)
results_df['Total_Partial_R2'] = results_df['R2_4_Full'] - results_df['R2_1_Trend']
idx = results_df.groupby(['Dist Code', 'crop'])['Total_Partial_R2'].idxmax()
best_shift_df = results_df.loc[idx].copy()

crops = ['maize', 'rice', 'sorghum_kharif', 'sorghum_rabi']
titles = ["Maize", "Rice", "Sorghum Kharif", "Sorghum Rabi"]

color_map = {
    -1: '#440154', # Purple
     0: '#21918c', # Teal
     1: '#fde725'  # Yellow
}

# Load Boundaries
if 'state_boundaries_gdf' not in locals():
    state_boundaries_gdf = districts_gdf.dissolve(by='state_shp').reset_index()

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
rows = ["All Districts", "Top 50% Cropping Intensity"]

for col_idx, crop in enumerate(crops):
    crop_results = best_shift_df[best_shift_df['crop'] == crop]
    geo_data = districts_gdf.merge(crop_results, on='Dist Code', how='inner')

    # Calculate Intensity
    cf_df = get_cropping_fraction(crop)
    geo_data = geo_data.merge(cf_df, on='Dist Code', how='left')
    geo_data['cropping_fraction'] = geo_data['cropping_fraction'].fillna(0)
    threshold = geo_data['cropping_fraction'].quantile(0.5)

    # --- ROW 1 ---
    ax1 = axes[0, col_idx]
    geo_data.plot(color=geo_data['year_shift'].map(color_map), ax=ax1)
    state_boundaries_gdf.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)
    ax1.set_axis_off()
    ax1.set_title(f"{titles[col_idx]}", fontsize=14, fontweight='bold')

    # --- ROW 2 ---
    ax2 = axes[1, col_idx]
    top_50 = geo_data[geo_data['cropping_fraction'] >= threshold]
    bottom_50 = geo_data[geo_data['cropping_fraction'] < threshold]

    if not bottom_50.empty:
        bottom_50.plot(ax=ax2, color='#e0e0e0', edgecolor='white', linewidth=0.1)
    if not top_50.empty:
        top_50.plot(color=top_50['year_shift'].map(color_map), ax=ax2)

    state_boundaries_gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)
    ax2.set_axis_off()

# Row Labels
for r, label in enumerate(rows):
    axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes,
                    rotation=90, va='center', fontsize=16, fontweight='bold')

# --- UPDATED LEGEND ---
legend_patches = [
    mpatches.Patch(color=color_map[-1], label='Shift -1 (Future/Next Season)'),
    mpatches.Patch(color=color_map[0],  label='Shift 0 (Concurrent/Standard)'),
    mpatches.Patch(color=color_map[1],  label='Shift 1 (Past/Lagged)')
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 0.02))

plt.suptitle("Winning Year Shift by Crop: Validating the Data Timeline", fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.show()