# FILE: 06_Driver_Sensitivities.ipynb
# CELL 1: Setup & Environment

print("--- Initializing Driver Sensitivity Analysis (Square One) ---")
%pip install pandas geopandas rasterio shapely fiona pyproj xarray netCDF4 matplotlib seaborn -q

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd "/content/drive/MyDrive/IndiaProject_Colab"

# Load Project Functions
%run 00_Functions_and_Setup.ipynb

# Load Master Data & Shapefile
if 'yld' not in locals():
    yld = pd.read_csv("data/processed/icrisat_yield_area_irrig_4crops_long_2000_2017.csv")
if 'districts_gdf' not in locals():
    districts_gdf = gpd.read_file("data/processed/districts_joined.gpkg")
    state_boundaries_gdf = districts_gdf.dissolve(by='state_shp').reset_index()

# Define Core Production Filter
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

print("[✓] Environment Ready.")