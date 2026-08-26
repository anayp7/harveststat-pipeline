# HarvestStat Pipeline

A reproducible pipeline for crop yield QA/QC and remote-sensing variance
analysis. Built to work with any country where FEWS-format yield data and
GAUL administrative boundaries are available.

Currently configured for **Thailand (TH)**, **Bangladesh (BD)**,
**Vietnam (VN)**, and **India (IN)**. Adding a new country requires editing
one YAML file — no code changes.

---

## What it does

Five sequential steps, each a standalone script:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `stablebound` (external) | Raw FEWS CSV + boundary GeoJSONs | Stable-boundary yield stats |
| 1 | `extract_csif.py` | `csif.zip` (CSIF all-sky) | `csif_<cc>.nc` per country |
| 2 | `aggregate_sif_by_crop.py` | CSIF NetCDF + CROPGRIDs + boundaries | `sif_by_crop_<cc>.csv` |
| 3 | `aggregate_annual.py` | Stablebound stats + SIF by crop + crop calendar | `annual_yield/sif/joined_<cc>.csv` |
| 4 | `qaqc_yield_checks.py` | Annual yield | QA/QC CSVs + choropleth maps |
| 5 | `sif_yield_variance.py` | Annual joined + QA flags | Pearson r maps, correlation CSV |

Two diagnostic tools that can be run at any point after Step 4:

- **`cross_sectional_anomaly.py`** — for each year, what fraction of districts had an anomalous yield? Filterable by crop.
- **`district_drilldown.py`** — deep-dive on a single district: CV flags, anomalous years with neighboring context, cross-crop check, mismatch rows, SIF-yield correlation stats.

---

## Repository layout

```
pipeline/           all pipeline scripts
  countries.yaml    per-country config — the only file to edit for a new country
  pipeline_config.py config loader (imported by all scripts)
  extract_csif.py
  aggregate_sif_by_crop.py
  aggregate_annual.py
  qaqc_yield_checks.py
  cross_sectional_anomaly.py
  district_drilldown.py
  sif_yield_variance.py
  PROGRESS_LOG.md   running record of decisions, findings, and bugs fixed

code/               India analysis notebooks (exploratory, predates this pipeline)
data/
  raw/              source data (large files excluded from git — see below)
    cropgrids/      CROPGRIDs NetCDF files
  processed/        all computed outputs (CSVs committed; NetCDFs excluded)
figures/
  bd/ th/ vn/       QA/QC maps and SIF-yield correlation maps per country
  india_exploratory/ India SM/temperature sensitivity figures
results/            India model output parquets
```

---

## Data sources (not included in this repo)

| File | Source | Notes |
|------|--------|-------|
| `csif.zip` | Zhang et al. (2018), Science Advances | ~16 annual NetCDFs; all-sky CSIF, 0.5 deg, 2001–2016 |
| `All_data_with_climate.csv` | Sacks et al. crop calendar database | Growing season planting/harvest dates by country and crop |
| `CROPGRIDSv1.08_NC_maps.zip` | [Figshare](https://figshare.com/articles/dataset/CROPGRIDs/21074736) | 770 MB; crop area NetCDFs |
| `stablebound_starter/` | Team member / FEWS | FEWS yield CSVs + GAUL boundaries; excluded (proprietary) |
| `stablebound_IN_package.zip` | Team member / DESAGRI | India: statistics, district shapefile, and a stablebound wheel. Extract into `stablebound_starter/`. Excluded (proprietary) |
| India NetCDFs (`data/raw/*.nc`) | ERA5 / CSIF | Excluded (too large) |

Stablebound outputs (`_out/stable/`) must be regenerated locally by running
the `stablebound` package — see `stablebound_starter/stablebound_TH_BD_VN_starter/README.md`.

---

## Setup

```bash
# Create and activate a Python environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install numpy pandas geopandas xarray rasterio scipy matplotlib pyyaml netCDF4
```

Stablebound requires its own separate environment — follow the instructions
in the stablebound README before running Step 0.

---

## Running the pipeline

All scripts are run from the **project root** (the folder containing this README).

```bash
# Step 1 — extract CSIF SIF for each country (clips to bbox, aggregates 4-day composites to monthly)
python pipeline/extract_csif.py             # all countries
python pipeline/extract_csif.py TH BD      # specific countries

# Step 2 — aggregate SIF to districts, weighted by CROPGRIDs crop area
python pipeline/aggregate_sif_by_crop.py

# Step 3 — collapse multi-season stats to one annual yield + SIF row per district-crop
python pipeline/aggregate_annual.py

# Step 4 — yield QA/QC checks (outputs CSVs + choropleth maps)
python pipeline/qaqc_yield_checks.py
python pipeline/qaqc_yield_checks.py BD    # one country

# Step 5 — SIF-yield Pearson r analysis (detrended, full vs. QA-cleaned)
python pipeline/sif_yield_variance.py
```

**Diagnostic tools** (run after Step 4):

```bash
# Cross-sectional anomaly — fraction of districts anomalous per year
python pipeline/cross_sectional_anomaly.py BD
python pipeline/cross_sectional_anomaly.py BD --crop rice_paddy
python pipeline/cross_sectional_anomaly.py BD --crop rice_paddy,wheat_grain

# District drilldown — full diagnostic for one district
python pipeline/district_drilldown.py BD BD.ADM2.00012
```

---

## Adding a new country

1. Obtain the FEWS yield CSV and GAUL boundary GeoJSONs for the country.
2. Run the `stablebound` package to build stable-boundary outputs.
3. Add one block to `pipeline/countries.yaml`:

```yaml
ET:                                 # two-letter code
  name: Ethiopia
  bbox: [33.0, 3.0, 48.0, 15.0]   # [lon_min, lat_min, lon_max, lat_max] with ~1° buffer
  stablebound_dir: stablebound_starter/ethiopia
  raw_csv: stablebound_starter/ethiopia/et_standard.csv
  base_year: 1990                  # stablebound lineage base year
  crop_map:
    teff: teff
    maize_corn: maize
    wheat_grain: wheat
  admin_name_aliases:              # omit if no spelling mismatches
    Addis Abeba: Addis Ababa
```

4. Re-run the pipeline scripts. All scripts pick up the new country code automatically — no Python changes needed.

---

## QA/QC checks (Step 4)

Five checks, all operating on the annual yield series before any remote-sensing data enters:

1. **Reported vs. calculated mismatch** — flags rows where the source CSV's reported yield and its computed production/area differ by >10%.
2. **Low coefficient of variation** — flags district-crop series with CV < 5% (suspiciously flat; likely copy-forward data entry).
3. **Relative CV outlier** — flags districts whose CV is >2 std above or below the same-crop mean across all districts.
4. **Single-year anomaly** — within each district-crop series, flags years where yield deviates >2 std from the series' own mean.
5. **Exact-repeat run** — flags runs of ≥3 consecutive years with an identical yield value; hard-flags at ≥4 years.

See `pipeline/PROGRESS_LOG.md` for the full reasoning behind each check, key findings per country, and known limitations.

---

## SIF-yield analysis (Step 5)

For each (district, crop) pair with ≥5 overlapping years of yield and SIF data:

- Both series are **linearly detrended** before correlating (removes shared secular trends from yield-technology and SIF-landuse drift).
- **Signed Pearson r** is reported alongside R² and p-value — direction matters and can be negative.
- Two scenarios side by side: **full** (all reported years) and **cleaned** (QA-flagged points and series removed).
- **Growing-season-matched SIF**: monthly SIF is averaged over only the active growing season months for each crop (from `data/raw/All_data_with_climate.csv`). Crops with calendar entries: rice (all months due to multi-season coverage), wheat (Nov–Apr), maize/TH (Apr–Sep), sugarcane (all months). Crops without entries (soybean, groundnut, cassava) use the 12-month mean.

QA exclusions applied for "cleaned": single-year anomalies at |z| > 3, exact-repeat runs of length ≥ 4, reported-vs-calculated mismatches > 10%, and absolute low-CV series. See `pipeline/PROGRESS_LOG.md` Section 9 for the rationale.

---

## Key findings so far

- **Vietnam sugarcane_for_sugar** shows a strong, often statistically significant *negative* r in southern Vietnam (mean r ≈ −0.32 full / −0.20 cleaned; 13–14 of 25 districts significant at p < 0.05). Direction matters — R² alone would have hidden this.
- **Rice shows weak/near-zero correlations in all three countries**, likely because SIF is aggregated over a flat 12-month window rather than a crop-specific growing season. Growing-season-matched SIF aggregation is the natural next step.
- **Bangladesh 2021** shows a near-country-wide maize yield collapse (>90% of districts anomalous in the same year, all low), followed by a sustained elevated-anomaly regime through 2024 — consistent with COVID-era surveying disruption and/or a methodology change.
