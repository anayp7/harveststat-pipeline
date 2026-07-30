# HarvestStat Dashboard

Interactive explorer for the pipeline's outputs — SIF–yield correlations,
QA/QC flags, and crop calendars — so you don't have to dig through
`figures/` folders or raw CSVs.

## Quick start

You need **Python 3.10+**. Nothing else — no stablebound checkout, no
CSIF/CROPGRIDs downloads.

```bash
git clone https://github.com/anayp7/harveststat-pipeline.git
cd harveststat-pipeline
pip install -r requirements.txt
streamlit run dashboard/app.py
```

It opens at <http://localhost:8501>. If a browser doesn't open on its own,
paste that URL in.

<details>
<summary>Prefer an isolated environment? (recommended)</summary>

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run dashboard/app.py
```
</details>

If `pip install geopandas` gives you trouble (it pulls compiled geo
libraries), conda is usually smoother:

```bash
conda create -n harveststat python=3.11 geopandas -c conda-forge
conda activate harveststat
pip install -r requirements.txt
```

## What you're looking at

**SIF ~ Yield Correlation** — pick a crop, a season, and a scenario. The map
shows the signed Pearson *r* between detrended yield and detrended CSIF per
district. Districts with a **heavy black outline** are significant at p < 0.05;
**light grey** districts have no data for that selection and are excluded from
every statistic shown. **Click any district** for its year-by-year series:
paired anomaly bars plus the scatter and fitted line the correlation describes.

- *Season* options list the actual months averaged, and which calendar they
  came from. Sacks et al. is published and peer-reviewed; anything marked
  **researched, unverified** was compiled from literature research and its
  citations have not been independently checked — treat those with more caution.
- *Scenario* — `full` uses every reported year; `cleaned` drops QA-flagged
  points first. In `cleaned`, removed years are called out under the chart.

**QA/QC Flags** — the yield-quality checks, each with a plain-English note on
what it means, why it exists, and its threshold.

**Crop Calendar** — FAO/GIEWS-style phase bars (sowing / growing / harvesting)
per crop-season, with an optional overlay showing the exact months the pipeline
averages CSIF over.

## Where the data comes from

The dashboard reads committed outputs only:

| Path | Contents |
|---|---|
| `data/processed/` | correlation results, QA/QC flags, annual + monthly SIF |
| `data/dashboard/` | precomputed bundle (see below) |
| `data/raw/*.csv` | crop calendars |

`data/dashboard/` exists because three inputs the dashboard needs live in the
gitignored `stablebound_starter/` checkout (~40 MB, and `stats_aggregated.csv`
holds proprietary FEWS rows we don't publish). `pipeline/prepare_dashboard_bundle.py`
distills them into ~2 MB of derived files — simplified boundaries, season-level
yields, and resolved QA exclusion sets — so a clean clone just works.

If you *do* have a stablebound checkout, the dashboard automatically prefers the
bundle anyway; delete `data/dashboard/` to force it back to the live files.

**Regenerate the bundle** after re-running the pipeline:

```bash
python pipeline/prepare_dashboard_bundle.py
```

## Note on sharing

The bundle contains **derived district-level yields**. The repository is
private; keep it that way unless you've confirmed with the FEWS data owner
that derived yields can be published. Deploying this to a public host
(Streamlit Community Cloud, a tunnel) makes those values publicly reachable.
